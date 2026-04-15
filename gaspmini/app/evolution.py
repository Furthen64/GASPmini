# app/evolution.py
# Between-epoch evolution: fitness scoring, selection, crossover, mutation.

from __future__ import annotations

import random
import copy
from typing import Optional

import app.config as config
from app.models import (
    CellType, ActionType, Creature, Genome, Gene, GenePattern,
    CreatureEpochResult, WorldState,
)
from app.world import make_random_genome
from app.logging_utils import debug_log, log


# ── Fitness ───────────────────────────────────────────────────────────────────

def compute_fitness(creature: Creature) -> float:
    lt = creature.lifetime
    stationary_age_ticks = lt.stationary_ticks * config.FITNESS_STATIONARY_AGE_FACTOR
    active_age_ticks = max(0.0, (lt.age_ticks - lt.stationary_ticks) + stationary_age_ticks)
    return (
        config.FITNESS_FOOD_WEIGHT * lt.food_eaten
        + config.FITNESS_AGE_WEIGHT * active_age_ticks
        + config.FITNESS_FAILED_WEIGHT * lt.failed_actions
    )


def collect_epoch_results(world: WorldState) -> list[CreatureEpochResult]:
    results = []
    for c in world.creatures:
        fitness = compute_fitness(c)
        results.append(CreatureEpochResult(
            creature_id=c.creature_id,
            food_eaten=c.lifetime.food_eaten,
            age_ticks=c.lifetime.age_ticks,
            failed_actions=c.lifetime.failed_actions,
            fitness=fitness,
        ))
    results.sort(key=lambda r: r.fitness, reverse=True)
    return results


# ── Crossover ─────────────────────────────────────────────────────────────────

def _crossover_genomes(parent_a: Genome, parent_b: Genome, rng: random.Random) -> Genome:
    """Simple crossover: child takes some genes from each parent."""
    genes_a = parent_a.genes
    genes_b = parent_b.genes
    max_len = max(len(genes_a), len(genes_b))
    child_genes: list[Gene] = []

    for i in range(max_len):
        if i < len(genes_a) and i < len(genes_b):
            gene = copy.deepcopy(genes_a[i] if rng.random() < 0.5 else genes_b[i])
        elif i < len(genes_a):
            gene = copy.deepcopy(genes_a[i])
        else:
            gene = copy.deepcopy(genes_b[i])
        gene.gene_id = i  # re-index
        child_genes.append(gene)

    # Inherit learning parameters from one parent or averaged
    if rng.random() < 0.5:
        lr = parent_a.learning_rate
        rd = parent_a.reward_decay
        er = parent_a.exploration_rate
        hl = parent_a.history_length
    else:
        lr = (parent_a.learning_rate + parent_b.learning_rate) / 2
        rd = (parent_a.reward_decay + parent_b.reward_decay) / 2
        er = (parent_a.exploration_rate + parent_b.exploration_rate) / 2
        hl = parent_a.history_length

    return Genome(
        genes=child_genes,
        learning_rate=lr,
        reward_decay=rd,
        exploration_rate=er,
        history_length=hl,
    )


# ── Mutation ──────────────────────────────────────────────────────────────────

def _mutate_gene_pattern(pattern: GenePattern, rng: random.Random) -> GenePattern:
    """Randomly alter exactly one field of the gene pattern."""
    cell_choices = list(CellType) + [None]
    action_choices = list(ActionType) + [None]
    bool_choices = [True, False, None]

    field = rng.choice([
        'current_cell', 'front_cell', 'left_cell', 'right_cell', 'back_cell',
        'last_action', 'last_action_success', 'hunger_bucket',
    ])

    # Build a plain dict of the current values, then change one field.
    vals = {
        'current_cell':        pattern.current_cell,
        'front_cell':          pattern.front_cell,
        'left_cell':           pattern.left_cell,
        'right_cell':          pattern.right_cell,
        'back_cell':           pattern.back_cell,
        'last_action':         pattern.last_action,
        'last_action_success': pattern.last_action_success,
        'hunger_bucket':       pattern.hunger_bucket,
    }

    if field in ('current_cell', 'front_cell', 'left_cell', 'right_cell', 'back_cell'):
        vals[field] = rng.choice(cell_choices)
    elif field == 'last_action':
        vals[field] = rng.choice(action_choices)
    elif field == 'last_action_success':
        vals[field] = rng.choice(bool_choices)
    else:  # hunger_bucket
        vals[field] = rng.choice([0, 1, 2, 3, None])

    return GenePattern(**vals)


def mutate_genome(genome: Genome, rng: random.Random) -> Genome:
    """Return a mutated copy of the genome.  Do not modify the original."""
    genome = copy.deepcopy(genome)

    # Mutate individual genes
    for gene in genome.genes:
        if rng.random() < config.MUTATION_RATE:
            mutation = rng.choice(['pattern', 'action', 'priority'])
            if mutation == 'pattern':
                gene.pattern = _mutate_gene_pattern(gene.pattern, rng)
            elif mutation == 'action':
                gene.action = rng.choice(list(ActionType))
            elif mutation == 'priority':
                gene.base_priority += rng.uniform(-config.PARAM_MUTATION_DELTA * 4, config.PARAM_MUTATION_DELTA * 4)

    # Occasionally add or remove a gene
    if rng.random() < 0.05 and len(genome.genes) < 20:
        new_id = max((g.gene_id for g in genome.genes), default=-1) + 1
        new_gene = Gene(
            gene_id=new_id,
            pattern=GenePattern(None, None, None, None, None, None, None, None),
            action=rng.choice(list(ActionType)),
            base_priority=0.0,
        )
        genome.genes.append(new_gene)
        if config.DEBUG_EVOLUTION:
            debug_log(f"Mutation: added gene {new_id}")
    elif rng.random() < 0.05 and len(genome.genes) > 3:
        removed = genome.genes.pop(rng.randrange(len(genome.genes)))
        if config.DEBUG_EVOLUTION:
            debug_log(f"Mutation: removed gene {removed.gene_id}")

    # Mutate genome-level float parameters
    if rng.random() < config.PARAM_MUTATION_RATE:
        genome.learning_rate = max(0.01, genome.learning_rate + rng.uniform(-config.PARAM_MUTATION_DELTA, config.PARAM_MUTATION_DELTA))
    if rng.random() < config.PARAM_MUTATION_RATE:
        genome.reward_decay = max(0.1, min(0.99, genome.reward_decay + rng.uniform(-config.PARAM_MUTATION_DELTA, config.PARAM_MUTATION_DELTA)))
    if rng.random() < config.PARAM_MUTATION_RATE:
        genome.exploration_rate = max(0.0, min(0.5, genome.exploration_rate + rng.uniform(-config.PARAM_MUTATION_DELTA, config.PARAM_MUTATION_DELTA)))

    return genome


# ── Next generation ───────────────────────────────────────────────────────────

def evolve_next_generation(
    world: WorldState,
    results: list[CreatureEpochResult],
    rng: random.Random,
) -> list[Genome]:
    """
    Build the genome list for the next epoch.
    Lifetime state is NOT carried over; only genomes.
    """
    population_size = len(world.creatures)
    if population_size == 0:
        return [make_random_genome(rng) for _ in range(10)]

    n_elite = max(1, round(population_size * config.ELITE_FRACTION))
    n_random = max(1, round(population_size * config.RANDOM_NEW_FRACTION))
    n_offspring = population_size - n_elite - n_random

    # Map creature_id → genome
    genome_map = {c.creature_id: c.genome for c in world.creatures}

    # Sorted results (already sorted descending by fitness)
    elite_ids = [r.creature_id for r in results[:n_elite]]
    elite_genomes = [copy.deepcopy(genome_map[cid]) for cid in elite_ids]

    if config.DEBUG_EVOLUTION:
        top = results[0] if results else None
        fitness_str = f"top_fitness={top.fitness:.2f}" if top else "no results"
        log(
            f"Evolution: pop={population_size}  elites={n_elite}  "
            f"offspring={n_offspring}  random={n_random}  {fitness_str}"
        )

    # Offspring: crossover from elites
    offspring_genomes: list[Genome] = []
    for _ in range(n_offspring):
        pa = rng.choice(elite_genomes)
        pb = rng.choice(elite_genomes)
        child = _crossover_genomes(pa, pb, rng)
        child = mutate_genome(child, rng)
        offspring_genomes.append(child)

    # Fresh random creatures
    random_genomes = [make_random_genome(rng, gene_count=config.INITIAL_GENE_COUNT) for _ in range(n_random)]

    next_gen = elite_genomes + offspring_genomes + random_genomes
    # Re-assign gene ids to be sequential within each genome
    for genome in next_gen:
        for i, gene in enumerate(genome.genes):
            gene.gene_id = i

    return next_gen
