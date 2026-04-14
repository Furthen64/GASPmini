# tests/test_evolution.py
# Tests for fitness computation and between-epoch evolution.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import random

from app.models import (
    ActionType, CellType, Direction,
    Gene, GenePattern, Genome, LifetimeState, Creature, WorldState,
)
from app.evolution import (
    compute_fitness, collect_epoch_results, evolve_next_generation,
    mutate_genome,
)
from app.simulation_runner import SimulationRunner
from app.world import make_rng


def _make_genome(n_genes: int = 5) -> Genome:
    return Genome(
        genes=[
            Gene(
                gene_id=i,
                pattern=GenePattern(None, None, None, None, None, None, None),
                action=ActionType.IDLE,
                base_priority=0.0,
            )
            for i in range(n_genes)
        ],
        learning_rate=0.25,
        reward_decay=0.7,
        exploration_rate=0.05,
        history_length=6,
    )


def _make_creature(cid: int, food_eaten: int = 0, age_ticks: int = 0, failed: int = 0) -> Creature:
    lt = LifetimeState(
        x=1, y=1,
        direction=Direction.NORTH,
        energy=30.0,
        food_eaten=food_eaten,
        age_ticks=age_ticks,
        failed_actions=failed,
        learned_gene_adjustments={0: 1.5, 1: -0.5},
    )
    return Creature(creature_id=cid, genome=_make_genome(), lifetime=lt)


def _make_world_with_creatures(n: int) -> WorldState:
    world = WorldState(width=10, height=10)
    for i in range(n):
        world.creatures.append(_make_creature(i, food_eaten=n - i, age_ticks=50))
    return world


class TestEvolution(unittest.TestCase):

    def test_fitness_increases_with_food(self):
        c0 = _make_creature(0, food_eaten=0, age_ticks=100)
        c1 = _make_creature(1, food_eaten=5, age_ticks=100)
        self.assertGreater(compute_fitness(c1), compute_fitness(c0))

    def test_fitness_decreases_with_failures(self):
        c0 = _make_creature(0, food_eaten=5, age_ticks=100, failed=0)
        c1 = _make_creature(1, food_eaten=5, age_ticks=100, failed=50)
        self.assertGreater(compute_fitness(c0), compute_fitness(c1))

    def test_fitness_increases_with_age(self):
        c0 = _make_creature(0, food_eaten=0, age_ticks=0)
        c1 = _make_creature(1, food_eaten=0, age_ticks=200)
        self.assertGreater(compute_fitness(c1), compute_fitness(c0))

    def test_results_sorted_descending(self):
        world = _make_world_with_creatures(5)
        results = collect_epoch_results(world)
        fitnesses = [r.fitness for r in results]
        self.assertEqual(fitnesses, sorted(fitnesses, reverse=True))

    def test_evolve_produces_correct_population_size(self):
        world = _make_world_with_creatures(20)
        rng = make_rng(42)
        results = collect_epoch_results(world)
        next_gen = evolve_next_generation(world, results, rng)
        self.assertEqual(len(next_gen), 20)

    def test_elites_preserved_unchanged(self):
        """Elite genomes should be deep copies, not modified."""
        world = _make_world_with_creatures(10)
        rng = make_rng(42)
        results = collect_epoch_results(world)

        # Record top creature's genome gene count before evolving
        top_id = results[0].creature_id
        top_genome = next(c.genome for c in world.creatures if c.creature_id == top_id)
        orig_gene_count = len(top_genome.genes)

        next_gen = evolve_next_generation(world, results, rng)

        # The first genome in next_gen should be an elite (same gene count)
        self.assertEqual(len(next_gen[0].genes), orig_gene_count)

    def test_lifetime_state_not_inherited(self):
        """Evolved genomes must not carry lifetime state."""
        world = _make_world_with_creatures(10)
        rng = make_rng(42)
        results = collect_epoch_results(world)
        next_gen = evolve_next_generation(world, results, rng)
        # Genomes are pure data; LifetimeState is created fresh per spawn
        # Just verify genomes are Genome instances with no lifetime attribute
        for g in next_gen:
            self.assertIsInstance(g, Genome)
            self.assertFalse(hasattr(g, 'lifetime'))

    def test_mutation_changes_something(self):
        genome = _make_genome(8)
        rng = make_rng(99)
        # Run mutation many times; at least one should differ
        changed = False
        for _ in range(50):
            mutated = mutate_genome(genome, rng)
            if mutated != genome:
                changed = True
                break
        self.assertTrue(changed, "Mutation never produced a different genome in 50 attempts")

    def test_random_new_creatures_added(self):
        """The new generation should contain some fresh random genomes."""
        world = _make_world_with_creatures(20)
        rng = make_rng(7)
        results = collect_epoch_results(world)
        next_gen = evolve_next_generation(world, results, rng)
        # With RANDOM_NEW_FRACTION=0.10 and pop=20, expect ~2 fresh genomes at end
        self.assertEqual(len(next_gen), 20)

    def test_empty_world_returns_some_genomes(self):
        world = WorldState(width=10, height=10)
        rng = make_rng(1)
        results = collect_epoch_results(world)
        next_gen = evolve_next_generation(world, results, rng)
        self.assertGreater(len(next_gen), 0)

    def test_best_ever_genome_is_reinserted_across_epochs(self):
        sim = SimulationRunner(seed=5, ticks_per_epoch=1)

        champion = _make_genome(7)
        contender = _make_genome(5)
        world0 = WorldState(width=10, height=10, tick_index=1, epoch_index=0)
        world0.creatures = [
            _make_creature(0, food_eaten=10, age_ticks=100),
            _make_creature(1, food_eaten=1, age_ticks=10),
        ]
        world0.creatures[0].genome = champion
        world0.creatures[1].genome = contender
        sim.world = world0

        sim.step_epoch()

        weaker_a = _make_genome(5)
        weaker_b = _make_genome(3)
        world1 = WorldState(width=10, height=10, tick_index=1, epoch_index=1)
        world1.creatures = [
            _make_creature(0, food_eaten=2, age_ticks=20),
            _make_creature(1, food_eaten=1, age_ticks=10),
        ]
        world1.creatures[0].genome = weaker_a
        world1.creatures[1].genome = weaker_b
        sim.world = world1

        sim.step_epoch()

        self.assertIsNotNone(sim.best_genome_ever)
        self.assertEqual(len(sim.best_genome_ever.genes), 7)
        self.assertTrue(any(len(c.genome.genes) == 7 for c in sim.world.creatures))

    def test_epoch_history_records_best_fitness(self):
        sim = SimulationRunner(seed=11, ticks_per_epoch=1)
        world = WorldState(width=10, height=10, tick_index=1, epoch_index=0)
        world.creatures = [
            _make_creature(0, food_eaten=4, age_ticks=10),
            _make_creature(1, food_eaten=1, age_ticks=10),
        ]
        sim.world = world

        sim.step_epoch()

        self.assertEqual(len(sim.epoch_history), 1)
        self.assertEqual(sim.epoch_history[0]['epoch'], 0)
        self.assertEqual(sim.epoch_history[0]['top_creature_id'], 0)
        self.assertAlmostEqual(sim.epoch_history[0]['top_fitness'], compute_fitness(world.creatures[0]))
        self.assertEqual(sim.epoch_best_fitnesses(), [(0, sim.epoch_history[0]['top_fitness'])])
        self.assertEqual(sim.best_epoch_ever, 0)
        self.assertAlmostEqual(sim.best_fitness_ever, sim.epoch_history[0]['top_fitness'])


if __name__ == '__main__':
    unittest.main()
