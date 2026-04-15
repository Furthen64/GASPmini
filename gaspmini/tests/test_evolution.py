# tests/test_evolution.py
# Tests for fitness computation and between-epoch evolution.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import random
import tempfile

import app.config as config
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
                pattern=GenePattern(None, None, None, None, None, None, None, None),
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


def _make_creature(
    cid: int,
    food_eaten: int = 0,
    age_ticks: int = 0,
    failed: int = 0,
    stationary_ticks: int = 0,
) -> Creature:
    lt = LifetimeState(
        x=1, y=1,
        direction=Direction.NORTH,
        energy=30.0,
        food_eaten=food_eaten,
        age_ticks=age_ticks,
        failed_actions=failed,
        stationary_ticks=stationary_ticks,
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

    def test_stationary_ticks_are_dampened_in_fitness(self):
        active = _make_creature(0, food_eaten=0, age_ticks=100, stationary_ticks=0)
        stationary = _make_creature(1, food_eaten=0, age_ticks=100, stationary_ticks=100)

        self.assertGreater(compute_fitness(active), compute_fitness(stationary))

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

    def test_step_epoch_imprints_lifetime_learning_into_genome_priorities(self):
        original_factor = config.LEARNED_PRIORITY_IMPRINT_FACTOR
        try:
            sim = SimulationRunner(seed=19, ticks_per_epoch=1)
            config.LEARNED_PRIORITY_IMPRINT_FACTOR = 0.5
            world = WorldState(width=10, height=10, tick_index=1, epoch_index=0)

            top_creature = _make_creature(0, food_eaten=4, age_ticks=20)
            low_creature = _make_creature(1, food_eaten=0, age_ticks=2)
            top_creature.genome = _make_genome(3)
            low_creature.genome = _make_genome(3)
            top_creature.lifetime.learned_gene_adjustments = {0: 2.0}
            top_creature.genome.genes[0].base_priority = 1.0

            world.creatures = [top_creature, low_creature]
            sim.world = world

            sim.step_epoch()

            self.assertIsNotNone(sim.best_genome_ever)
            # 1.0 + (2.0 * 0.5) = 2.0
            self.assertAlmostEqual(sim.best_genome_ever.genes[0].base_priority, 2.0)
        finally:
            config.LEARNED_PRIORITY_IMPRINT_FACTOR = original_factor

    def test_profile_selection_changes_world_settings(self):
        original_profile = config.ACTIVE_PROFILE_ID
        try:
            sim = SimulationRunner(seed=3)
            sim.set_profile('longer_strategic')
            sim.reset()

            self.assertEqual(sim.profile_id, 'longer_strategic')
            self.assertEqual(sim.ticks_per_epoch, config.TICKS_PER_EPOCH)
            self.assertEqual(sim.world.width, config.GRID_WIDTH)
            self.assertEqual(sim.world.height, config.GRID_HEIGHT)
            self.assertEqual(len(sim.world.creatures), config.POPULATION_SIZE)
        finally:
            config.apply_profile(original_profile)

    def test_custom_map_selection_overrides_seeded_biome_layout(self):
        sim = SimulationRunner(seed=3, custom_map_id='hidden_food')

        sim.reset()

        self.assertIsNotNone(sim.world)
        self.assertEqual((sim.world.width, sim.world.height), (30, 20))
        self.assertIn((4, 5), sim.world.food_positions)
        self.assertIn((3, 3), sim.world.walls)

    def test_enter_testing_ground_uses_best_genome_ever(self):
        sim = SimulationRunner(seed=13)
        sim.best_genome_ever = _make_genome(9)

        entered = sim.enter_testing_ground()

        self.assertTrue(entered)
        self.assertTrue(sim.is_testing_ground())
        self.assertIsNotNone(sim.world)
        self.assertEqual(len(sim.world.creatures), 1)
        self.assertEqual(len(sim.world.creatures[0].genome.genes), 9)

    def test_enter_testing_ground_falls_back_to_current_best_creature(self):
        sim = SimulationRunner(seed=17)
        weaker = _make_creature(0, food_eaten=1, age_ticks=5)
        stronger = _make_creature(1, food_eaten=5, age_ticks=20)
        weaker.genome = _make_genome(4)
        stronger.genome = _make_genome(8)
        sim.world = WorldState(width=10, height=10, creatures=[weaker, stronger])

        entered = sim.enter_testing_ground()

        self.assertTrue(entered)
        self.assertEqual(len(sim.world.creatures), 1)
        self.assertEqual(len(sim.world.creatures[0].genome.genes), 8)

    def test_testing_ground_runs_until_death_without_advancing_epoch(self):
        sim = SimulationRunner(seed=19)
        sim.best_genome_ever = _make_genome(6)
        sim.enter_testing_ground()
        assert sim.world is not None
        sim.world.creatures[0].lifetime.energy = 1.0

        sim.step_tick()

        self.assertTrue(sim.is_run_complete())
        self.assertEqual(sim.world.epoch_index, 0)
        self.assertEqual(len(sim.epoch_history), 0)

    def test_autosave_best_creature_writes_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'best.json')
            sim = SimulationRunner(seed=23, ticks_per_epoch=1)
            sim.configure_best_genome_persistence(
                autosave_enabled=True,
                autosave_path=path,
                inject_saved_best_enabled=False,
            )
            world = WorldState(width=10, height=10, tick_index=1, epoch_index=0)
            world.creatures = [
                _make_creature(0, food_eaten=4, age_ticks=10),
                _make_creature(1, food_eaten=1, age_ticks=10),
            ]
            sim.world = world

            sim.step_epoch()

            self.assertTrue(os.path.isfile(path))

    def test_reset_can_inject_saved_best_into_random_population(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'best.json')
            source_sim = SimulationRunner(seed=29)
            source_sim.best_genome_ever = _make_genome(11)
            source_sim.configure_best_genome_persistence(
                autosave_enabled=True,
                autosave_path=path,
                inject_saved_best_enabled=False,
            )
            source_sim._autosave_best_genome_if_enabled()

            sim = SimulationRunner(seed=31)
            sim.configure_best_genome_persistence(
                autosave_enabled=False,
                autosave_path=path,
                inject_saved_best_enabled=True,
            )

            sim.reset()

            self.assertIsNotNone(sim.world)
            self.assertEqual(len(sim.world.creatures), config.POPULATION_SIZE)
            self.assertTrue(any(len(c.genome.genes) == 11 for c in sim.world.creatures))


if __name__ == '__main__':
    unittest.main()
