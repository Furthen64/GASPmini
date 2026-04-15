# tests/test_learning.py
# Tests for within-epoch reward propagation.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from app.models import (
    ActionType, CellType,
    Gene, GenePattern, Genome, LifetimeState, Creature, SensorField, HistoryEntry,
)
from app.learning import apply_reward_to_history, record_history
from app.gene_logic import state_adjustment_for_gene


def _make_creature(
    learning_rate: float = 0.25,
    reward_decay: float = 0.7,
    history_length: int = 6,
) -> Creature:
    genome = Genome(
        genes=[
            Gene(
                gene_id=i,
                pattern=GenePattern(None, None, None, None, None, None, None, None),
                action=ActionType.IDLE,
                base_priority=0.0,
            )
            for i in range(5)
        ],
        learning_rate=learning_rate,
        reward_decay=reward_decay,
        history_length=history_length,
    )
    lifetime = LifetimeState(
        x=5, y=5,
        energy=30.0,
    )
    return Creature(creature_id=0, genome=genome, lifetime=lifetime)


def _dummy_sensor() -> SensorField:
    return SensorField(
        current_cell=CellType.EMPTY,
        north_cell=CellType.EMPTY,
        east_cell=CellType.EMPTY,
        south_cell=CellType.EMPTY,
        west_cell=CellType.EMPTY,
        last_action=ActionType.IDLE,
        last_action_success=True,
        hunger_bucket=0,
    )


def _add_history(creature: Creature, gene_id: int, tick: int, reward: float = 0.0) -> None:
    entry = HistoryEntry(
        sensor=_dummy_sensor(),
        gene_id=gene_id,
        action=ActionType.IDLE,
        reward=reward,
        action_success=True,
        tick_index=tick,
    )
    record_history(creature, entry)


def _food_sensor() -> SensorField:
    return SensorField(
        current_cell=CellType.EMPTY,
        north_cell=CellType.FOOD,
        east_cell=CellType.EMPTY,
        south_cell=CellType.EMPTY,
        west_cell=CellType.EMPTY,
        last_action=ActionType.IDLE,
        last_action_success=True,
        hunger_bucket=0,
    )


class TestLearning(unittest.TestCase):

    def test_single_reward_propagates_to_most_recent_gene(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=0.7)
        _add_history(creature, gene_id=2, tick=0)
        apply_reward_to_history(creature, reward=10.0)
        adj = creature.lifetime.learned_gene_adjustments.get(2, 0.0)
        # 10.0 * 0.7^0 * 1.0 = 10.0
        self.assertAlmostEqual(adj, 10.0)

    def test_reward_decays_with_steps_back(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=0.5)
        _add_history(creature, gene_id=0, tick=0)
        _add_history(creature, gene_id=1, tick=1)
        _add_history(creature, gene_id=2, tick=2)
        apply_reward_to_history(creature, reward=8.0)

        adj_2 = creature.lifetime.learned_gene_adjustments.get(2, 0.0)
        adj_1 = creature.lifetime.learned_gene_adjustments.get(1, 0.0)
        adj_0 = creature.lifetime.learned_gene_adjustments.get(0, 0.0)

        # step 0 back: gene 2 → 8.0 * 0.5^0 = 8.0
        self.assertAlmostEqual(adj_2, 8.0)
        # step 1 back: gene 1 → 8.0 * 0.5^1 = 4.0
        self.assertAlmostEqual(adj_1, 4.0)
        # step 2 back: gene 0 → 8.0 * 0.5^2 = 2.0
        self.assertAlmostEqual(adj_0, 2.0)

    def test_negative_reward_decreases_adjustment(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=1.0)
        _add_history(creature, gene_id=3, tick=0)
        apply_reward_to_history(creature, reward=-5.0)
        adj = creature.lifetime.learned_gene_adjustments.get(3, 0.0)
        self.assertAlmostEqual(adj, -5.0)

    def test_adjustments_are_cumulative(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=1.0)
        _add_history(creature, gene_id=0, tick=0)
        apply_reward_to_history(creature, reward=3.0)
        _add_history(creature, gene_id=0, tick=1)
        apply_reward_to_history(creature, reward=4.0)
        adj = creature.lifetime.learned_gene_adjustments.get(0, 0.0)
        # First reward: 3.0 propagated to gene 0 (steps_back=0)
        # Second reward: 4.0 propagated to gene 0 (most recent) + gene 0 again (steps_back=1)
        # = 3.0 + 4.0 + 4.0 = 11.0
        self.assertAlmostEqual(adj, 11.0)

    def test_only_recent_history_adjusted_with_max_steps(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=1.0)
        for tick in range(5):
            _add_history(creature, gene_id=tick, tick=tick)

        apply_reward_to_history(creature, reward=1.0, max_steps_back=2)

        # Only the last 2 entries (gene 4 and gene 3) should be adjusted
        self.assertAlmostEqual(creature.lifetime.learned_gene_adjustments.get(4, 0.0), 1.0)
        self.assertAlmostEqual(creature.lifetime.learned_gene_adjustments.get(3, 0.0), 1.0)
        self.assertEqual(creature.lifetime.learned_gene_adjustments.get(2, 0.0), 0.0)
        self.assertEqual(creature.lifetime.learned_gene_adjustments.get(1, 0.0), 0.0)
        self.assertEqual(creature.lifetime.learned_gene_adjustments.get(0, 0.0), 0.0)

    def test_learned_adjustments_reset_on_new_lifetime(self):
        """Adjustments live in LifetimeState and reset when a new LifetimeState is created."""
        creature = _make_creature()
        _add_history(creature, gene_id=0, tick=0)
        apply_reward_to_history(creature, reward=10.0)
        self.assertNotEqual(creature.lifetime.learned_gene_adjustments, {})
        self.assertNotEqual(creature.lifetime.learned_state_gene_adjustments, {})

        # Simulate new epoch: replace lifetime
        creature.lifetime = LifetimeState(
            x=1, y=1,
            energy=30.0,
        )
        self.assertEqual(creature.lifetime.learned_gene_adjustments, {})
        self.assertEqual(creature.lifetime.learned_state_gene_adjustments, {})

    def test_history_trimmed_to_max_length(self):
        creature = _make_creature(history_length=3)
        for tick in range(10):
            _add_history(creature, gene_id=tick % 5, tick=tick)
        self.assertLessEqual(len(creature.lifetime.history), 3)

    def test_empty_history_no_crash(self):
        creature = _make_creature()
        # Should not raise
        apply_reward_to_history(creature, reward=5.0)
        self.assertEqual(creature.lifetime.learned_gene_adjustments, {})

    def test_n_step_return_uses_trajectory_rewards(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=0.5)
        _add_history(creature, gene_id=0, tick=0, reward=1.0)
        _add_history(creature, gene_id=1, tick=1, reward=2.0)
        _add_history(creature, gene_id=2, tick=2, reward=4.0)

        # Reward event propagated over recent trajectory:
        # gene 2: 4.0
        # gene 1: 2.0 + 0.5 * 4.0 = 4.0
        # gene 0: 1.0 + 0.5 * 4.0 = 3.0
        apply_reward_to_history(creature, reward=0.0)

        self.assertAlmostEqual(creature.lifetime.learned_gene_adjustments.get(2, 0.0), 4.0)
        self.assertAlmostEqual(creature.lifetime.learned_gene_adjustments.get(1, 0.0), 4.0)
        self.assertAlmostEqual(creature.lifetime.learned_gene_adjustments.get(0, 0.0), 3.0)

    def test_state_specific_adjustment_only_applies_in_matching_state(self):
        creature = _make_creature(learning_rate=1.0, reward_decay=1.0)
        entry = HistoryEntry(
            sensor=_food_sensor(),
            gene_id=2,
            action=ActionType.IDLE,
            reward=0.0,
            action_success=True,
            tick_index=0,
        )
        record_history(creature, entry)

        apply_reward_to_history(creature, reward=5.0)

        self.assertAlmostEqual(state_adjustment_for_gene(creature, _food_sensor(), 2), 5.0)
        self.assertEqual(state_adjustment_for_gene(creature, _dummy_sensor(), 2), 0.0)


if __name__ == '__main__':
    unittest.main()
