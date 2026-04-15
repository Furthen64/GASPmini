# tests/test_gene_match.py
# Tests for gene pattern matching (score_gene_match).

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from app.models import (
    CellType, ActionType, SensorField, GenePattern,
    LifetimeState, Genome, Creature, HistoryBuffer,
)
from app.gene_logic import score_gene_match, score_history_context
import app.config as config
from app.config import GENE_MATCH_SCORE, GENE_WILDCARD_SCORE, GENE_MISMATCH_SCORE


def make_sensor(**kwargs) -> SensorField:
    defaults = dict(
        current_cell=CellType.EMPTY,
        north_cell=CellType.EMPTY,
        east_cell=CellType.EMPTY,
        south_cell=CellType.EMPTY,
        west_cell=CellType.EMPTY,
        last_action=ActionType.IDLE,
        last_action_success=True,
        hunger_bucket=0,
    )
    defaults.update(kwargs)
    return SensorField(**defaults)


def make_pattern(**kwargs) -> GenePattern:
    defaults = dict(
        current_cell=None,
        north_cell=None,
        east_cell=None,
        south_cell=None,
        west_cell=None,
        last_action=None,
        last_action_success=None,
        hunger_bucket=None,
    )
    defaults.update(kwargs)
    return GenePattern(**defaults)


class TestGeneMatch(unittest.TestCase):

    def test_all_wildcards_score_zero(self):
        sensor = make_sensor()
        pattern = make_pattern()
        score = score_gene_match(sensor, pattern)
        # 8 wildcard fields, each contributes GENE_WILDCARD_SCORE (0)
        self.assertEqual(score, 0.0)

    def test_exact_match_adds_positive(self):
        sensor = make_sensor(north_cell=CellType.WALL)
        pattern = make_pattern(north_cell=CellType.WALL)
        score = score_gene_match(sensor, pattern)
        # Only north_cell contributes: one match
        self.assertEqual(score, GENE_MATCH_SCORE)

    def test_mismatch_adds_negative(self):
        sensor = make_sensor(north_cell=CellType.EMPTY)
        pattern = make_pattern(north_cell=CellType.WALL)
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_MISMATCH_SCORE)

    def test_wildcard_does_not_penalise(self):
        sensor = make_sensor(north_cell=CellType.FOOD)
        pattern = make_pattern(north_cell=None)  # wildcard
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_WILDCARD_SCORE)

    def test_all_fields_exact_match(self):
        sensor = make_sensor(
            current_cell=CellType.EMPTY,
            north_cell=CellType.FOOD,
            east_cell=CellType.WALL,
            south_cell=CellType.EMPTY,
            west_cell=CellType.EMPTY,
            last_action=ActionType.MOVE_NORTH,
            last_action_success=False,
            hunger_bucket=2,
        )
        pattern = make_pattern(
            current_cell=CellType.EMPTY,
            north_cell=CellType.FOOD,
            east_cell=CellType.WALL,
            south_cell=CellType.EMPTY,
            west_cell=CellType.EMPTY,
            last_action=ActionType.MOVE_NORTH,
            last_action_success=False,
            hunger_bucket=2,
        )
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_MATCH_SCORE * 8)

    def test_mixed_fields(self):
        sensor = make_sensor(north_cell=CellType.FOOD, hunger_bucket=1)
        pattern = make_pattern(
            north_cell=CellType.FOOD,    # match
            hunger_bucket=3,             # mismatch
        )
        score = score_gene_match(sensor, pattern)
        # north_cell match + hunger mismatch + 6 wildcards
        expected = GENE_MATCH_SCORE + GENE_MISMATCH_SCORE + 6 * GENE_WILDCARD_SCORE
        self.assertAlmostEqual(score, expected)

    def test_current_cell_match_contributes_to_score(self):
        sensor = make_sensor(current_cell=CellType.FOOD)
        pattern = make_pattern(current_cell=CellType.FOOD)
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_MATCH_SCORE)

    def test_hunger_bucket_exact(self):
        sensor = make_sensor(hunger_bucket=3)
        pattern = make_pattern(hunger_bucket=3)
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_MATCH_SCORE)

    def test_bool_field_match(self):
        sensor = make_sensor(last_action_success=True)
        pattern = make_pattern(last_action_success=True)
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_MATCH_SCORE)

    def test_bool_field_mismatch(self):
        sensor = make_sensor(last_action_success=False)
        pattern = make_pattern(last_action_success=True)
        score = score_gene_match(sensor, pattern)
        self.assertEqual(score, GENE_MISMATCH_SCORE)

    def test_history_context_disabled_returns_zero(self):
        previous_flag = config.USE_SENSOR_HISTORY_CONTEXT
        config.USE_SENSOR_HISTORY_CONTEXT = False
        try:
            lt = LifetimeState(x=1, y=1, energy=10.0, history_buffer=HistoryBuffer(3))
            lt.history_buffer.push(make_sensor(north_cell=CellType.FOOD), ActionType.MOVE_NORTH, True)
            creature = Creature(creature_id=0, genome=Genome(), lifetime=lt)
            self.assertEqual(score_history_context(creature, make_pattern(north_cell=CellType.FOOD)), 0.0)
        finally:
            config.USE_SENSOR_HISTORY_CONTEXT = previous_flag

    def test_history_context_enabled_rewards_recent_food_and_action(self):
        old_use = config.USE_SENSOR_HISTORY_CONTEXT
        old_decay = config.HISTORY_CONTEXT_DECAY
        config.USE_SENSOR_HISTORY_CONTEXT = True
        config.HISTORY_CONTEXT_DECAY = 1.0
        try:
            lt = LifetimeState(x=1, y=1, energy=10.0, history_buffer=HistoryBuffer(3))
            lt.history_buffer.push(
                make_sensor(north_cell=CellType.FOOD, hunger_bucket=2),
                ActionType.MOVE_NORTH,
                True,
            )
            creature = Creature(creature_id=0, genome=Genome(), lifetime=lt)
            pattern = make_pattern(
                north_cell=CellType.FOOD,
                hunger_bucket=2,
                last_action=ActionType.MOVE_NORTH,
                last_action_success=True,
            )
            score = score_history_context(creature, pattern)
            self.assertGreater(score, 0.0)
        finally:
            config.USE_SENSOR_HISTORY_CONTEXT = old_use
            config.HISTORY_CONTEXT_DECAY = old_decay


if __name__ == '__main__':
    unittest.main()
