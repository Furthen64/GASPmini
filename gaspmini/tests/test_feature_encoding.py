import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app.config as config
from app.feature_encoding import encode_pattern_for_matching, encode_sensor_for_matching
from app.gene_logic import score_gene_match
from app.models import ActionType, CellType, GenePattern, SensorField


class TestFeatureEncoding(unittest.TestCase):

    def _sensor(self, **kwargs) -> SensorField:
        defaults = dict(
            current_cell=CellType.EMPTY,
            front_cell=CellType.EMPTY,
            left_cell=CellType.EMPTY,
            right_cell=CellType.EMPTY,
            back_cell=CellType.EMPTY,
            last_action=ActionType.IDLE,
            last_action_success=True,
            hunger_bucket=0,
        )
        defaults.update(kwargs)
        return SensorField(**defaults)

    def _pattern(self, **kwargs) -> GenePattern:
        defaults = dict(
            current_cell=None,
            front_cell=None,
            left_cell=None,
            right_cell=None,
            back_cell=None,
            last_action=None,
            last_action_success=None,
            hunger_bucket=None,
        )
        defaults.update(kwargs)
        return GenePattern(**defaults)

    def test_compact_sensor_encoding(self):
        old = config.SENSOR_ENCODER_MODE
        config.SENSOR_ENCODER_MODE = 'compact'
        try:
            sensor = self._sensor(
                front_cell=CellType.WALL,
                left_cell=CellType.CREATURE,
                right_cell=CellType.FOOD,
                back_cell=CellType.EMPTY,
                hunger_bucket=3,
            )
            encoded = encode_sensor_for_matching(sensor)
            # food=right(3), obstacles front+left => 0b0011, creature nearby=1, high hunger=2
            self.assertEqual(encoded, (3, 0b0011, 1, 2))
        finally:
            config.SENSOR_ENCODER_MODE = old

    def test_compact_pattern_encoding_and_match_score(self):
        old = config.SENSOR_ENCODER_MODE
        config.SENSOR_ENCODER_MODE = 'compact'
        try:
            sensor = self._sensor(
                front_cell=CellType.FOOD,
                left_cell=CellType.EMPTY,
                right_cell=CellType.WALL,
                back_cell=CellType.EMPTY,
                hunger_bucket=2,
            )
            pattern = self._pattern(
                front_cell=CellType.FOOD,
                left_cell=CellType.EMPTY,
                right_cell=CellType.WALL,
                back_cell=CellType.EMPTY,
                hunger_bucket=1,
            )
            # All 4 compact fields become concrete and should match.
            self.assertEqual(score_gene_match(sensor, pattern), config.GENE_MATCH_SCORE * 4)
            self.assertEqual(encode_pattern_for_matching(pattern), (1, 0b0100, 0, 1))
        finally:
            config.SENSOR_ENCODER_MODE = old

    def test_legacy_mode_keeps_detailed_features(self):
        old = config.SENSOR_ENCODER_MODE
        config.SENSOR_ENCODER_MODE = 'legacy'
        try:
            sensor = self._sensor(front_cell=CellType.FOOD)
            encoded = encode_sensor_for_matching(sensor)
            self.assertEqual(len(encoded), 8)
            self.assertEqual(encoded[1], CellType.FOOD)
        finally:
            config.SENSOR_ENCODER_MODE = old


if __name__ == '__main__':
    unittest.main()
