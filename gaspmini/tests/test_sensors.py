# Tests for local sensing.

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models import ActionType, CellType, Creature, Direction, Gene, GenePattern, Genome, LifetimeState, WorldState
from app.sensors import build_sensor_data


def _make_creature(x: int = 3, y: int = 3, direction: Direction = Direction.NORTH) -> Creature:
    genome = Genome(
        genes=[
            Gene(
                gene_id=0,
                pattern=GenePattern(None, None, None, None, None, None, None, None),
                action=ActionType.IDLE,
                base_priority=0.0,
            )
        ]
    )
    lifetime = LifetimeState(x=x, y=y, direction=direction, energy=30.0)
    return Creature(creature_id=0, genome=genome, lifetime=lifetime)


class TestSensors(unittest.TestCase):

    def test_sensor_reports_current_cell_food(self):
        creature = _make_creature()
        world = WorldState(width=8, height=8, food_positions={(3, 3)}, creatures=[creature])

        sensor = build_sensor_data(creature, world)

        self.assertEqual(sensor.current_cell, CellType.FOOD)

    def test_sensor_reports_adjacent_food_by_direction(self):
        creature = _make_creature(direction=Direction.NORTH)
        world = WorldState(
            width=8,
            height=8,
            food_positions={(3, 2), (2, 3), (4, 3), (3, 4)},
            creatures=[creature],
        )

        sensor = build_sensor_data(creature, world)

        self.assertEqual(sensor.front_cell, CellType.FOOD)
        self.assertEqual(sensor.left_cell, CellType.FOOD)
        self.assertEqual(sensor.right_cell, CellType.FOOD)
        self.assertEqual(sensor.back_cell, CellType.FOOD)


if __name__ == '__main__':
    unittest.main()