# Tests for action resolution in the simulation loop.

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app.config as config
from app.models import (
    ActionType,
    Creature,
    Direction,
    Gene,
    GenePattern,
    Genome,
    LifetimeState,
    WorldState,
)
from app.simulation import execute_action
from app.simulation import tick_creature


def _make_creature(
    x: int = 2,
    y: int = 2,
    direction: Direction = Direction.EAST,
    energy: float = 30.0,
) -> Creature:
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
    lifetime = LifetimeState(x=x, y=y, direction=direction, energy=energy)
    return Creature(creature_id=0, genome=genome, lifetime=lifetime)


class TestSimulation(unittest.TestCase):

    def test_move_forward_onto_food_does_not_consume_it(self):
        creature = _make_creature()
        world = WorldState(width=8, height=8, food_positions={(3, 2)}, creatures=[creature])

        result = execute_action(creature, ActionType.MOVE_FORWARD, world)

        self.assertTrue(result.success)
        self.assertEqual((creature.lifetime.x, creature.lifetime.y), (3, 2))
        self.assertIn((3, 2), world.food_positions)
        self.assertEqual(creature.lifetime.food_eaten, 0)
        self.assertEqual(creature.lifetime.energy, 30.0)
        self.assertEqual(result.reward, 0.0)

    def test_eat_does_not_consume_adjacent_food(self):
        creature = _make_creature()
        world = WorldState(width=8, height=8, food_positions={(3, 2)}, creatures=[creature])

        result = execute_action(creature, ActionType.EAT, world)

        self.assertFalse(result.success)
        self.assertIn((3, 2), world.food_positions)
        self.assertEqual(creature.lifetime.food_eaten, 0)
        self.assertEqual(creature.lifetime.energy, 30.0)
        self.assertEqual(result.reward, config.REWARD_FAILED_EAT)

    def test_eat_consumes_food_underfoot(self):
        creature = _make_creature()
        world = WorldState(width=8, height=8, food_positions={(2, 2)}, creatures=[creature])

        result = execute_action(creature, ActionType.EAT, world)

        self.assertTrue(result.success)
        self.assertNotIn((2, 2), world.food_positions)
        self.assertEqual(creature.lifetime.food_eaten, 1)
        self.assertEqual(creature.lifetime.energy, 30.0 + config.ENERGY_GAIN_FROM_FOOD)
        self.assertEqual(result.reward, config.REWARD_EAT_FOOD)

    def test_tick_creature_records_run_history_sample(self):
        creature = _make_creature()
        world = WorldState(width=8, height=8, creatures=[creature])

        tick_creature(creature, world)

        self.assertEqual(len(creature.lifetime.run_history), 1)
        sample = creature.lifetime.run_history[0]
        self.assertEqual(sample.age_ticks, 1)
        self.assertEqual(sample.food_eaten, 0)
        self.assertTrue(sample.alive)



if __name__ == '__main__':
    unittest.main()