import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.world import generate_world


class TestCustomMaps(unittest.TestCase):

    def test_hidden_food_layout_is_applied_exactly(self):
        world = generate_world(
            seed=7,
            epoch_index=3,
            population_size=0,
            genomes=[],
            custom_map_id='hidden_food',
        )

        self.assertEqual((world.width, world.height), (30, 20))
        self.assertEqual(
            world.food_positions,
            {
                (4, 5),
                (14, 5),
                (24, 5),
                (9, 13),
                (19, 13),
            },
        )
        self.assertEqual(len(world.walls), 40)
        self.assertTrue({(3, 3), (4, 3), (5, 3)}.issubset(world.walls))
        self.assertTrue({(8, 11), (9, 11), (10, 11)}.issubset(world.walls))

    def test_custom_map_layout_ignores_seeded_biome_generation(self):
        world_a = generate_world(seed=1, epoch_index=0, population_size=0, genomes=[], custom_map_id='hidden_food')
        world_b = generate_world(seed=999, epoch_index=25, population_size=0, genomes=[], custom_map_id='hidden_food')

        self.assertEqual(world_a.walls, world_b.walls)
        self.assertEqual(world_a.food_positions, world_b.food_positions)


if __name__ == '__main__':
    unittest.main()