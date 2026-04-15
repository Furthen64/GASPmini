import os
import sys
import tempfile
import unittest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models import ActionType, CellType, Gene, GenePattern, Genome
from app.genome_store import genome_file_exists, load_genome_from_file, save_genome_to_file


def _make_genome() -> Genome:
    return Genome(
        genes=[
            Gene(
                gene_id=0,
                pattern=GenePattern(CellType.FOOD, CellType.EMPTY, None, None, None, ActionType.MOVE_NORTH, True, 2),
                action=ActionType.IDLE,
                base_priority=1.25,
            )
        ],
        learning_rate=0.2,
        reward_decay=0.8,
        exploration_rate=0.03,
        history_length=8,
    )


class TestGenomeStore(unittest.TestCase):

    def test_save_and_load_round_trip(self):
        genome = _make_genome()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'best.json')

            save_genome_to_file(genome, path)

            self.assertTrue(genome_file_exists(path))
            loaded = load_genome_from_file(path)
            self.assertEqual(loaded, genome)

    def test_loading_legacy_relative_genome_raises_clear_error(self):
        legacy_payload = {
            'learning_rate': 0.2,
            'reward_decay': 0.8,
            'exploration_rate': 0.03,
            'history_length': 8,
            'genes': [
                {
                    'gene_id': 0,
                    'action': 'MOVE_FORWARD',
                    'base_priority': 1.25,
                    'pattern': {
                        'current_cell': 'FOOD',
                        'front_cell': 'EMPTY',
                        'left_cell': None,
                        'right_cell': None,
                        'back_cell': None,
                        'last_action': None,
                        'last_action_success': True,
                        'hunger_bucket': 2,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'legacy.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(legacy_payload, handle)

            with self.assertRaisesRegex(ValueError, 'facing-relative sensor fields'):
                load_genome_from_file(path)


if __name__ == '__main__':
    unittest.main()