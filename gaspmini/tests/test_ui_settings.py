import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PySide6.QtCore import QSettings

from app.ui_settings import (
    load_best_creature_persistence_settings,
    load_main_window_settings,
    save_best_creature_persistence_settings,
    save_main_window_settings,
)


class TestUiSettings(unittest.TestCase):

    def test_best_creature_persistence_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = os.path.join(temp_dir, 'ui-settings.ini')
            writer = QSettings(settings_path, QSettings.Format.IniFormat)

            save_best_creature_persistence_settings(
                writer,
                autosave_enabled=True,
                inject_saved_best_enabled=True,
                autosave_path='saved/genome.json',
            )

            reader = QSettings(settings_path, QSettings.Format.IniFormat)
            values = load_best_creature_persistence_settings(reader)

            self.assertEqual(
                values,
                {
                    'autosave_enabled': True,
                    'inject_saved_best_enabled': True,
                    'autosave_path': 'saved/genome.json',
                },
            )

    def test_main_window_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = os.path.join(temp_dir, 'ui-settings.ini')
            writer = QSettings(settings_path, QSettings.Format.IniFormat)

            save_main_window_settings(
                writer,
                autosave_enabled=True,
                inject_saved_best_enabled=False,
                autosave_path='saved/genome.json',
                profile_id='longer_strategic',
                custom_map_id='hidden_food',
                ticks_per_epoch=320,
                seed=12345,
                testing_ground_enabled=True,
            )

            reader = QSettings(settings_path, QSettings.Format.IniFormat)
            values = load_main_window_settings(
                reader,
                default_profile_id='short_chaotic',
                default_custom_map_id='',
                default_ticks_per_epoch=160,
                default_seed=42,
            )

            self.assertEqual(
                values,
                {
                    'autosave_enabled': True,
                    'inject_saved_best_enabled': False,
                    'autosave_path': 'saved/genome.json',
                    'profile_id': 'longer_strategic',
                    'custom_map_id': 'hidden_food',
                    'ticks_per_epoch': 320,
                    'seed': 12345,
                    'testing_ground_enabled': True,
                },
            )


if __name__ == '__main__':
    unittest.main()