import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app.config as config
from app.simulation_runner import SimulationRunner


class TestSimulationRunnerMetrics(unittest.TestCase):

    def test_epoch_history_records_encoder_metrics(self):
        old_mode = config.SENSOR_ENCODER_MODE
        try:
            runner = SimulationRunner(seed=1, ticks_per_epoch=2)
            config.SENSOR_ENCODER_MODE = 'compact'
            runner.reset(seed=1)
            runner.step_epoch()

            self.assertTrue(runner.epoch_history)
            latest = runner.epoch_history[-1]
            self.assertIn('action_value_variance', latest)
            self.assertIn('learning_speed', latest)
            self.assertEqual(latest['sensor_encoder_mode'], 'compact')
        finally:
            config.SENSOR_ENCODER_MODE = old_mode


if __name__ == '__main__':
    unittest.main()
