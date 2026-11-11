import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from wandb_logger import finish_run, log_epoch, start_run


class WandbLoggerTests(unittest.TestCase):
    def test_disabled_has_no_dependency_or_network_side_effect(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(start_run("/tmp", "a" * 64, {"name": "trial"}, {}))
            log_epoch(None, {"epoch": 1, "f1": .6})
            finish_run(None, {})

    def test_online_mode_requires_key(self):
        with patch.dict(os.environ, {"WANDB_PROJECT": "ai301-a3"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY"):
                start_run("/tmp", "a" * 64, {"name": "trial"}, {})

    def test_logs_epochs_and_summary_with_run_identity(self):
        run = MagicMock()
        mock_sdk = SimpleNamespace(init=MagicMock(return_value=run))
        with patch.dict(sys.modules, {"wandb": mock_sdk}):
            with patch.dict(os.environ, {"WANDB_PROJECT": "ai301-a3",
                                          "WANDB_ENTITY": "team",
                                          "WANDB_API_KEY": "placeholder"},
                            clear=True):
                active = start_run("/tmp", "abcdef012345", {"name": "LSTM"}, {"name": "GPU"})
                log_epoch(active, {"epoch": 2, "train_loss": .12, "f1": .81})
                finish_run(active, {"best_epoch": 2, "checkpoint_sha256": "abc",
                                    "train_rows": 24000, "validation_rows": 1000,
                                    "epochs": 2, "seconds": 8.,
                                    "metrics": {"f1": .81, "precision": .82}})
        self.assertEqual(mock_sdk.init.call_args.kwargs["id"], "abcdef01")
        self.assertEqual(mock_sdk.init.call_args.kwargs["project"], "ai301-a3")
        run.log.assert_called_once_with(
            {"epoch": 2, "train_loss": .12, "f1": .81}, step=2)
        self.assertEqual(run.summary.update.call_args.args[0]["best_f1"], .81)
        run.finish.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
