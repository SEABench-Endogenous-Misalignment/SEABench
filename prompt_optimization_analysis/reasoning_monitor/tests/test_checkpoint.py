import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_hashes(self):
        record = json.loads((CHECKPOINT / "CHECKPOINT.json").read_text())
        expected = {
            "split_sha256": CHECKPOINT / "split.json",
            "config_sha256": CHECKPOINT / "config.json",
            "progression_model_sha256": CHECKPOINT / "progression_filter" / "monitor.joblib",
            "selector_model_sha256": CHECKPOINT / "selector_verifier" / "stage1.joblib",
            "verifier_prompt_sha256": CHECKPOINT / "verifier_prompt.md",
            "final_decision_sha256": CHECKPOINT / "final" / "selection_and_report.json",
        }
        for field, path in expected.items():
            self.assertEqual(record[field], digest(path), field)

    def test_reported_operating_point(self):
        result = json.loads(
            (CHECKPOINT / "final" / "selection_and_report.json").read_text()
        )
        metrics = result["metrics"]["overall"]
        self.assertEqual(
            (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]),
            (56, 102, 11, 23),
        )
        self.assertAlmostEqual(metrics["accuracy"], 158 / 192)
        self.assertAlmostEqual(metrics["failure_recall"], 56 / 79)
        self.assertAlmostEqual(metrics["false_positive_rate"], 11 / 113)

    def test_surface_specific_thresholds(self):
        result = json.loads(
            (CHECKPOINT / "final" / "selection_and_report.json").read_text()
        )
        self.assertEqual(
            set(result["surface_thresholds"]),
            {"controller", "memory", "tools_skills"},
        )
        for thresholds in result["surface_thresholds"].values():
            self.assertIn("local_threshold", thresholds)
            self.assertIn("verifier_threshold", thresholds)

    def test_progression_model_configuration(self):
        config = json.loads((CHECKPOINT / "config.json").read_text())
        self.assertEqual(config["progression_tree_count"], 300)
        self.assertEqual(config["progression_tree_depth"], 16)
        self.assertEqual(config["progression_minimum_leaf_size"], 2)
        self.assertEqual(config["progression_feature_fraction"], 0.30)


if __name__ == "__main__":
    unittest.main()
