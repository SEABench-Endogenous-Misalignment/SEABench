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
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (18, 40, 3, 11))
        self.assertAlmostEqual(metrics["accuracy"], 58 / 72)
        self.assertAlmostEqual(metrics["failure_recall"], 18 / 29)
        self.assertAlmostEqual(metrics["false_positive_rate"], 3 / 43)


if __name__ == "__main__":
    unittest.main()
