from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTest(unittest.TestCase):
    @staticmethod
    def argument_defaults(filename: str) -> dict[str, object]:
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        defaults: dict[str, object] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument" or not node.args:
                continue
            try:
                option = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[option] = ast.literal_eval(keyword.value)
        return defaults

    def test_python_sources_parse(self) -> None:
        sources = sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
        self.assertGreater(len(sources), 20)
        for source in sources:
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    def test_machine_readable_paper_protocol(self) -> None:
        protocol = json.loads((ROOT / "configs" / "paper_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["input_size"], 640)
        self.assertEqual(protocol["train_batch_size"], 6)
        self.assertEqual(protocol["epochs"], 200)
        self.assertEqual(protocol["optimizer"], "AdamW")
        self.assertEqual(protocol["seeds"], [42, 123, 3407, 2025, 2026])
        self.assertEqual(protocol["loss"]["side_output_count"], 4)
        self.assertEqual(set(protocol["datasets"]), {"uav_crack500", "cracktree260"})

    def test_training_defaults_match_protocol(self) -> None:
        protocol = json.loads((ROOT / "configs" / "paper_protocol.json").read_text(encoding="utf-8"))
        defaults = self.argument_defaults("train.py")
        expected = {
            "--epochs": protocol["epochs"],
            "--image-size": protocol["input_size"],
            "--batch-size": protocol["train_batch_size"],
            "--eval-batch-size": protocol["evaluation_batch_size"],
            "--lr": protocol["initial_learning_rate"],
            "--weight-decay": protocol["weight_decay"],
            "--warmup-epochs": protocol["warmup_epochs"],
            "--ema-decay": protocol["ema_decay"],
            "--semantic-weight": protocol["loss"]["semantic_weight"],
            "--boundary-weight": protocol["loss"]["boundary_weight"],
            "--side-weight": protocol["loss"]["side_output_weight"],
            "--boundary-width": protocol["loss"]["boundary_width_pixels"],
            "--clip-grad": protocol["gradient_clip_norm"],
        }
        for option, value in expected.items():
            self.assertEqual(defaults[option], value, option)

    def test_frozen_split_counts_and_disjointness(self) -> None:
        expected = {
            "uav_crack500_grouped_v2.json": (282, 70, 48),
            "cracktree260_grouped_v2.json": (182, 39, 39),
        }
        for filename, counts in expected.items():
            payload = json.loads((ROOT / "splits" / filename).read_text(encoding="utf-8"))
            parts = [payload["train"], payload["validation"], payload["test"]]
            self.assertEqual(tuple(map(len, parts)), counts)
            image_sets = [{item["image"] for item in part} for part in parts]
            group_sets = [{item["group"] for item in part} for part in parts]
            for left in range(3):
                for right in range(left + 1, 3):
                    self.assertFalse(image_sets[left] & image_sets[right])
                    self.assertFalse(group_sets[left] & group_sets[right])

    def test_readme_is_clean_and_current(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Semantic-Prior-Guided", readme)
        self.assertIn("200 epochs", readme)
        self.assertIn("CrackLS315 is excluded", readme)
        self.assertNotIn("\ufffd", readme)

    def test_required_runtime_dependencies_are_declared(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("torch", "torchvision", "numpy", "pillow", "scipy", "scikit-image", "opencv-python", "matplotlib"):
            self.assertIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
