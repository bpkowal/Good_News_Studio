import json
import unittest

import ellipsis_synthetic as synthetic
from ellipsis_corpus import split_of

PROBE = (
    "If Maria pulls the lever, one worker will die. If she does not, five workers will die.",
    "Sam can save the child or the dog.",
    "If Maria does not pull the lever, five workers will die.",
)


class SyntheticSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in synthetic.OUT_PATH.read_text(encoding="utf-8").splitlines()]

    def test_rows_come_from_training_groups_only(self):
        self.assertGreaterEqual(len(self.rows), 100)
        for row in self.rows:
            self.assertEqual(row["source_split"], "train")
            self.assertEqual(split_of(row["source_group"]), "train")
            self.assertIn(row["gold"], row["text"])
            self.assertIn(row["pattern"], {"main", "nearer_finite", "inner_finite"})

    def test_dilemma_lines_are_absent(self):
        texts = {row["text"] for row in self.rows}
        for probe in PROBE:
            self.assertNotIn(probe, texts)
        for row in self.rows:
            lowered = row["text"].lower()
            self.assertNotIn("pull the lever", lowered)
            self.assertNotIn("the medicine", lowered)

    def test_inner_and_nearer_patterns_are_present(self):
        patterns = {row["pattern"] for row in self.rows}
        self.assertEqual(patterns, {"main", "nearer_finite", "inner_finite"})


if __name__ == "__main__":
    unittest.main()
