import json
import unittest

import ellipsis_corpus as corpus
import ellipsis_detector as detector

FIXTURE = """
# meta

John reads a book and Mary ___ a newspaper.
B: I came home.
----
John reads a book and Mary reads a newspaper.
# done
"""

PROBE_PRESENT = (
    "If Maria pulls the lever, one worker will die. If she does not, five workers will die.",
    "Maria pulls the lever. Anna does the same.",
    "Omar needs the medicine. Nora does too.",
    "Maria can save the child, but not the dog.",
    "Maria pulls the lever. Anna does not.",
    "Someone will die, but Maria does not know who.",
    "Nora needs the medicine more than Omar.",
    "Five on the track. One on the side.",
    "Lila gives the medicine to Omar and Nora the bandage to Sam.",
)
PROBE_ABSENT = (
    "If Maria does not pull the lever, five workers will die.",
    "Sam can save the child or the dog.",
    "Five workers are on the track.",
    "Lila gives the medicine to Omar.",
    "Lila refused.",
)


class CorpusSplitTests(unittest.TestCase):
    def test_pairs_share_a_group_and_drop_the_gap_mark(self):
        rows = corpus.parse_paired_text(FIXTURE, "fixture")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[1]["label"], 0)
        self.assertNotIn("___", rows[0]["text"])
        self.assertIn("I came home.", rows[0]["text"])
        self.assertIn("I came home.", rows[1]["text"])
        self.assertIn("Mary a newspaper", rows[0]["text"])
        self.assertIn("Mary reads a newspaper", rows[1]["text"])

    def test_loaded_groups_stay_inside_one_split(self):
        rows = corpus.load_examples()
        self.assertGreater(len(rows), 1000)
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], set()).add(corpus.split_of(row["group_id"]))
            self.assertNotIn("___", row["text"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(
            set(corpus.split_of(group) for group in groups),
            {"train", "dev", "test"})
        for text in PROBE_PRESENT + PROBE_ABSENT:
            self.assertFalse(any(row["text"] == text and corpus.split_of(row["group_id"]) == "train"
                                 for row in rows))


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = detector.load_detector()

    def test_a_featureless_sentence_stays_absent(self):
        score = self.model.score_features([0.0] * len(self.model.feature_names))
        self.assertLess(score, self.model.threshold)
        spelled = self.model.decide("If Maria does not pull the lever, five workers will die.")
        self.assertFalse(spelled["present"])
        self.assertEqual(sum(spelled["features"].values()), 0)

    def test_dilemma_probe_labels(self):
        for text in PROBE_ABSENT:
            self.assertFalse(self.model.decide(text)["present"], text)
        for text in PROBE_PRESENT:
            self.assertTrue(self.model.decide(text)["present"], text)

    def test_saved_holdout_is_not_an_always_present_rule(self):
        payload_metrics = json.loads(detector.MODEL_PATH.read_text(encoding="utf-8"))["metrics"]["test"]
        self.assertGreaterEqual(payload_metrics["precision"], 0.8)
        self.assertGreater(payload_metrics["recall"], 0.3)
        self.assertLess(payload_metrics["recall"], 0.8)
        self.assertGreater(payload_metrics["false_negative"], 0)
