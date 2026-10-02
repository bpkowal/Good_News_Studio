import unittest

import numpy as np

import parsing_game_S as s


def _policy():
    return s.CEMPolicy(np.zeros((4, len(s.FEATURE_NAMES))))


def _selected(sentence):
    analyses = s.classify_to_attachments(s.get_nlp()(sentence))
    if len(analyses) != 1:
        raise AssertionError(f"{sentence!r} produced {len(analyses)} to-attachments")
    return analyses[0]


def _destination_texts(parsed):
    mentions = {item["mention_id"]: item for item in parsed["entities"]}
    texts = []
    for event in parsed["events"]:
        for ref in event["roles"].get("destination", []):
            texts.append(mentions[ref]["text"])
    return texts


class ToAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = _policy()
        s.get_to_attachment_model()

    def test_training_constructions_are_recovered(self):
        for sentence, label in s.TO_ATTACHMENT_TRAINING:
            with self.subTest(sentence=sentence):
                self.assertEqual(_selected(sentence)["selected"], label)

    def test_work_keeps_both_token_identities(self):
        analysis = _selected("All of them were late to work.")
        self.assertEqual(analysis["token_identities"], ["noun", "verb"])
        self.assertIn("infinitival_complement", analysis["structural_candidates"])
        self.assertIn("directional_pp", analysis["structural_candidates"])
        self.assertIn("unresolved", analysis["structural_candidates"])
        self.assertNotIn("degree_result_infinitive", analysis["structural_candidates"])

    def test_determiner_closes_the_verb_identity(self):
        analysis = _selected("They were late to the office.")
        self.assertEqual(analysis["token_identities"], ["noun"])
        self.assertNotIn("infinitival_complement", analysis["structural_candidates"])
        self.assertEqual(analysis["selected"], "directional_pp")

    def test_late_to_work_is_a_destination_not_an_event(self):
        parsed = s.parse_world_state("All of them were late to work.", self.policy)
        analysis = parsed["to_attachment_analyses"][0]
        self.assertEqual(analysis["selected_id"], "C2")
        self.assertEqual(analysis["selected"], "directional_pp")
        self.assertFalse(analysis["abstained"])
        self.assertFalse(analysis["eligible_for_world_state"])
        self.assertEqual(parsed["schema_version"], 9)
        predicates = [event["predicate"]["lemma"] for event in parsed["events"]]
        self.assertNotIn("work", predicates)
        self.assertEqual(_destination_texts(parsed), ["work"])
        self.assertFalse(any(frame.predicate_lemma == "work"
                             for frame in s.extract_proposition_frames("All of them were late to work.")))

    def test_eager_to_work_stays_an_infinitival_complement(self):
        parsed = s.parse_world_state("They were eager to work.", self.policy)
        analysis = parsed["to_attachment_analyses"][0]
        self.assertEqual(analysis["selected_id"], "C1")
        self.assertEqual(_destination_texts(parsed), [])
        frames = s.extract_proposition_frames("They were eager to work.")
        work = next(frame for frame in frames if frame.predicate_lemma == "work")
        self.assertEqual(work.to_attachment["selected"], "infinitival_complement")

    def test_degree_marker_selects_result_infinitive_when_the_verb_is_forced(self):
        analysis = _selected("They were too late to leave.")
        self.assertEqual(analysis["token_identities"], ["verb"])
        self.assertEqual(analysis["selected"], "degree_result_infinitive")
        self.assertIn("degree_result_infinitive", analysis["structural_candidates"])
        self.assertNotIn("directional_pp", analysis["structural_candidates"])
        parsed = s.parse_world_state("They were too late to leave.", self.policy)
        self.assertIn("leave", [event["predicate"]["lemma"] for event in parsed["events"]])
        self.assertEqual(_destination_texts(parsed), [])

    def test_too_late_to_work_abstains(self):
        analysis = _selected("They were too late to work.")
        self.assertEqual(analysis["selected_id"], "C4")
        self.assertTrue(analysis["abstained"])
        self.assertIn("directional_pp", analysis["structural_candidates"])
        self.assertIn("degree_result_infinitive", analysis["structural_candidates"])
        parsed = s.parse_world_state("They were too late to work.", self.policy)
        self.assertNotIn("work", [event["predicate"]["lemma"] for event in parsed["events"]])
        self.assertEqual(_destination_texts(parsed), [])
        self.assertTrue(any("to_attachment_unresolved" in event["issues"]
                            for event in parsed["events"]))

    def test_unknown_head_abstains_when_work_is_ambiguous(self):
        analysis = _selected("They were foobish to work.")
        self.assertEqual(analysis["selected"], "unresolved")
        self.assertEqual(analysis["head_semantic_class"], "unknown")

    def test_glad_and_hard_prefer_infinitives(self):
        self.assertEqual(_selected("They were glad to help.")["selected"], "infinitival_complement")
        analysis = _selected("The metal is hard to work.")
        self.assertEqual(analysis["token_identities"], ["noun", "verb"])
        self.assertEqual(analysis["selected"], "infinitival_complement")

    def test_object_control_infinitives_remain_propositions(self):
        for sentence, lemma in (
            ("The flood caused the library to close.", "close"),
            ("The storm forced the school to close.", "close"),
            ("The manager allowed the workers to leave.", "leave"),
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(_selected(sentence)["selected"], "infinitival_complement")
                frames = s.extract_proposition_frames(sentence)
                self.assertIn(lemma, [frame.predicate_lemma for frame in frames])

    def test_causal_lead_to_pair_is_not_suppressed(self):
        records = s.collect_claim_evidence("Rain leads to flooding.")
        paired = [record for record in records if len(record["entities"]) == 2]
        self.assertTrue(paired)
        heads = sorted(entity["head_text"].lower() for entity in paired[0]["entities"])
        self.assertEqual(heads, ["flooding", "rain"])
        self.assertEqual(_selected("Rain leads to flooding.")["selected"], "unresolved")
        parsed = s.parse_world_state("Rain leads to flooding.", self.policy)
        self.assertFalse(any("to_attachment_unresolved" in event["issues"]
                             for event in parsed["events"]))


if __name__ == "__main__":
    unittest.main()
