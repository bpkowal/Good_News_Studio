"""Behavioral regressions for J's evidence/claim boundary (local spaCy required)."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import parsing_game_J as parser


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, cls.history = parser.train_policy()

    def claim(self, sentence):
        return parser.parse_sentence(sentence, self.policy)

    def test_import_has_no_training_logging_or_model_load(self):
        root = str(Path(parser.__file__).parent)
        with tempfile.TemporaryDirectory() as directory:
            code = (f"import sys; sys.path.insert(0, {root!r}); "
                    "import parsing_game_J as j; "
                    "assert j.get_nlp.cache_info().currsize == 0; "
                    "assert sys.stdout is sys.__stdout__")
            result = subprocess.run([sys.executable, "-c", code], cwd=directory,
                                    text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_default_seed_regression_suites(self):
        for suite in (parser.TRAIN_EXAMPLES, parser.TEST_EXAMPLES,
                      parser.NEGATION_GENERALIZATION_EXAMPLES, parser.ROBUSTNESS_EXAMPLES,
                      parser.LEXICAL_HOLDOUT, parser.UNKNOWN_HOLDOUT, parser.EPISTEMIC_HOLDOUT):
            with self.subTest(sentences=[e["sentence"] for e in suite]):
                metrics, records = parser.evaluate_suite(suite, self.policy)
                self.assertEqual(metrics["end_to_end_correct"], 1.0,
                                 [r["sentence"] for r in records if not r["end_to_end_correct"]])

    def test_denial_retains_direction_without_positive_edge(self):
        claim = self.claim("Disease was not induced by exposure.")
        self.assertEqual((claim["source"], claim["target"]), ("exposure", "Disease"))
        self.assertEqual(claim["relation_type"], "causal")
        self.assertEqual(claim["assertion_status"], "denied")
        self.assertFalse(claim["eligible_for_world_state"])

    def test_asserted_claim_can_pass_gate(self):
        claim = self.claim("Exposure induces disease.")
        self.assertTrue(claim["eligible_for_world_state"])
        self.assertEqual(claim["validation_reasons"], [])

    def test_modal_negation_retains_both_scopes(self):
        claim = self.claim("Exposure might not induce disease.")
        self.assertEqual(claim["assertion_status"], "possible")
        self.assertEqual(claim["assertion"]["polarity"], "negative")
        self.assertEqual(set(claim["assertion"]["statuses"]), {"possible", "denied"})
        self.assertFalse(claim["eligible_for_world_state"])

    def test_scoped_and_unsupported_claims_never_commit(self):
        for sentence in (
            "Rain probably causes flooding.", "No rain causes flooding.",
            "Rain causes no flooding.", "Rain failed to cause flooding.",
            "Rain does not seem to cause flooding.",
            "Researchers say rain causes flooding.",
            "Rain caused flooding, according to scientists.",
            "If rain causes flooding, roads close.", "Does rain cause flooding?",
            '"Rain causes flooding."',
            "Rain and snow cause flooding.",
            "Rain causes flooding and smoke triggers alarm.",
            "Rain causes flooding. Smoke triggers alarm.",
        ):
            with self.subTest(sentence=sentence):
                self.assertFalse(self.claim(sentence)["eligible_for_world_state"])

    def test_contrastive_negation_is_not_predicate_denial(self):
        for sentence in ("Rain not snow causes flooding.", "Rain causes flooding not drought."):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.claim(sentence)["assertion_status"], "asserted")

    def test_association_does_not_mean_causality_is_false(self):
        claim = self.claim("Exposure associates with disease.")
        self.assertEqual(claim["relation_type"], "association")
        self.assertEqual(claim["assertion_status"], "asserted")
        self.assertIsNone(claim["direction"])
        self.assertFalse(claim["eligible_for_world_state"])

    def test_unknown_semantics_preserve_predicate_and_abstain(self):
        for item in parser.UNKNOWN_HOLDOUT:
            claim = self.claim(item["sentence"])
            self.assertEqual(claim["relation_type"], "unresolved")
            self.assertIsNotNone(claim["evidence"]["selected_candidate"]["index"])
            self.assertFalse(claim["eligible_for_world_state"])

    def test_missing_arguments_do_not_mean_no_causation(self):
        for sentence in ("", "Hello.", "Rain causes."):
            claim = self.claim(sentence)
            self.assertEqual(claim["relation_type"], "unresolved")
            self.assertFalse(claim["eligible_for_world_state"])
            json.dumps(claim)

    def test_pos_and_lemma_errors_do_not_veto_semantic_evidence(self):
        doc = parser.get_nlp()("Flooding results from rain.")
        doc[1].pos_ = "NOUN"
        doc[1].lemma_ = "wrong_lemma"
        with patch.object(parser, "get_nlp", return_value=lambda sentence: doc):
            evidence = parser.collect_evidence(doc.text)
        self.assertEqual(evidence["selected_candidate"]["semantic_class"], "causal")
        self.assertEqual(evidence["features"]["predicate_pos_verb"], 0)
        self.assertEqual(evidence["features"]["semantic_reverse"], 1)

    def test_gold_entity_fields_never_affect_features(self):
        original = parser.TRAIN_EXAMPLES[0]
        poisoned = dict(original, entity1="wrong", entity2="also wrong", correct_action=999)
        np.testing.assert_array_equal(parser.extract_channels(original),
                                      parser.extract_channels(poisoned))

    def test_heldout_lexical_content_is_disjoint_from_training(self):
        nlp = parser.get_nlp()
        content = lambda examples: {
            t.lemma_.lower() for item in examples for t in nlp(item["sentence"])
            if t.pos_ in {"NOUN", "PROPN", "VERB"} and not t.is_stop
        }
        self.assertFalse(content(parser.TRAIN_EXAMPLES) & content(parser.LEXICAL_HOLDOUT))
        training_text = {e["sentence"] for e in parser.TRAIN_EXAMPLES}
        self.assertFalse(training_text & {e["sentence"] for e in parser.UNKNOWN_HOLDOUT})

    def test_trace_contributions_reconstruct_scores(self):
        claim = self.claim("Exposure induces disease.")
        decision = claim["decision"]
        for action, score in enumerate(decision["scores"]):
            self.assertAlmostEqual(sum(decision["contributions"][parser.ACTION_NAMES[action]].values()),
                                   score)
        for token in claim["evidence"]["tokens"]:
            self.assertEqual(claim["sentence"][token["start"]:token["end"]], token["text"])
        json.dumps(claim)

    def test_disagreement_and_ties_block_commitment(self):
        for desired in (1, 2, 3, None):
            weights = np.zeros((4, len(parser.FEATURE_NAMES)))
            if desired is not None:
                weights[desired, -1] = 10
            claim = parser.parse_sentence("Rain causes flooding.", parser.CEMPolicy(weights))
            self.assertFalse(claim["eligible_for_world_state"])
            self.assertEqual(claim["relation_type"], "unresolved")

    def test_policy_roundtrip_and_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            self.policy.save(path)
            loaded = parser.CEMPolicy.load(path)
            np.testing.assert_array_equal(loaded.weights, self.policy.weights)
            self.assertEqual(parser.parse_sentence("Rain causes flooding.", loaded),
                             self.claim("Rain causes flooding."))
        with self.assertRaises(ValueError):
            parser.CEMPolicy(np.zeros((4, 17)))
        with self.assertRaises(ValueError):
            parser.CEMPolicy(np.full((4, len(parser.FEATURE_NAMES)), np.nan))

    def test_audit_and_generation_history(self):
        self.assertEqual(len(self.history), 80)
        self.assertEqual([h["generation"] for h in self.history], list(range(1, 81)))
        self.assertEqual(parser.find_representation_collisions(parser.TRAIN_EXAMPLES), [])
        item = parser.TRAIN_EXAMPLES[0]
        conflicts = parser.find_representation_collisions([
            item, dict(item, correct_action=(item["correct_action"] + 1) % 4)])
        self.assertEqual(len(conflicts), 1)

    def test_multi_relation_heldout_exact_lists(self):
        metrics, records = parser.evaluate_multi_suite(parser.MULTI_RELATION_HOLDOUT, self.policy)
        self.assertEqual(metrics["exact_claim_list_accuracy"], 1,
                         [r for r in records if not r["exact_match"]])

    def test_independent_clauses_preserve_full_mentions(self):
        text = "Heavy rain causes severe flooding and strong winds trigger power outages."
        claims = parser.parse_claims(text, self.policy)
        self.assertEqual([(c["source"], c["target"]) for c in claims],
                         [("Heavy rain", "severe flooding"), ("strong winds", "power outages")])
        self.assertTrue(all(c["eligible_for_world_state"] for c in claims))
        self.assertEqual(len({c["claim_id"] for c in claims}), 2)
        for claim in claims:
            for entity in claim["entities"]:
                self.assertEqual(text[entity["start"]:entity["end"]], entity["text"])

    def test_shared_subject_scope_and_new_subject_reset(self):
        cases = [
            ("Rain causes flooding and triggers landslides.", ["asserted", "asserted"]),
            ("Rain may cause flooding and trigger landslides.", ["possible", "possible"]),
            ("Rain may cause flooding, but smoke triggers alarms.", ["possible", "asserted"]),
            ("Rain does not cause flooding, but smoke triggers alarms.", ["denied", "asserted"]),
            ("Rain causes flooding and does not trigger alarms.", ["asserted", "denied"]),
        ]
        for text, statuses in cases:
            with self.subTest(text=text):
                claims = parser.parse_claims(text, self.policy)
                self.assertEqual([c["assertion_status"] for c in claims], statuses)
                self.assertEqual([c["eligible_for_world_state"] for c in claims],
                                 [s == "asserted" for s in statuses])

    def test_fronted_agent_preserves_roles_not_textual_direction(self):
        ordinary = self.claim("Flooding was caused by rain.")
        fronted = self.claim("By rain, flooding was caused.")
        self.assertEqual((ordinary["source"].lower(), ordinary["target"].lower()),
                         (fronted["source"].lower(), fronted["target"].lower()))
        self.assertEqual(ordinary["direction"], "second_to_first")
        self.assertEqual(fronted["direction"], "first_to_second")
        self.assertTrue(fronted["eligible_for_world_state"])

    def test_multiple_passives_and_sentence_boundaries(self):
        for text in ("Flooding was caused by rain and damage was caused by wind.",
                     "Flooding was caused by rain. Damage was caused by wind."):
            with self.subTest(text=text):
                claims = parser.parse_claims(text, self.policy)
                self.assertEqual([(c["source"].lower(), c["target"].lower()) for c in claims],
                                 [("rain", "flooding"), ("wind", "damage")])
                self.assertTrue(all(c["eligible_for_world_state"] for c in claims))

    def test_nominal_event_spans(self):
        claims = parser.parse_claims(
            "Unfair blame causes intense stress and disrupting sleep causes severe fatigue.",
            self.policy)
        self.assertEqual([(c["source"], c["target"]) for c in claims],
                         [("Unfair blame", "intense stress"), ("disrupting sleep", "severe fatigue")])
        self.assertEqual(claims[1]["source_entity"]["kind"], "event")
        self.assertEqual(self.claim("Stress causes disrupting sleep.")["target"], "disrupting sleep")

    def test_span_lookup_handles_repeats_without_substring_matches(self):
        doc = parser.get_nlp()("Severe flooding follows severe flooding and rainfall.")
        spans = parser.find_entity_spans(doc, "severe flooding")
        self.assertEqual(len(spans), 2)
        self.assertNotEqual(spans[0]["start"], spans[1]["start"])
        self.assertEqual(parser.find_entity_spans(doc, "rain"), [])
        self.assertEqual(parser.find_entity_spans(doc, ""), [])
        self.assertEqual(parser.find_entity_spans(
            parser.get_nlp()("Disrupting sleep causes fatigue."), "disrupting sleep")[0]["text"],
                         "Disrupting sleep")

    def test_collectives_disjunction_and_ambiguous_scope_stay_flagged(self):
        for text in ("Rain and snow cause flooding.", "Rain causes flooding and erosion.",
                     "Rain causes flooding or smoke triggers alarms.",
                     "Rain does not cause flooding and trigger landslides."):
            with self.subTest(text=text):
                claims = parser.parse_claims(text, self.policy)
                self.assertTrue(all(not c["eligible_for_world_state"] for c in claims))
        self.assertEqual(self.claim("Rain and snow cause flooding.")["source"], "Rain and snow")

    def test_unknown_relation_is_not_lost_beside_known_relation(self):
        for text in ("Rain accompanies flooding and smoke triggers alarms.",
                     "Rain causes flooding and wind modulates erosion."):
            with self.subTest(text=text):
                claims = parser.parse_claims(text, self.policy)
                self.assertEqual(len(claims), 2)
                self.assertEqual(sorted(c["relation_type"] for c in claims), ["causal", "unresolved"])

    def test_repeated_names_remain_distinct_mentions(self):
        claims = parser.parse_claims("Rain causes flooding and rain triggers landslides.", self.policy)
        self.assertEqual(len(claims), 2)
        self.assertEqual([c["source"].lower() for c in claims], ["rain", "rain"])
        self.assertNotEqual(claims[0]["source_entity"]["start"], claims[1]["source_entity"]["start"])

    def test_missing_second_argument_does_not_borrow_previous_clause(self):
        claims = parser.parse_claims("Rain causes flooding and smoke triggers.", self.policy)
        self.assertEqual(len(claims), 2)
        self.assertTrue(claims[0]["eligible_for_world_state"])
        self.assertEqual(claims[1]["entities"], [])
        self.assertFalse(claims[1]["eligible_for_world_state"])

    def test_legacy_api_exposes_all_claims_without_silent_commitment(self):
        claim = self.claim("Rain causes flooding and smoke triggers alarms.")
        self.assertEqual(len(claim["claims"]), 2)
        self.assertFalse(claim["eligible_for_world_state"])
        self.assertIn("multiple_claims_use_parse_claims", claim["validation_reasons"])

    def test_rolling_user_suite_keeps_only_latest_five(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.json"
            entries = []
            for index in range(6):
                entries = parser.append_user_probe(parser.load_user_suite(path), f"Probe {index}.")
                parser.save_user_suite(path, entries)
            self.assertEqual([e["sentence"] for e in parser.load_user_suite(path)],
                             [f"Probe {i}." for i in range(1, 6)])
            self.assertNotIn("Probe 0.", path.read_text())
            self.assertEqual(parser.append_user_probe(entries, "   "), entries)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["probes.json"])

    def test_participial_modifier_does_not_replace_inventory_predicate(self):
        text = "A warehouse has two energy-saving generators remaining."
        records = parser.collect_claim_evidence(text)
        predicates = [r["tokens"][r["selected_candidate"]["index"]]["text"]
                      for r in records if "selected_candidate" in r]
        self.assertNotIn("saving", predicates)
        inventory = next(r for r in records
                         if r["tokens"][r["selected_candidate"]["index"]]["lemma"] == "have")
        self.assertEqual([e["text"] for e in inventory["entities"]],
                         ["A warehouse", "two energy-saving generators"])
        self.assertEqual(inventory["assertion"]["status"], "asserted")
        self.assertEqual(inventory["selected_candidate"]["semantic_class"], "unknown")

    def test_relative_clause_coordination_preserves_attachment_ambiguity(self):
        text = ("A courier can carry the package to a customer who ordered it yesterday, "
                "or reroute it to a depot where workers await deliveries.")
        records = parser.collect_claim_evidence(text)
        redirect = next(r for r in records
                        if r["tokens"][r["selected_candidate"]["index"]]["lemma"] == "reroute")
        self.assertEqual(redirect["entities"][0]["text"], "A courier")
        self.assertEqual(redirect["assertion"]["status"], "possible")
        self.assertIn("ambiguous_coordination_attachment", redirect["issues"])
        self.assertEqual(len(redirect["selected_candidate"]["subject_attachment_candidates"]), 2)

    def test_corrupt_user_suite_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.json"
            path.write_text("broken input", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "left unchanged"):
                parser.load_user_suite(path)
            self.assertEqual(path.read_text(), "broken input")

    def test_user_probes_are_unscored_and_failures_are_visible(self):
        entries = parser.append_user_probe([], "Rain causes flooding.")
        entries = parser.append_user_probe(entries, "A failing input.")
        with patch.object(parser, "parse_claims", side_effect=[[], ValueError("probe failed")]):
            evaluated = parser.evaluate_user_suite(entries, self.policy)
        self.assertEqual(evaluated[0]["last_result"], {"status": "unscored", "claims": []})
        self.assertEqual(evaluated[1]["last_result"]["error_type"], "ValueError")
        self.assertNotIn("last_result", entries[0])

    def test_interactive_probe_is_persisted_but_not_archived_in_dated_logs(self):
        sentence = "Quasar light accompanies unusual oscillations."
        original_training = json.dumps(parser.TRAIN_EXAMPLES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path = root / "policy.json"
            self.policy.save(policy_path)
            suite_path = root / "probes.json"
            output = io.StringIO()
            with patch.object(sys.stdin, "isatty", return_value=True), \
                    patch("builtins.input", return_value=sentence) as prompt, redirect_stdout(output):
                parser.main(["--load-policy", str(policy_path), "--user-suite", str(suite_path),
                             "--output-dir", str(root / "logs")])
            prompt.assert_called_once()
            self.assertEqual(parser.load_user_suite(suite_path)[0]["sentence"], sentence)
            self.assertIn("YOUR SENTENCE PROBES", output.getvalue())
            self.assertIn("no accuracy score", output.getvalue())
            for path in (root / "logs").iterdir():
                self.assertNotIn(sentence, path.read_text())
            with patch("builtins.input", side_effect=AssertionError("must not prompt")), \
                    redirect_stdout(io.StringIO()):
                parser.main(["--load-policy", str(policy_path), "--user-suite", str(suite_path),
                             "--output-dir", str(root / "logs"), "--no-prompt",
                             "--add-sentence", "Rain causes flooding."])
            self.assertEqual(len(parser.load_user_suite(suite_path)), 2)
        self.assertEqual(json.dumps(parser.TRAIN_EXAMPLES), original_training)


if __name__ == "__main__":
    unittest.main()
