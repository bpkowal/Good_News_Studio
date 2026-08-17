from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from global_workspace.framework_retrieval import (
    CorpusPassage,
    EvidenceThresholds,
    EvidenceTier,
    corpus_fingerprint,
    format_evidence_context,
    load_approved_documents,
    load_corpus_passages,
    retrieve_framework_evidence,
)


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_query(self, _query):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]


class DualScoreEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_query(self, query):
        if query == "framework lens":
            return [1.0, 0.0, 0.0]
        if query == "case facts":
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]


def passage(text, document_id, tags):
    return CorpusPassage(
        passage_id=f"{document_id}:p0",
        document_id=document_id,
        framework="virtue",
        source_file=f"{document_id}.md",
        text=text,
        source_kind="commentary",
        metadata={"author": document_id, "source": "Test", "tags": tags},
    )


class TestFrameworkRAG(unittest.TestCase):
    def test_typographic_and_markdown_quotes_are_ingested(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary)
            (corpus / "approved.md").write_text(
                """---
title: Practical Wisdom
author: Test Philosopher
source: Test Source
status: approved
retracted: false
tags: [virtue_ethics, practical_wisdom]
---

“Practical wisdom notices the morally salient circumstances.”

> Character is formed through repeated action.

This paragraph is commentary rather than a direct quotation.
""",
                encoding="utf-8",
            )
            passages = load_corpus_passages(corpus, framework="virtue")

        self.assertEqual(len(passages), 3)
        self.assertEqual(
            [item.source_kind for item in passages],
            ["direct_quote", "direct_quote", "commentary"],
        )
        self.assertIn("Practical wisdom", passages[0].text)
        self.assertFalse(passages[1].text.startswith(">"))

    def test_unapproved_and_retracted_documents_are_excluded(self):
        template = """---
title: {title}
author: Test
source: Test
status: {status}
retracted: {retracted}
tags: [virtue_ethics]
---

Evidence body.
"""
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary)
            (corpus / "approved.md").write_text(
                template.format(title="Approved", status="approved", retracted="false")
            )
            (corpus / "pending.md").write_text(
                template.format(title="Pending", status="pending", retracted="false")
            )
            (corpus / "retracted.md").write_text(
                template.format(title="Retracted", status="approved", retracted="true")
            )
            documents = load_approved_documents(corpus, framework="virtue")

        self.assertEqual([item.metadata["title"] for item in documents], ["Approved"])

    def test_core_adjacent_and_rejected_evidence_have_bounded_influence(self):
        passages = [
            passage("core", "core-doc", ["virtue_ethics"]),
            passage("outside-identity", "outside-doc", ["decision_theory"]),
            passage("weak", "weak-doc", ["virtue_ethics"]),
        ]
        result = retrieve_framework_evidence(
            passages,
            query="test case",
            embedder=FakeEmbedder(
                {
                    "core": [0.9, 0.4359],
                    "outside-identity": [1.0, 0.0],
                    "weak": [0.1, 0.995],
                }
            ),
            query_lens="virtue ethics practical wisdom",
            identity_tags={"virtue_ethics", "virtue"},
            thresholds=EvidenceThresholds(core=0.28, adjacent=0.18),
        )

        self.assertEqual(
            [item.tier for item in result.evidence],
            [EvidenceTier.CORE, EvidenceTier.ADJACENT],
        )
        self.assertEqual(result.rejected_count, 1)
        context = format_evidence_context(result)
        self.assertIn("interpretive context only", context)
        self.assertIn("[E1 | CORE", context)
        self.assertIn("[E2 | ADJACENT", context)

    def test_only_two_adjacent_passages_are_admitted_without_core_evidence(self):
        passages = [
            passage("first", "first-doc", ["virtue_ethics"]),
            passage("second", "second-doc", ["virtue_ethics"]),
            passage("third", "third-doc", ["virtue_ethics"]),
        ]
        result = retrieve_framework_evidence(
            passages,
            query="test case",
            embedder=FakeEmbedder(
                {
                    "first": [0.25, 0.9682],
                    "second": [0.23, 0.9732],
                    "third": [0.20, 0.9798],
                }
            ),
            query_lens="virtue ethics",
            identity_tags={"virtue_ethics"},
            thresholds=EvidenceThresholds(core=0.28, adjacent=0.18),
        )

        self.assertEqual(len(result.evidence), 2)
        self.assertTrue(all(item.tier is EvidenceTier.ADJACENT for item in result.evidence))

    def test_adjacent_deontological_role_cannot_become_kantian_core(self):
        core = passage("kantian", "kant-doc", ["deontology", "duty"])
        adjacent = passage("rossian", "ross-doc", ["deontology", "duty"])
        object.__setattr__(
            core,
            "metadata",
            {**core.metadata, "framework_role": "kantian_core"},
        )
        object.__setattr__(
            adjacent,
            "metadata",
            {**adjacent.metadata, "framework_role": "adjacent_deontology"},
        )
        result = retrieve_framework_evidence(
            [core, adjacent],
            query="case",
            embedder=FakeEmbedder(
                {"kantian": [0.9, 0.4359], "rossian": [1.0, 0.0]}
            ),
            query_lens="strict Kantian ethics",
            identity_tags={"deontology", "duty"},
            core_evidence_roles={"kantian_core"},
            thresholds=EvidenceThresholds(core=0.36, adjacent=0.25),
        )

        self.assertEqual(
            [item.tier for item in result.evidence],
            [EvidenceTier.CORE, EvidenceTier.ADJACENT],
        )

    def test_case_relevance_ranks_evidence_only_after_identity_qualification(self):
        first = passage("generic-core", "generic-doc", ["deontology"])
        second = passage("case-core", "case-doc", ["deontology"])
        for item in (first, second):
            object.__setattr__(
                item,
                "metadata",
                {**item.metadata, "framework_role": "kantian_core"},
            )
        result = retrieve_framework_evidence(
            [first, second],
            query="case facts",
            embedder=DualScoreEmbedder(
                {
                    "generic-core": [0.90, 0.10, 0.0],
                    "case-core": [0.80, 0.60, 0.0],
                }
            ),
            query_lens="framework lens",
            identity_tags={"deontology"},
            core_evidence_roles={"kantian_core"},
            thresholds=EvidenceThresholds(core=0.36, adjacent=0.25),
            separate_identity_scoring=True,
        )

        self.assertEqual(result.evidence[0].passage.text, "case-core")
        self.assertTrue(all(item.tier is EvidenceTier.CORE for item in result.evidence))

    def test_deontology_corpus_has_explicit_framework_roles(self):
        documents = load_approved_documents(
            "deontological_corpus",
            framework="deontological",
        )
        roles = [item.metadata.get("framework_role") for item in documents]

        self.assertEqual(roles.count("kantian_core"), 7)
        self.assertEqual(roles.count("adjacent_deontology"), 8)
        self.assertNotIn(None, roles)

    def test_corpus_fingerprint_changes_with_corpus_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary)
            path = corpus / "source.md"
            path.write_text("first", encoding="utf-8")
            first = corpus_fingerprint(corpus, framework="virtue")
            path.write_text("second", encoding="utf-8")
            second = corpus_fingerprint(corpus, framework="virtue")

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
