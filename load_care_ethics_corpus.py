"""Compatibility wrapper for the shared framework corpus loader."""

from global_workspace.framework_retrieval import (
    load_chroma_corpus,
    sanitize_chroma_metadata as sanitize_metadata,
)


def load_care_ethics_corpus(required_tag=None, embedder=None):
    return load_chroma_corpus(
        "care_ethics_corpus",
        collection_name="care_ethics",
        framework="care",
        required_tag=required_tag,
        embedder=embedder,
    )
