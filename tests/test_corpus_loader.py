import unittest
import os
from pathlib import Path
from corpus_loader import load_corpus

class TestCorpusLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary test corpus directory
        self.test_dir = Path("test_corpus")
        self.test_dir.mkdir(exist_ok=True)

        # Create an approved file
        approved_file = self.test_dir / "approved.md"
        approved_file.write_text("""---
title: Approved Test
author: Test Philosopher
status: approved
---

This is a test document that should be loaded.
""")

        # Create a pending file
        pending_file = self.test_dir / "pending.md"
        pending_file.write_text("""---
title: Pending Test
author: Shadow Philosopher
status: pending
---

This document should be skipped.
""")

    def test_only_approved_files_are_loaded(self):
        corpus = load_corpus(str(self.test_dir))
        self.assertEqual(len(corpus), 1)
        self.assertEqual(corpus[0]["metadata"]["title"], "Approved Test")
        self.assertIn("should be loaded", corpus[0]["text"])

    def tearDown(self):
        # Clean up test files
        for file in self.test_dir.iterdir():
            file.unlink()
        self.test_dir.rmdir()

if __name__ == "__main__":
    unittest.main()