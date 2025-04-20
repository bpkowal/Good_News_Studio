import unittest
from pathlib import Path
from corpus_loader import load_corpus

class TestRealCorpusLoad(unittest.TestCase):
    def test_real_corpus_only_loads_approved(self):
        # Load real corpus
        corpus = load_corpus("utilitarian_corpus")

        # There should be at least one document (adjust if needed)
        self.assertGreater(len(corpus), 0)

        for doc in corpus:
            meta = doc["metadata"]
            text = doc["text"]

            # Make sure only approved content is loaded
            self.assertEqual(meta.get("status"), "approved")

            # Make sure required metadata exists
            self.assertIn("title", meta)
            self.assertIn("author", meta)
            self.assertIn("source", meta)

            # Make sure text content is not empty
            self.assertTrue(len(text.strip()) > 0)

if __name__ == "__main__":
    unittest.main()