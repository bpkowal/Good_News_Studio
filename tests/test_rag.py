import unittest
from _legacy.rag_core import store_document, retrieve_relevant_docs
from _legacy.generator import generate_response

class TestRAGIntegration(unittest.TestCase):
    def test_basic_context_retrieval_and_response(self):
        # Setup known document
        doc_text = "Dragons live in volcanic mountains where the skies are filled with ash."
        store_document(doc_text)
        print("Document stored.")

        # User query related to the doc
        query = "What is the weather like where dragons live?"
        # Debug
        context = retrieve_relevant_docs(query)
        print ("🔍 Retrieved Context:\n", context)

        # Retrieve context and pass to generator
        response = generate_response(query)
        print("🧠 Model Response:\n", response)

        # Assert something about the output (very basic for now)
        self.assertIn("volcanic", response.lower())

    def test_no_context_available(self):
        # Deliberately avoid storing anything
        query = "What is the capital of Mars?"
        context = retrieve_relevant_docs(query)
        print("🔍 Retrieved Context (should be empty):\n", context)

        response = generate_response(query)
        print("🧠 No-context Response:\n", response)

        self.assertTrue("i don't know" in response.lower() or len(response) > 10)

if __name__ == '__main__':
    unittest.main()