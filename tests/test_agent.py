"""Unit tests for the Legal Bee RAG agent components."""

import sys, unittest, logging
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)


class TestIntentDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.retrieval.intent_detector import IntentDetector
        cls.detector = IntentDetector()

    def test_language_bangla(self):
        self.assertEqual(self.detector.detect_language("আইন কী"), "bn")

    def test_language_english(self):
        self.assertEqual(self.detector.detect_language("What is the law?"), "en")

    def test_language_mixed_bangla_dominant(self):
        self.assertEqual(
            self.detector.detect_language("আমার landlord আমাকে evict করেছে"),
            "bn",
        )

    def test_query_type_law_search(self):
        self.assertEqual(
            self.detector.detect_query_type("Show me Article 90E"),
            "law_search",
        )

    def test_query_type_act_summary(self):
        self.assertEqual(
            self.detector.detect_query_type("Summarize the Penal Code"),
            "act_summary",
        )

    def test_query_type_amendment(self):
        self.assertEqual(
            self.detector.detect_query_type("What changed in the Amendment Act?"),
            "amendment_question",
        )

    def test_query_type_fact_analysis(self):
        self.assertEqual(
            self.detector.detect_query_type("My landlord has evicted me illegally"),
            "fact_analysis",
        )

    def test_query_type_legal_question(self):
        self.assertEqual(
            self.detector.detect_query_type("What is the punishment for theft?"),
            "legal_question",
        )

    def test_no_results_when_zero(self):
        self.assertEqual(
            self.detector.detect_query_type("random text", retrieved_count=0),
            "no_results",
        )

    def test_legal_domain_criminal(self):
        self.assertEqual(
            self.detector.detect_legal_domain("What is the punishment for murder?"),
            "criminal",
        )

    def test_legal_domain_election(self):
        self.assertEqual(
            self.detector.detect_legal_domain("What are the nomination rules?"),
            "election",
        )

    def test_legal_domain_employment(self):
        self.assertEqual(
            self.detector.detect_legal_domain("What are government service rules?"),
            "employment",
        )

    def test_needs_rewrite_short(self):
        self.assertTrue(self.detector.needs_rewrite("What is nomination?"))

    def test_needs_rewrite_with_section(self):
        self.assertFalse(self.detector.needs_rewrite("What is Article 90E?"))


class TestCitationBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.retrieval.citation_builder import CitationBuilder
        from app.models.schemas import RetrievedChunk
        cls.builder = CitationBuilder()
        cls.RetrievedChunk = RetrievedChunk

    def test_build_from_prebuilt_citation(self):
        chunks = [
            self.RetrievedChunk(
                text="test",
                citation="Penal Code, 1860, Section 379",
                act_name="Penal Code",
                hierarchy={"section": "379"},
            )
        ]
        citations = self.builder.build(chunks)
        self.assertIn("Penal Code, 1860, Section 379", citations)

    def test_build_from_hierarchy_fallback(self):
        chunks = [
            self.RetrievedChunk(
                text="test",
                citation="",
                act_name="Penal Code",
                hierarchy={"section": "379", "article": "20"},
            )
        ]
        citations = self.builder.build(chunks)
        self.assertTrue(any("Section 379" in c for c in citations))

    def test_deduplicates(self):
        chunks = [
            self.RetrievedChunk(citation="Penal Code, 1860, Section 379"),
            self.RetrievedChunk(citation="Penal Code, 1860, Section 379"),
        ]
        citations = self.builder.build(chunks)
        self.assertEqual(len(citations), 1)

    def test_empty_chunks(self):
        citations = self.builder.build([])
        self.assertEqual(citations, [])

    def test_context_string_format(self):
        chunks = [
            self.RetrievedChunk(
                text="The punishment for theft is...",
                citation="Penal Code, Section 379",
                score=0.85,
            )
        ]
        ctx = self.builder.build_context_string(chunks)
        self.assertIn("[Chunk 1]", ctx)
        self.assertIn("Section 379", ctx)
        self.assertIn("0.85", ctx)

    def test_build_references(self):
        chunks = [
            self.RetrievedChunk(
                references=[{"type": "amends", "target": "Original Act"}, {"type": "act", "target": "2020 Act"}],
            )
        ]
        refs = self.builder.build_references(chunks)
        self.assertEqual(len(refs), 2)


class TestConfig(unittest.TestCase):
    def test_config_loads(self):
        from app.config import config
        self.assertTrue(len(config.qdrant_url) > 0)
        self.assertTrue(len(config.collection_name) > 0)

    def test_config_validate_missing(self):
        from app.config import Config
        cfg = Config(qdrant_url="", qdrant_api_key="", google_api_key="", llm_provider="gemini", cerebras_api_key="", openrouter_api_key="")
        errors = cfg.validate()
        self.assertGreater(len(errors), 0)


class TestSchemas(unittest.TestCase):
    def test_chat_request_validation(self):
        from app.models.schemas import ChatRequest
        req = ChatRequest(question="What is the law?")
        self.assertEqual(req.question, "What is the law?")
        self.assertEqual(req.user_type, "general")

    def test_agent_response_defaults(self):
        from app.models.schemas import AgentResponse
        resp = AgentResponse(question="test", answer="test answer")
        self.assertEqual(resp.confidence, "medium")
        self.assertEqual(resp.language_detected, "en")
        self.assertTrue(len(resp.timestamp) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
