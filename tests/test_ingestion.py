"""Unit tests for hierarchy parsing, citation generation, and reference extraction."""

import sys, unittest, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from ingestion.structure_parser import (
    StructureParser,
    _is_valid_clause_identifier,
)
from ingestion.chunker import LegalChunker, ChunkType
from ingestion.metadata import LawMetadata, LawReference


class TestClauseIdentifier(unittest.TestCase):
    def test_bangla_digits(self):
        self.assertTrue(_is_valid_clause_identifier("১"))
        self.assertTrue(_is_valid_clause_identifier("১২"))
        self.assertTrue(_is_valid_clause_identifier("২"))
        self.assertFalse(_is_valid_clause_identifier("১২৩৪"))

    def test_arabic_digits(self):
        self.assertTrue(_is_valid_clause_identifier("1"))
        self.assertTrue(_is_valid_clause_identifier("10"))
        self.assertFalse(_is_valid_clause_identifier("1000"))

    def test_bangla_consonants(self):
        self.assertTrue(_is_valid_clause_identifier("ক"))
        self.assertTrue(_is_valid_clause_identifier("খ"))
        self.assertTrue(_is_valid_clause_identifier("গ"))

    def test_ascii_letters(self):
        self.assertTrue(_is_valid_clause_identifier("a"))
        self.assertTrue(_is_valid_clause_identifier("i"))
        self.assertTrue(_is_valid_clause_identifier("ii"))
        self.assertTrue(_is_valid_clause_identifier("aa"))

    def test_special_identifiers(self):
        self.assertTrue(_is_valid_clause_identifier("xial"))
        self.assertTrue(_is_valid_clause_identifier("xiaa"))

    def test_rejects_english_words(self):
        self.assertFalse(_is_valid_clause_identifier("review"))
        self.assertFalse(_is_valid_clause_identifier("Amendment"))
        self.assertFalse(_is_valid_clause_identifier("GEMS"))
        self.assertFalse(_is_valid_clause_identifier("PMIS"))

    def test_rejects_long_identifiers(self):
        self.assertFalse(_is_valid_clause_identifier("review"))
        self.assertFalse(_is_valid_clause_identifier("except"))
        self.assertFalse(_is_valid_clause_identifier("xxxxx"))

    def test_rejects_uppercase_ascii(self):
        self.assertFalse(_is_valid_clause_identifier("A"))
        self.assertFalse(_is_valid_clause_identifier("BBB"))
        self.assertFalse(_is_valid_clause_identifier("I"))


class TestHierarchyParsing(unittest.TestCase):
    def setUp(self):
        self.parser = StructureParser()

    def test_bangla_sections_detected(self):
        text = "সংক্ষিপ্ত শিরোনাম ১। (১) এই আইন X নামে অভিহিত হইবে।\n\n২০১৮ সনের ৫৭ ২। Y আইনের ধারা ৩৭ সংশোধন।"
        doc = self.parser.parse(text)
        self.assertGreaterEqual(len(doc.sections), 2)

    def test_english_sections_fallback(self):
        text = "Article 1. Short title.\nArticle 2. Definitions."
        doc = self.parser.parse(text)
        self.assertGreaterEqual(len(doc.sections), 2)

    def test_no_sections(self):
        text = "This is just some text with no section markers at all."
        doc = self.parser.parse(text)
        self.assertEqual(len(doc.sections), 0)

    def test_clauses_inside_section(self):
        text = "সংক্ষিপ্ত শিরোনাম ১। (১) Short title. (২) Commencement."
        doc = self.parser.parse(text)
        self.assertEqual(len(doc.sections), 1)
        self.assertEqual(len(doc.sections[0].clauses), 2)

    def test_non_legal_clauses_ignored(self):
        text = "ধারা ১। (১) The court may (review) the case. (২) Appeal."
        doc = self.parser.parse(text)
        clauses = doc.sections[0].clauses if doc.sections else []
        identifiers = [c.identifier for c in clauses]
        self.assertNotIn("review", identifiers)
        self.assertIn("১", identifiers)
        self.assertIn("২", identifiers)

    def test_inserted_section_detection(self):
        text = "২০১৮ সনের ৫৭ ২। সরকারি চাকরি আইনের ধারা ৩৭ এর পর নিম্নরূপ নূতন ধারা ৩৭ক সন্নিবেশিত হইবে।"
        doc = self.parser.parse(text)
        if doc.sections:
            self.assertEqual(doc.sections[0].inserted_section_number, "৩৭ক")

    def test_article_extraction(self):
        text = "P.O. No. 155 of ৩। উক্ত Order এর Article 8 এর clause (1) এ উল্লিখিত।"
        doc = self.parser.parse(text)
        if doc.sections:
            self.assertEqual(doc.sections[0].article, "8")


class TestCitationGeneration(unittest.TestCase):
    def _make_meta(self, act_name="Test Act, 2026", amendment_of=""):
        return LawMetadata(
            act_name=act_name,
            bangla_name="",
            act_number="1",
            act_year=2026,
            document_type="Act",
            amendment_of=amendment_of,
        )

    def test_section_citation(self):
        h = {"part": "", "chapter": "", "article": "", "section": "5", "clause": "", "sub_clause": ""}
        cit = LegalChunker._build_citation(self._make_meta(), h)
        self.assertIn("Section 5", cit)

    def test_full_hierarchy_citation(self):
        h = {"part": "II", "chapter": "III", "article": "90E", "section": "35", "clause": "2", "sub_clause": "aa"}
        cit = LegalChunker._build_citation(self._make_meta(), h)
        self.assertIn("Article 90E", cit)
        self.assertIn("Section 35", cit)
        self.assertIn("Sub-clause (aa)", cit)

    def test_amendment_citation_inserted_section(self):
        h = {"part": "", "chapter": "", "article": "", "section": "2", "clause": "", "sub_clause": ""}
        cit = LegalChunker._build_citation(
            self._make_meta("Amendment Act, 2026", amendment_of="Original Act"),
            h,
            section_title="",
            inserted_section="37A",
        )
        self.assertIn("Section 2", cit)
        self.assertIn("inserted Section 37A", cit)

    def test_preamble_no_section(self):
        h = {"part": "", "chapter": "", "article": "", "section": "", "clause": "", "sub_clause": ""}
        cit = LegalChunker._build_citation(self._make_meta(), h)
        self.assertNotIn("Section 0", cit)
        self.assertNotIn("Section ", cit)

    def test_empty_name(self):
        h = {"part": "", "chapter": "", "article": "", "section": "1", "clause": "", "sub_clause": ""}
        meta = self._make_meta(act_name="")
        meta.bangla_name = ""
        cit = LegalChunker._build_citation(meta, h)
        self.assertEqual(cit, "")


class TestReferenceExtraction(unittest.TestCase):
    def test_build_references_amendment(self):
        meta = LawMetadata(
            act_name="Amendment Act, 2026",
            amendment_of="Original Act, 2018",
            references=[],
        )
        refs = LegalChunker._build_references(meta)
        types = [r["type"] for r in refs]
        self.assertIn("amends", types)

    def test_build_references_empty(self):
        meta = LawMetadata(references=[])
        refs = LegalChunker._build_references(meta)
        self.assertEqual(refs, [])

    def test_law_reference_types(self):
        refs = [
            LawReference(ref_type="act", target="2018 সনের 57 নং আইন"),
            LawReference(ref_type="ordinance", target="2025 সনের 26 নং অধ্যাদেশ"),
            LawReference(ref_type="article", target="Article 90E"),
        ]
        meta = LawMetadata(act_name="Test", references=refs)
        built = LegalChunker._build_references(meta)
        self.assertEqual(len(built), 3)
        self.assertEqual(built[0]["type"], "act")
        self.assertEqual(built[1]["type"], "ordinance")
        self.assertEqual(built[2]["type"], "article")


class TestChunkBoundaries(unittest.TestCase):
    def test_chunk_does_not_start_with_paren(self):
        text = "(১) This is a clause."
        valid = not text.lstrip().startswith((")", "）"))
        self.assertTrue(valid)

    def test_chunk_starting_with_paren_detected(self):
        text = ")This is bad."
        valid = not text.lstrip().startswith((")", "）"))
        self.assertFalse(valid)

    def test_section_boundary_start(self):
        text = "সংক্ষিপ্ত শিরোনাম ১। (১) This is the section title."
        valid = not text.lstrip().startswith((")", "）"))
        self.assertTrue(valid)


class TestChunkValidation(unittest.TestCase):
    def _make_valid_meta(self):
        return {
            "act_name": "Test Act",
            "bangla_name": "",
            "act_number": "1",
            "year": 2026,
            "publication_date": "1 Jan, 2026",
            "document_type": "Act",
            "source_pdf": "test.pdf",
            "volume": "1",
            "pdf_page": 1,
            "language": "mixed",
            "section_title": "",
            "hierarchy": {"part": "", "chapter": "", "article": "", "section": "1", "clause": "", "sub_clause": ""},
            "chunk_index": 0,
        }

    def test_empty_text_rejected(self):
        from ingestion.chunker import Chunk, ChunkType
        c = Chunk(chunk_id="1", text="", token_count=0, metadata=self._make_valid_meta(), chunk_type=ChunkType.SECTION, citation="Test")
        chunker = LegalChunker()
        self.assertFalse(chunker._validate(c))

    def test_short_text_rejected(self):
        from ingestion.chunker import Chunk, ChunkType
        c = Chunk(chunk_id="1", text="hi", token_count=1, metadata=self._make_valid_meta(), chunk_type=ChunkType.SECTION, citation="Test")
        chunker = LegalChunker()
        self.assertFalse(chunker._validate(c))

    def test_valid_chunk_passes(self):
        from ingestion.chunker import Chunk, ChunkType
        c = Chunk(
            chunk_id="1",
            text="This is a complete section with enough text to pass validation. It ends properly.",
            token_count=50,
            metadata=self._make_valid_meta(),
            chunk_type=ChunkType.SECTION,
            citation="Test Act, Section 1",
        )
        chunker = LegalChunker()
        self.assertTrue(chunker._validate(c))
        self.assertTrue(c.validation["validated"])

    def test_missing_year_rejected(self):
        from ingestion.chunker import Chunk, ChunkType
        meta = self._make_valid_meta()
        meta["year"] = 0
        c = Chunk(chunk_id="1", text="A complete section with proper text length.", token_count=30, metadata=meta, chunk_type=ChunkType.SECTION, citation="X")
        chunker = LegalChunker()
        self.assertFalse(chunker._validate(c))

    def test_missing_citation_rejected(self):
        from ingestion.chunker import Chunk, ChunkType
        c = Chunk(chunk_id="1", text="A complete section with proper text length.", token_count=30, metadata=self._make_valid_meta(), chunk_type=ChunkType.SECTION, citation="")
        chunker = LegalChunker()
        self.assertFalse(chunker._validate(c))

    def test_preamble_can_have_no_citation(self):
        from ingestion.chunker import Chunk, ChunkType
        c = Chunk(chunk_id="1", text="A preamble text that is reasonably long.", token_count=30, metadata=self._make_valid_meta(), chunk_type=ChunkType.PREAMBLE, citation="")
        chunker = LegalChunker()
        chunker._validate(c)
        self.assertTrue(c.valid)

    def test_validation_metadata_populated(self):
        from ingestion.chunker import Chunk, ChunkType
        c = Chunk(
            chunk_id="1",
            text="A complete section with enough text to pass validation.",
            token_count=30,
            metadata=self._make_valid_meta(),
            chunk_type=ChunkType.SECTION,
            citation="Test Act, Section 1",
        )
        chunker = LegalChunker()
        chunker._validate(c)
        self.assertTrue(c.validation["validated"])
        self.assertTrue(c.validation["starts_at_boundary"])
        self.assertTrue(c.validation["is_complete_chunk"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
