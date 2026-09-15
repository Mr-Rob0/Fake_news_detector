"""Unit tests for the claim extraction module (Phase 4 foundation)."""

import unittest
from src.claim_extractor import (
    Claim,
    split_into_sentences,
    is_viable_claim,
    extract_claims,
)


class TestClaimExtractor(unittest.TestCase):
    """Test suite for claim extraction, sentence splitting, and filtering."""

    def test_split_into_sentences_preserves_abbreviations(self):
        text = "Dr. Smith met with the U.S. envoy in Jan. 2024. The repo rate was 6.5% according to the RBI."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("Dr. Smith met with the U.S. envoy in Jan. 2024.", sentences[0])
        self.assertIn("6.5%", sentences[1])

    def test_split_empty_text(self):
        self.assertEqual(split_into_sentences(""), [])
        self.assertEqual(split_into_sentences("   "), [])

    def test_filter_excludes_questions(self):
        self.assertFalse(is_viable_claim("Did the government increase taxes by 10%?"))

    def test_filter_excludes_short_fragments(self):
        self.assertFalse(is_viable_claim("Too short."))
        self.assertFalse(is_viable_claim("No."))

    def test_filter_excludes_pure_opinions(self):
        self.assertFalse(is_viable_claim("In my opinion the movie was really wonderful and exciting."))
        self.assertFalse(is_viable_claim("I think this is not fair to anyone involved."))

    def test_filter_retains_factual_statements(self):
        self.assertTrue(is_viable_claim("The Reserve Bank of India increased the repo rate to 6.5%."))
        self.assertTrue(is_viable_claim("NASA launched a lunar mission with a budget of $20 billion."))

    def test_extract_claims_single_claim(self):
        text = "The Reserve Bank of India increased the repo rate to 6.5% on Thursday."
        claims = extract_claims(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_id, "C001")
        self.assertEqual(claims[0].text, text)

    def test_extract_claims_multiple_ordered_ids(self):
        text = (
            "The Reserve Bank of India increased the repo rate to 6.5%. "
            "How does this affect normal citizens? "
            "In my opinion everyone should save money. "
            "The finance ministry announced new tax slabs for FY 2025. "
            "Automobile sales fell by 12 percent across the country."
        )
        claims = extract_claims(text, max_claims=3)
        self.assertEqual(len(claims), 3)
        self.assertEqual([c.claim_id for c in claims], ["C001", "C002", "C003"])
        # Questions and pure opinions should have been excluded
        for claim in claims:
            self.assertFalse(claim.text.endswith("?"))
            self.assertNotIn("In my opinion", claim.text)

    def test_extract_claims_empty_or_whitespace(self):
        self.assertEqual(extract_claims(""), [])
        self.assertEqual(extract_claims("   \n\t  "), [])


if __name__ == "__main__":
    unittest.main()
