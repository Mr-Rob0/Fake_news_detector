"""Automated tests for input handling and validation logic."""

import unittest
from src.input_handler import (
    validate_text,
    validate_url,
    validate_and_normalize_input,
    MIN_TEXT_LENGTH,
    MIN_WORD_COUNT
)


class TestInputHandler(unittest.TestCase):
    """Test suite for news input validation and normalization."""

    def test_both_inputs_empty(self):
        result = validate_and_normalize_input("", "")
        self.assertFalse(result.is_valid)
        self.assertIn("Please provide either news text or an article URL", result.error_message)
        self.assertIsNone(result.normalized_input)

    def test_both_inputs_whitespace_only(self):
        result = validate_and_normalize_input("   ", "   ")
        self.assertFalse(result.is_valid)
        self.assertIn("Please provide either news text or an article URL", result.error_message)

    def test_both_inputs_provided(self):
        result = validate_and_normalize_input(
            "Scientists discover water on Mars in underground lakes.",
            "https://www.bbc.com/news/science-123"
        )
        self.assertFalse(result.is_valid)
        self.assertIn("not both", result.error_message)
        self.assertIsNone(result.normalized_input)

    def test_text_too_short(self):
        short_text = "Breaking news!"
        self.assertLess(len(short_text), MIN_TEXT_LENGTH)
        result = validate_and_normalize_input(short_text, "")
        self.assertFalse(result.is_valid)
        self.assertIn("too short", result.error_message)

    def test_text_too_few_words(self):
        few_words = "Supercalifragilisticexpialidocious_headline"
        self.assertGreaterEqual(len(few_words), MIN_TEXT_LENGTH)
        result = validate_and_normalize_input(few_words, "")
        self.assertFalse(result.is_valid)
        self.assertIn(f"at least {MIN_WORD_COUNT} words", result.error_message)

    def test_text_valid_normalization(self):
        raw_text = "   The local authorities declared a public holiday tomorrow due to severe rainfall.   \n"
        result = validate_and_normalize_input(raw_text, "")
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.normalized_input)
        self.assertEqual(result.normalized_input.input_type, "text")
        self.assertEqual(
            result.normalized_input.cleaned_content,
            "The local authorities declared a public holiday tomorrow due to severe rainfall."
        )
        self.assertEqual(result.normalized_input.metadata["word_count"], 12)
        self.assertEqual(
            result.normalized_input.metadata["character_count"],
            len("The local authorities declared a public holiday tomorrow due to severe rainfall.")
        )

    def test_url_missing_scheme(self):
        result = validate_and_normalize_input("", "www.reuters.com/world/article1")
        self.assertFalse(result.is_valid)
        self.assertIn("protocol", result.error_message.lower())

    def test_url_unsupported_protocol(self):
        result = validate_and_normalize_input("", "ftp://reuters.com/files/news.txt")
        self.assertFalse(result.is_valid)
        self.assertIn("protocol 'ftp'", result.error_message.lower())

    def test_url_with_spaces(self):
        result = validate_and_normalize_input("", "https://example .com/news")
        self.assertFalse(result.is_valid)
        self.assertIn("spaces", result.error_message.lower())

    def test_url_missing_valid_domain(self):
        result = validate_and_normalize_input("", "https://invalid_host")
        self.assertFalse(result.is_valid)
        self.assertIn("domain", result.error_message.lower())

    def test_url_valid_https(self):
        valid_url = "https://www.thehindu.com/news/national/article.html"
        result = validate_and_normalize_input("", valid_url)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.normalized_input)
        self.assertEqual(result.normalized_input.input_type, "url")
        self.assertEqual(result.normalized_input.cleaned_content, valid_url)
        self.assertEqual(result.normalized_input.metadata["domain"], "www.thehindu.com")
        self.assertEqual(result.normalized_input.metadata["scheme"], "https")


if __name__ == "__main__":
    unittest.main()
