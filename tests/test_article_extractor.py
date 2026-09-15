"""Automated unit tests for article extraction and status classification using mocked HTTP responses."""

import unittest
from unittest.mock import patch, MagicMock
import requests
from bs4 import BeautifulSoup
from src.article_extractor import (
    extract_article_from_url,
    extract_article_from_html,
    clean_text,
    extract_title,
    ExtractionStatus,
    MIN_ARTICLE_WORDS
)


class TestArticleExtractor(unittest.TestCase):
    """Test suite covering article extraction, text normalization, and error classification."""

    def test_text_cleaning(self):
        raw = "  This   is a    sentence with   tabs\t\tand   spaces. \n\n\n\nAnother paragraph here.  "
        cleaned = clean_text(raw)
        self.assertEqual(
            cleaned,
            "This is a sentence with tabs and spaces.\n\nAnother paragraph here."
        )

    def test_title_extraction_opengraph_priority(self):
        html = """
        <html>
            <head>
                <meta property="og:title" content="OpenGraph Headline News" />
                <title>Old Title Tag - Example News</title>
            </head>
            <body><h1>Header Headline</h1></body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(extract_title(soup), "OpenGraph Headline News")

    def test_title_extraction_fallback_to_h1(self):
        html = """
        <html>
            <head><title>Page Title Tag - Site Name</title></head>
            <body><h1>H1 Primary Article Headline</h1></body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(extract_title(soup), "H1 Primary Article Headline")

    def test_title_extraction_fallback_to_title_tag_with_suffix_cleaned(self):
        html = """
        <html>
            <head><title>Breaking News Story About Science - Global News Daily</title></head>
            <body></body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(extract_title(soup), "Breaking News Story About Science")

    def test_successful_article_extraction(self):
        sample_html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Scientists Announce Major Breakthrough in Clean Energy - Tech Daily</title>
                <meta property="og:title" content="Scientists Announce Major Breakthrough in Clean Energy" />
                <link rel="canonical" href="https://techdaily.com/articles/clean-energy" />
            </head>
            <body>
                <header><nav><a href="/">Home</a><a href="/news">News</a></nav></header>
                <article>
                    <h1>Scientists Announce Major Breakthrough in Clean Energy</h1>
                    <p>Researchers at the International Renewable Lab confirmed a significant breakthrough in fusion technology today.</p>
                    <p>The newly developed containment system managed to maintain stable plasma temperatures for over fifteen minutes consecutively.</p>
                    <p>Engineers stated that this commercial milestone could drastically reduce the timeline needed for scalable clean electricity across multiple power grids globally.</p>
                </article>
                <footer><p>© 2026 Tech Daily. All rights reserved.</p></footer>
            </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.text = sample_html
        mock_resp.url = "https://techdaily.com/articles/clean-energy"

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://techdaily.com/articles/clean-energy")

        self.assertTrue(result.success)
        self.assertEqual(result.status, ExtractionStatus.SUCCESS)
        self.assertIsNotNone(result.article)
        self.assertEqual(result.article.title, "Scientists Announce Major Breakthrough in Clean Energy")
        self.assertEqual(result.article.domain, "techdaily.com")
        self.assertEqual(result.article.canonical_url, "https://techdaily.com/articles/clean-energy")
        self.assertIn("breakthrough in fusion technology", result.article.text)
        self.assertNotIn("© 2026 Tech Daily", result.article.text)
        self.assertGreaterEqual(result.article.word_count, MIN_ARTICLE_WORDS)

    def test_empty_html_response(self):
        result = extract_article_from_html("", "https://example.com/empty")
        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.EMPTY_PAGE)
        self.assertIn("empty", result.error_message.lower())

    def test_article_content_missing(self):
        empty_page = "<html><body><header>Nav</header><footer>Footer</footer></body></html>"
        result = extract_article_from_html(empty_page, "https://example.com/no-content")
        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.NO_ARTICLE_CONTENT)
        self.assertIn("no readable article content", result.error_message.lower())

    def test_article_content_too_short(self):
        short_html = """
        <html>
            <body>
                <h1>Quick Headline</h1>
                <p>A few short words here.</p>
            </body>
        </html>
        """
        result = extract_article_from_html(short_html, "https://example.com/short")
        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.CONTENT_TOO_SHORT)
        self.assertIn("too short", result.error_message.lower())

    def test_http_401_access_denied(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://reuters.com/world/exclusive")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.ACCESS_DENIED)
        self.assertIn("did not allow automated access", result.error_message)
        self.assertIn("paste", result.suggested_action.lower())

    def test_http_403_access_denied(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://apnews.com/article/press-release")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.ACCESS_DENIED)
        self.assertIn("did not allow automated access", result.error_message)
        self.assertIn("paste", result.suggested_action.lower())

    def test_http_404_not_found(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://example.com/deleted-article")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.NOT_FOUND)
        self.assertIn("not found", result.error_message.lower())

    def test_http_429_rate_limited(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://example.com/busy-server")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.RATE_LIMITED)
        self.assertIn("rate limiting", result.error_message.lower())

    def test_http_500_server_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://example.com/crash")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.SERVER_ERROR)
        self.assertIn("temporarily unavailable", result.error_message.lower())

    def test_timeout_failure(self):
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Timed out")):
            result = extract_article_from_url("https://slow.example.com/news")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.TIMEOUT)
        self.assertIn("timed out", result.error_message.lower())

    def test_connection_network_failure(self):
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("DNS failure")):
            result = extract_article_from_url("https://unreachable.example.com")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.NETWORK_ERROR)
        self.assertIn("failed to establish a network connection", result.error_message.lower())

    def test_non_html_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.url = "https://example.com/document.pdf"

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://example.com/document.pdf")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.NON_HTML)
        self.assertIn("non-html", result.error_message.lower())

    def test_invalid_url_protocol(self):
        result = extract_article_from_url("ftp://example.com/news.txt")
        self.assertFalse(result.success)
        self.assertEqual(result.status, ExtractionStatus.INVALID_URL)
        self.assertIn("unsupported", result.error_message.lower())

    def test_redirect_handling(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = """
        <html>
            <head><title>Redirected Article Story Title</title></head>
            <body>
                <article>
                    <p>First substantive paragraph containing plenty of interesting journalistic details regarding the recent summit.</p>
                    <p>Second substantive paragraph explaining the key bilateral agreements signed by international representatives.</p>
                    <p>Third substantive paragraph providing additional background context on the geopolitical implications of the treaty.</p>
                </article>
            </body>
        </html>
        """
        mock_resp.url = "https://example.com/final-destination"

        with patch("requests.get", return_value=mock_resp):
            result = extract_article_from_url("https://example.com/short-link")

        self.assertTrue(result.success)
        self.assertEqual(result.status, ExtractionStatus.SUCCESS)
        self.assertEqual(result.article.final_url, "https://example.com/final-destination")

    def test_nested_boilerplate_decomposed_attrs_regression(self):
        """Regression test for AttributeError when nested tags in boilerplate are decomposed."""
        html_with_nested_boilerplate = """
        <!DOCTYPE html>
        <html>
            <head><title>Jagran-Style Article With Nested Ads - News Site</title></head>
            <body>
                <div class="cookie-banner-wrapper">
                    <div class="cookie-box">
                        <span class="cookie-label">Accept cookies</span>
                    </div>
                </div>
                <div id="ad-container-top">
                    <div class="ad-slot"><span class="ad-text">Sponsor</span></div>
                </div>
                <article>
                    <h1>Jagran-Style Article With Nested Ads</h1>
                    <p>Parliamentary discussions resumed earlier this morning with cross-party dialogue on key legislative priorities.</p>
                    <p>Party leaders expressed their strong commitments to constitutional accountability and swift legal governance.</p>
                    <p>Further updates from parliamentary committees are expected throughout the remaining days of the current legislative session.</p>
                </article>
            </body>
        </html>
        """
        result = extract_article_from_html(html_with_nested_boilerplate, "https://example.com/politics/news-item")
        self.assertTrue(result.success)
        self.assertEqual(result.status, ExtractionStatus.SUCCESS)
        self.assertIn("Parliamentary discussions resumed", result.article.text)
        self.assertNotIn("Accept cookies", result.article.text)
        self.assertNotIn("Sponsor", result.article.text)

    def test_internal_error_classification_not_server_error(self):
        """Ensures that application/parser exceptions are classified as INTERNAL_ERROR, never SERVER_ERROR."""
        with patch("src.article_extractor.extract_article_from_html", side_effect=AttributeError("Simulated bug")):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"Content-Type": "text/html"}
            mock_resp.text = "<html><body>Some content</body></html>"

            with patch("requests.get", return_value=mock_resp):
                result = extract_article_from_url("https://example.com/test")

            self.assertFalse(result.success)
            self.assertEqual(result.status, ExtractionStatus.INTERNAL_ERROR)
            self.assertNotEqual(result.status, ExtractionStatus.SERVER_ERROR)
            self.assertIn("internal error", result.error_message.lower())

    def test_related_articles_removal_english_and_hindi(self):
        """Ensures that in-article related stories and 'Also Read' links are stripped."""
        html = """
        <html>
            <head><title>Finance Ministry Announces New Economic Policies - Daily News</title></head>
            <body>
                <article>
                    <h1>Finance Ministry Announces New Economic Policies</h1>
                    <p>The finance minister presented a comprehensive series of fiscal measures earlier this afternoon.</p>
                    <p>Also Read: Top five market indicators to watch closely before investing in commodities.</p>
                    <p>Officials highlighted that revenue growth has maintained consistent trajectory over consecutive quarters.</p>
                    <p>यह भी पढ़ें: बजट से जुड़ी पांच अहम बातें जो हर नागरिक के लिए जानना जरूरी हैं।</p>
                    <p>The new framework aims to encourage sustainable domestic manufacturing and strengthen supply chains.</p>
                </article>
            </body>
        </html>
        """
        result = extract_article_from_html(html, "https://example.com/finance/news")
        self.assertTrue(result.success)
        self.assertIn("The finance minister presented", result.article.text)
        self.assertIn("Officials highlighted that revenue growth", result.article.text)
        self.assertIn("The new framework aims to encourage", result.article.text)
        self.assertNotIn("Also Read:", result.article.text)
        self.assertNotIn("यह भी पढ़ें:", result.article.text)

    def test_word_boundary_spacing_across_inline_tags(self):
        """Ensures words across inline tags like <span> or <b> are not fused together."""
        html = """
        <html>
            <head><title>Headline Test - Example</title></head>
            <body>
                <article>
                    <p>Opposition leader <span>Mallikarjun</span> <b>Kharge</b> submitted an official inquiry letter.</p>
                    <p>The committee agreed to review the requested documentation during their scheduled weekly assembly.</p>
                    <p>A formal statement from the party spokesperson is expected later this evening for journalists.</p>
                </article>
            </body>
        </html>
        """
        result = extract_article_from_html(html, "https://example.com/test-spacing")
        self.assertTrue(result.success)
        self.assertIn("Opposition leader Mallikarjun Kharge submitted", result.article.text)
        self.assertNotIn("MallikarjunKharge", result.article.text)

    def test_photo_credit_exclusion_in_figcaption(self):
        """Ensures photo credits in figcaption are excluded while genuine paragraphs remain intact."""
        html = """
        <html>
            <head><title>Summit Coverage - News</title></head>
            <body>
                <article>
                    <h1>Summit Coverage</h1>
                    <figure>
                        <figcaption>Photo: PTI / Representative image</figcaption>
                    </figure>
                    <p>International delegates gathered at the convention hall for initial bilateral consultations.</p>
                    <p>Discussions centered primarily around regional security arrangements and shared environmental standards.</p>
                    <p>Final signatures on the multilateral pact are scheduled for the concluding ceremony on Sunday.</p>
                </article>
            </body>
        </html>
        """
        result = extract_article_from_html(html, "https://example.com/summit")
        self.assertTrue(result.success)
        self.assertIn("International delegates gathered", result.article.text)
        self.assertNotIn("Photo: PTI", result.article.text)


if __name__ == "__main__":
    unittest.main()
