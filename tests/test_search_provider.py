"""Unit tests for SearchProvider implementations (MockSearchProvider and TavilySearchProvider).

All tests execute completely offline using mock responses. No external network calls
or real API tokens are used.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests

from src.search_provider import (
    SearchProvider,
    SearchResult,
    SearchResponse,
    SearchStatus,
    MockSearchProvider,
    TavilySearchProvider,
    extract_domain_from_url,
    get_default_search_provider,
    is_real_search_configured,
)
from src.claim_extractor import Claim
from src.evidence_retriever import retrieve_evidence_for_claim


class TestSearchProvider(unittest.TestCase):
    """Test suite for search provider abstraction and mock behaviors."""

    def test_extract_domain_from_url(self):
        self.assertEqual(extract_domain_from_url("https://www.bbc.com/news/world-123"), "bbc.com")
        self.assertEqual(extract_domain_from_url("http://rbi.org.in/scripts/press.aspx"), "rbi.org.in")
        self.assertEqual(extract_domain_from_url("https://sub.domain.gov.in:8080/path"), "sub.domain.gov.in")
        self.assertEqual(extract_domain_from_url("not-a-url"), "")

    def test_search_result_domain_auto_population(self):
        res = SearchResult(
            title="Sample News",
            url="https://www.reuters.com/business/finance-article",
            domain="",
            snippet="A snippet about finance.",
        )
        self.assertEqual(res.domain, "reuters.com")

    def test_mock_search_provider_canned_queries(self):
        provider = MockSearchProvider()
        mock_items = [
            SearchResult(
                title="Mock RBI Report",
                url="https://rbi.org.in/mock-report",
                domain="rbi.org.in",
                snippet="Repo rate held at 6.5%.",
                publication_date="2024-02-08",
                rank=1,
            )
        ]
        provider.set_mock_results("rbi repo rate", mock_items)

        response = provider.search("RBI repo rate 6.5%", max_results=3)
        self.assertEqual(response.status, SearchStatus.SUCCESS)
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].title, "Mock RBI Report")

    def test_mock_search_provider_forced_rate_limit(self):
        provider = MockSearchProvider(
            force_status=SearchStatus.RATE_LIMITED,
            error_message="Too many requests (HTTP 429)",
        )
        response = provider.search("any query")
        self.assertEqual(response.status, SearchStatus.RATE_LIMITED)
        self.assertEqual(len(response.results), 0)
        self.assertIn("Too many requests", response.error_message)

    def test_mock_search_provider_forced_network_failure(self):
        provider = MockSearchProvider(
            force_status=SearchStatus.NETWORK_FAILURE,
            error_message="Connection timeout",
        )
        response = provider.search("any query")
        self.assertEqual(response.status, SearchStatus.NETWORK_FAILURE)
        self.assertEqual(len(response.results), 0)

    def test_mock_search_provider_forced_no_results(self):
        provider = MockSearchProvider(force_status=SearchStatus.NO_RESULTS)
        response = provider.search("unknown query")
        self.assertEqual(response.status, SearchStatus.NO_RESULTS)
        self.assertEqual(len(response.results), 0)


class TestTavilySearchProvider(unittest.TestCase):
    """Offline unit tests for TavilySearchProvider using mocked HTTP calls."""

    def setUp(self):
        self.provider = TavilySearchProvider(api_key="tvly-test-dummy-key")

    # 1. Valid API response
    @patch("requests.post")
    def test_tavily_valid_api_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "query": "RBI repo rate 6.5%",
            "results": [
                {
                    "title": "RBI Monetary Policy Feb 2024",
                    "url": "https://www.rbi.org.in/press/2024-repo",
                    "content": "The repo rate was kept steady at 6.50% by the MPC.",
                    "published_date": "2024-02-08T10:00:00Z",
                },
                {
                    "title": "Reuters: RBI rate decision analysis",
                    "url": "https://www.reuters.com/business/finance/rbi-rate",
                    "content": "Reserve Bank of India maintains policy repo rate.",
                    "published_date": "2024-02-08",
                },
            ],
        }
        mock_post.return_value = mock_resp

        response = self.provider.search("RBI repo rate 6.5%", max_results=3)

        self.assertEqual(response.status, SearchStatus.SUCCESS)
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.results[0].title, "RBI Monetary Policy Feb 2024")
        self.assertEqual(response.results[0].url, "https://www.rbi.org.in/press/2024-repo")
        self.assertEqual(response.results[0].domain, "rbi.org.in")
        self.assertEqual(response.results[0].publication_date, "2024-02-08")
        self.assertEqual(response.results[0].rank, 1)
        self.assertEqual(response.results[1].rank, 2)

    # 2. Multiple results respect max_results bound
    @patch("requests.post")
    def test_tavily_results_respect_max_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "query": "test query",
            "results": [
                {"title": f"Title {i}", "url": f"https://example{i}.com", "content": f"Snippet {i}"}
                for i in range(1, 10)
            ],
        }
        mock_post.return_value = mock_resp

        response = self.provider.search("test query", max_results=3)
        self.assertEqual(response.status, SearchStatus.SUCCESS)
        self.assertEqual(len(response.results), 3)

    # 3. Zero results
    @patch("requests.post")
    def test_tavily_zero_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"query": "obscure statement", "results": []}
        mock_post.return_value = mock_resp

        response = self.provider.search("obscure statement")
        self.assertEqual(response.status, SearchStatus.NO_RESULTS)
        self.assertEqual(len(response.results), 0)

    # 4. Missing API key
    def test_tavily_missing_api_key(self):
        empty_provider = TavilySearchProvider(api_key="")
        response = empty_provider.search("test query")
        self.assertEqual(response.status, SearchStatus.PROVIDER_UNCONFIGURED)
        self.assertIn("missing", response.error_message.lower())

    # 5. Invalid API key (HTTP 401 / 403)
    @patch("requests.post")
    def test_tavily_invalid_api_key(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"detail":{"error":"Unauthorized"}}'
        mock_post.return_value = mock_resp

        response = self.provider.search("test query")
        self.assertEqual(response.status, SearchStatus.API_FAILURE)
        self.assertIn("401", response.error_message)

    # 6. Rate limit (HTTP 429)
    @patch("requests.post")
    def test_tavily_rate_limit(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = '{"detail":{"error":"Too Many Requests"}}'
        mock_post.return_value = mock_resp

        response = self.provider.search("test query")
        self.assertEqual(response.status, SearchStatus.RATE_LIMITED)
        self.assertIn("429", response.error_message)

    # 7. Network failure (ConnectionError)
    @patch("requests.post")
    def test_tavily_network_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        response = self.provider.search("test query")
        self.assertEqual(response.status, SearchStatus.NETWORK_FAILURE)
        self.assertIn("connect", response.error_message.lower())

    # 8. Timeout
    @patch("requests.post")
    def test_tavily_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        response = self.provider.search("test query")
        self.assertEqual(response.status, SearchStatus.NETWORK_FAILURE)
        self.assertIn("timed out", response.error_message.lower())

    # 9. Malformed API response (Invalid JSON or unexpected format)
    @patch("requests.post")
    def test_tavily_malformed_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_resp

        response = self.provider.search("test query")
        self.assertEqual(response.status, SearchStatus.INVALID_RESPONSE)

    # 10. Integration with EvidenceRetriever preserving claim_id
    @patch("requests.post")
    def test_tavily_integration_with_evidence_retriever(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "query": "RBI repo rate 6.5%",
            "results": [
                {
                    "title": "Government Press Release",
                    "url": "https://pib.gov.in/PressReleasePage.aspx?PRID=12345",
                    "content": "The government noted the RBI rate decision.",
                    "published_date": "2024-02-08",
                }
            ],
        }
        mock_post.return_value = mock_resp

        claim = Claim(claim_id="C001", text="The Reserve Bank of India increased the repo rate to 6.5%.", source_sentence_index=0)
        evidence_result = retrieve_evidence_for_claim(claim, self.provider)

        self.assertEqual(evidence_result.status, SearchStatus.SUCCESS)
        self.assertEqual(len(evidence_result.evidence_items), 1)
        item = evidence_result.evidence_items[0]
        self.assertEqual(item.claim_id, "C001")
        self.assertEqual(item.evidence_id, "E001")
        self.assertEqual(item.domain, "pib.gov.in")
        self.assertTrue(item.is_priority_source)
        self.assertEqual(item.priority_tier, "Government / Official")


if __name__ == "__main__":
    unittest.main()
