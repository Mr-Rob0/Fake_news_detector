"""Unit tests for the Evidence Retrieval module (Phase 5).

Covers all required scenarios using mocked search provider responses:
- Query generation (single and multiple claims)
- Structured mapping and field preservation (title, URL, domain, date)
- Error statuses (no-results, API failure, rate limiting, network error)
- Multiple sources, limits, and domain diversity
- Traceability between claim IDs and evidence items
- Strict absence of fabricated evidence or verdicts
"""

import unittest
from src.claim_extractor import Claim
from src.search_provider import (
    SearchResult,
    SearchResponse,
    SearchStatus,
    MockSearchProvider,
)
from src.evidence_retriever import (
    SearchQuery,
    EvidenceItem,
    ClaimEvidenceResult,
    generate_search_query,
    classify_source_priority,
    retrieve_evidence_for_claim,
    retrieve_evidence_for_claims,
)


class TestEvidenceRetriever(unittest.TestCase):
    """Automated test suite for Evidence Retrieval using mocked search responses."""

    def setUp(self):
        self.claim1 = Claim(
            claim_id="C001",
            text="The Reserve Bank of India increased the repo rate to 6.5%.",
            source_sentence_index=0,
        )
        self.claim2 = Claim(
            claim_id="C002",
            text="NASA successfully launched the Artemis lunar satellite mission on Monday.",
            source_sentence_index=1,
        )

    # 1. One claim -> search query generated
    def test_01_single_claim_generates_focused_query(self):
        query_obj = generate_search_query(self.claim1)
        self.assertEqual(query_obj.claim_id, "C001")
        self.assertIsInstance(query_obj.query_text, str)
        # Should preserve key entities & numbers, avoiding entire article or full stopwords
        self.assertTrue("RBI" in query_obj.query_text or "repo" in query_obj.query_text.lower())
        self.assertIn("6.5%", query_obj.query_text)
        self.assertNotIn("increased the repo rate to", query_obj.query_text.lower())

    # 2. Multiple claims -> each gets its own unique query
    def test_02_multiple_claims_get_distinct_queries(self):
        q1 = generate_search_query(self.claim1)
        q2 = generate_search_query(self.claim2)
        self.assertEqual(q1.claim_id, "C001")
        self.assertEqual(q2.claim_id, "C002")
        self.assertNotEqual(q1.query_text, q2.query_text)
        self.assertIn("6.5%", q1.query_text)
        self.assertIn("Artemis", q2.query_text)

    # 3. Search results correctly mapped to evidence items
    def test_03_search_results_correctly_mapped_to_evidence_items(self):
        provider = MockSearchProvider()
        mock_result = SearchResult(
            title="Monetary Policy Report 2024",
            url="https://www.rbi.org.in/press/report-101",
            domain="rbi.org.in",
            snippet="The Monetary Policy Committee decided to maintain the repo rate at 6.5%.",
            publication_date="2024-02-08",
            rank=1,
        )
        provider.set_mock_results("rbi repo rate 6.5%", [mock_result])

        res: ClaimEvidenceResult = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.status, SearchStatus.SUCCESS)
        self.assertEqual(len(res.evidence_items), 1)

        item = res.evidence_items[0]
        self.assertEqual(item.claim_id, "C001")
        self.assertEqual(item.evidence_id, "E001")
        self.assertEqual(item.title, "Monetary Policy Report 2024")
        self.assertEqual(item.url, "https://www.rbi.org.in/press/report-101")
        self.assertEqual(item.domain, "rbi.org.in")
        self.assertEqual(item.snippet, mock_result.snippet)
        self.assertEqual(item.publication_date, "2024-02-08")
        self.assertEqual(item.source_rank, 1)

    # 4. Source title preserved
    def test_04_source_title_preserved(self):
        provider = MockSearchProvider()
        expected_title = "Breaking: Official Economic Review 2024"
        mock_result = SearchResult(
            title=expected_title,
            url="https://reuters.com/article/1",
            domain="reuters.com",
            snippet="Snippet text",
        )
        provider.set_mock_results("rbi repo rate 6.5%", [mock_result])

        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.evidence_items[0].title, expected_title)

    # 5. URL preserved
    def test_05_url_preserved(self):
        provider = MockSearchProvider()
        expected_url = "https://www.reuters.com/business/finance/rbi-rate-decision"
        mock_result = SearchResult(
            title="Reuters Repo Rate",
            url=expected_url,
            domain="reuters.com",
            snippet="Snippet text",
        )
        provider.set_mock_results("rbi repo rate 6.5%", [mock_result])

        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.evidence_items[0].url, expected_url)

    # 6. Domain extracted correctly
    def test_06_domain_extracted_correctly(self):
        provider = MockSearchProvider()
        mock_result = SearchResult(
            title="BBC News Article",
            url="https://www.bbc.com/news/world-asia-india-67890",
            domain="",  # Auto-extracted in post_init or retriever
            snippet="Snippet text",
        )
        provider.set_mock_results("rbi repo rate 6.5%", [mock_result])

        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.evidence_items[0].domain, "bbc.com")

    # 7. Publication date handled when available and when missing
    def test_07_publication_date_handled_when_available_and_none(self):
        provider = MockSearchProvider()
        res_with_date = SearchResult(
            title="With Date",
            url="https://thehindu.com/article-1",
            domain="thehindu.com",
            snippet="Snippet 1",
            publication_date="2024-02-08",
        )
        res_no_date = SearchResult(
            title="Without Date",
            url="https://apnews.com/article-2",
            domain="apnews.com",
            snippet="Snippet 2",
            publication_date=None,
        )
        provider.set_mock_results("rbi repo rate 6.5%", [res_with_date, res_no_date])

        res = retrieve_evidence_for_claim(self.claim1, provider, max_sources=3)
        self.assertEqual(res.evidence_items[0].publication_date, "2024-02-08")
        self.assertIsNone(res.evidence_items[1].publication_date)

    # 8. No-result case
    def test_08_no_result_case_handled_gracefully(self):
        provider = MockSearchProvider(force_status=SearchStatus.NO_RESULTS)
        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.status, SearchStatus.NO_RESULTS)
        self.assertEqual(len(res.evidence_items), 0)
        self.assertEqual(res.source_count, 0)

    # 9. Search API failure
    def test_09_search_api_failure_handled(self):
        provider = MockSearchProvider(
            force_status=SearchStatus.API_FAILURE,
            error_message="External Search API 500 Internal Error",
        )
        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.status, SearchStatus.API_FAILURE)
        self.assertEqual(len(res.evidence_items), 0)
        self.assertIn("500 Internal Error", res.error_message)

    # 10. Rate-limit response
    def test_10_rate_limit_response_handled(self):
        provider = MockSearchProvider(
            force_status=SearchStatus.RATE_LIMITED,
            error_message="HTTP 429: Query quota exceeded",
        )
        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.status, SearchStatus.RATE_LIMITED)
        self.assertEqual(len(res.evidence_items), 0)
        self.assertIn("429", res.error_message)

    # 11. Network failure
    def test_11_network_failure_handled(self):
        provider = MockSearchProvider(
            force_status=SearchStatus.NETWORK_FAILURE,
            error_message="DNS resolution timeout",
        )
        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.status, SearchStatus.NETWORK_FAILURE)
        self.assertEqual(len(res.evidence_items), 0)
        self.assertIn("timeout", res.error_message)

    # 12. Multiple sources returned and bounded by limit
    def test_12_multiple_sources_bounded_by_limit(self):
        provider = MockSearchProvider()
        mock_list = [
            SearchResult(title=f"Source {i}", url=f"https://source{i}.com/news", domain=f"source{i}.com", snippet=f"Snippet {i}")
            for i in range(1, 6)
        ]
        provider.set_mock_results("rbi repo rate 6.5%", mock_list)

        # Bounded by max_sources=3
        res = retrieve_evidence_for_claim(self.claim1, provider, max_sources=3)
        self.assertEqual(res.status, SearchStatus.SUCCESS)
        self.assertEqual(len(res.evidence_items), 3)
        self.assertEqual([e.evidence_id for e in res.evidence_items], ["E001", "E002", "E003"])

    # 13. Claim/evidence association preserved (Traceability)
    def test_13_claim_evidence_traceability_preserved(self):
        provider = MockSearchProvider()
        mock_list1 = [SearchResult(title="RBI 1", url="https://rbi.org.in/1", domain="rbi.org.in", snippet="S1")]
        mock_list2 = [SearchResult(title="NASA 1", url="https://nasa.gov/1", domain="nasa.gov", snippet="S2")]
        provider.set_mock_results("rbi repo rate 6.5%", mock_list1)
        provider.set_mock_results("artemis", mock_list2)

        batch_results = retrieve_evidence_for_claims([self.claim1, self.claim2], provider)
        self.assertEqual(len(batch_results), 2)

        # Claim 1 traceability
        self.assertEqual(batch_results[0].claim.claim_id, "C001")
        for item in batch_results[0].evidence_items:
            self.assertEqual(item.claim_id, "C001")

        # Claim 2 traceability
        self.assertEqual(batch_results[1].claim.claim_id, "C002")
        for item in batch_results[1].evidence_items:
            self.assertEqual(item.claim_id, "C002")

    # 14. No fabricated evidence: empty or failed search never produces synthetic evidence
    def test_14_no_fabricated_evidence(self):
        provider = MockSearchProvider(force_status=SearchStatus.NO_RESULTS)
        res = retrieve_evidence_for_claim(self.claim1, provider)
        self.assertEqual(res.evidence_items, [])

    # 15. Source diversity: identical URLs deduplicated and excessive same-domain results capped
    def test_15_source_diversity_and_deduplication(self):
        provider = MockSearchProvider()
        redundant_results = [
            SearchResult(title="Story A", url="https://reuters.com/story-1", domain="reuters.com", snippet="Snippet A"),
            SearchResult(title="Story A Duplicate", url="https://reuters.com/story-1", domain="reuters.com", snippet="Snippet A dup"),
            SearchResult(title="Story B", url="https://reuters.com/story-2", domain="reuters.com", snippet="Snippet B"),
            SearchResult(title="Story C", url="https://reuters.com/story-3", domain="reuters.com", snippet="Snippet C"),
            SearchResult(title="Story D", url="https://bbc.com/story-4", domain="bbc.com", snippet="Snippet D"),
        ]
        provider.set_mock_results("rbi repo rate 6.5%", redundant_results)

        # max_sources=3, max_per_domain=2
        res = retrieve_evidence_for_claim(self.claim1, provider, max_sources=3, max_per_domain=2)
        domains = [item.domain for item in res.evidence_items]
        # At most 2 from reuters.com
        self.assertLessEqual(domains.count("reuters.com"), 2)
        # bbc.com should be included to ensure diversity
        self.assertIn("bbc.com", domains)
        # No duplicate URLs
        urls = [item.url for item in res.evidence_items]
        self.assertEqual(len(urls), len(set(urls)))

    # 16. Priority tier categorization
    def test_16_priority_tier_classification(self):
        is_gov, tier_gov = classify_source_priority("pib.gov.in")
        self.assertTrue(is_gov)
        self.assertEqual(tier_gov, "Government / Official")

        is_inst, tier_inst = classify_source_priority("rbi.org.in")
        self.assertTrue(is_inst)
        self.assertEqual(tier_inst, "Institutional / Regulatory")

        is_news, tier_news = classify_source_priority("reuters.com")
        self.assertTrue(is_news)
        self.assertEqual(tier_news, "Reputable News")

        is_gen, tier_gen = classify_source_priority("randomblog.xyz")
        self.assertFalse(is_gen)
        self.assertEqual(tier_gen, "General")


if __name__ == "__main__":
    unittest.main()
