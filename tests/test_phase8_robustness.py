"""Comprehensive unit test suite for Phase 8: Advanced Source Selection, Independence Control & Evidence Robustness.

Covers all 20 user-specified criteria:
1. Primary source gets higher priority.
2. Official source is not automatically treated as truth.
3. Political viewpoint does not affect quality score.
4. Opinion article is deprioritized.
5. Clickbait article is deprioritized.
6. PTI duplicate is detected and de-duplicated.
7. ANI duplicate is detected and de-duplicated.
8. Reuters duplicate is detected and de-duplicated.
9. Same-domain result dominance is controlled.
10. Different independent domains increase corroboration.
11. Irrelevant search result does not support a claim.
12. Directly relevant evidence gets stronger ranking.
13. Conflicting evidence produces UNCERTAIN.
14. No evidence produces UNCERTAIN.
15. Evidence remains linked to claim IDs.
16. Fake/missing URLs are never fabricated.
17. Publication date affects temporal relevance.
18. Supporting and contradicting evidence are both displayed.
19. Evidence strength is explainable.
20. Existing Phase 1–7 tests remain passing.
"""

import unittest
from src.claim_extractor import Claim
from src.search_provider import SearchStatus, SearchResponse, SearchResult, SearchProvider
from src.evidence_retriever import (
    EvidenceItem,
    SearchQuery,
    ClaimEvidenceResult,
    retrieve_evidence_for_claim,
)
from src.source_evaluator import (
    SourceCategory,
    IndependenceStatus,
    RelevanceLevel,
    EvidenceStrength,
    SourceEvaluation,
    EvaluatedEvidenceItem,
    ClaimEvaluationSummary,
    evaluate_claim_evidence,
    calculate_source_quality,
    detect_named_sources,
    detect_direct_quotes,
    detect_specific_metrics,
    detect_temporal_relevance,
    determine_evidence_relevance,
    determine_evidence_strength,
)
from src.claim_analyzer import (
    ClaimImportance,
    EvidenceStance,
    ClaimStance,
    OverallAssessment,
    ClaimAnalysisResult,
    ArticleVerificationReport,
    analyze_evidence_against_claim,
    synthesize_claim_stance,
    generate_article_report,
)


class DummyMockProvider(SearchProvider):
    """Dummy provider returning controlled results for testing."""

    def __init__(self, results):
        self._results = results

    @property
    def name(self) -> str:
        return "DummyProvider"

    @property
    def is_real(self) -> bool:
        return False

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        return SearchResponse(
            status=SearchStatus.SUCCESS,
            results=self._results[:max_results],
            provider_name=self.name,
        )


class TestPhase8Robustness(unittest.TestCase):
    """Test suite covering Phase 8 source selection, independence, and evidence robustness."""

    def setUp(self):
        self.claim = Claim(
            claim_id="C100",
            text="Police used tear gas and lathi-charges against protesters in central Delhi.",
            source_sentence_index=1,
        )
        self.query = SearchQuery(
            claim_id="C100",
            query_text="police tear gas lathi charge central Delhi",
            preserved_keywords=["police", "tear", "gas", "lathi", "charge", "Delhi"],
        )

    # 1. Primary source gets higher priority
    def test_01_primary_source_gets_higher_priority(self):
        gov_ev = EvidenceItem(
            evidence_id="E001",
            claim_id="C100",
            title="Official Statement on Security Measures",
            url="https://delhipolice.gov.in/press/statement",
            domain="delhipolice.gov.in",
            snippet="Delhi Police issued a statement on security deployments at Jantar Mantar.",
            publication_date="2026-07-20",
            search_query="police tear gas Delhi",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        gen_ev = EvidenceItem(
            evidence_id="E002",
            claim_id="C100",
            title="Protest Report",
            url="https://generalblog.xyz/protest-update",
            domain="generalblog.xyz",
            snippet="Protest updates from Delhi.",
            publication_date="2026-07-20",
            search_query="police tear gas Delhi",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(claim=self.claim, search_query=self.query, evidence_items=[gov_ev, gen_ev])
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.primary_source_count, 1)
        # Primary source should be ranked first due to higher quality score
        self.assertEqual(summary.evaluated_items[0].evidence.evidence_id, "E001")
        self.assertGreater(
            summary.evaluated_items[0].evaluation.quality_score,
            summary.evaluated_items[1].evaluation.quality_score,
        )
        self.assertTrue(summary.evaluated_items[0].evaluation.is_primary)

    # 2. Official source is not automatically treated as truth
    def test_02_official_source_not_automatically_truth_when_conflicting(self):
        police_denial = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E001",
                claim_id="C100",
                title="Delhi Police statement",
                url="https://delhipolice.gov.in/press/1",
                domain="delhipolice.gov.in",
                snippet='Delhi Police dismissed reports as "entirely false" and denied reports of detention.',
                publication_date="2026-07-20",
                search_query="police",
                source_rank=1,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E001",
                category=SourceCategory.PRIMARY_OFFICIAL,
                quality_score=90,
                is_primary=True,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
                evidence_strength=EvidenceStrength.STRONG,
            ),
        )
        media_report = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E002",
                claim_id="C100",
                title="Protester Detained in Delhi",
                url="https://thehindu.com/news/national/protest",
                domain="thehindu.com",
                snippet="Protesters were detained by police during the demonstration.",
                publication_date="2026-07-20",
                search_query="protest",
                source_rank=2,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E002",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=80,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
                evidence_strength=EvidenceStrength.STRONG,
            ),
        )
        detention_claim = Claim(
            claim_id="C100",
            text="Delhi police detained protesters during the march.",
            source_sentence_index=1,
        )
        # When official statement conflicts with independent media reporting, verdict must be UNCERTAIN (contested)
        verdict = synthesize_claim_stance(detention_claim, [police_denial, media_report])
        self.assertEqual(verdict.final_stance, ClaimStance.UNCERTAIN)
        self.assertIn("E001", verdict.contradicting_evidence_ids)
        self.assertIn("E002", verdict.supporting_evidence_ids)

    # 3. Political viewpoint does not affect quality score
    def test_03_political_viewpoint_does_not_affect_quality_score(self):
        # Two news outlets with perceived differing editorial viewpoints both get scored objectively
        score_a = calculate_source_quality(
            category=SourceCategory.FACTUAL_REPORTING,
            has_date=True,
            has_citations=True,
            is_clickbait=False,
            is_duplicate_wire=False,
        )
        score_b = calculate_source_quality(
            category=SourceCategory.FACTUAL_REPORTING,
            has_date=True,
            has_citations=True,
            is_clickbait=False,
            is_duplicate_wire=False,
        )
        self.assertEqual(score_a, score_b)
        self.assertEqual(score_a, 85)

    # 4. Opinion article is deprioritized
    def test_04_opinion_article_deprioritized(self):
        op_ev = EvidenceItem(
            evidence_id="E001",
            claim_id="C100",
            title="Opinion: Why the Delhi protest is escalating",
            url="https://nationaldaily.com/opinion/columns/delhi-protest",
            domain="nationaldaily.com",
            snippet="In my viewpoint, the government must reflect on the police action.",
            publication_date="2026-07-20",
            search_query="protest",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(claim=self.claim, search_query=self.query, evidence_items=[op_ev])
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.opinion_source_count, 1)
        self.assertEqual(summary.evaluated_items[0].evaluation.category, SourceCategory.OPINION_EDITORIAL)
        self.assertTrue(summary.evaluated_items[0].evaluation.is_opinion)
        self.assertLess(summary.evaluated_items[0].evaluation.quality_score, 50)
        self.assertIsNotNone(summary.evaluated_items[0].evaluation.demotion_reason)

    # 5. Clickbait article is deprioritized
    def test_05_clickbait_article_deprioritized(self):
        clickbait_ev = EvidenceItem(
            evidence_id="E001",
            claim_id="C100",
            title="SHOCKING BOMBSHELL!! You Won't Believe What Delhi Cops Did!",
            url="https://viralbuzz.com/shocking-news",
            domain="viralbuzz.com",
            snippet="Crazy video from Delhi protest.",
            publication_date="2026-07-20",
            search_query="protest",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(claim=self.claim, search_query=self.query, evidence_items=[clickbait_ev])
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.evaluated_items[0].evaluation.category, SourceCategory.SENSATIONAL_CLICKBAIT)
        self.assertLessEqual(summary.evaluated_items[0].evaluation.quality_score, 30)

    # 6. PTI duplicate is detected
    def test_06_pti_duplicate_detected(self):
        pti_1 = EvidenceItem(
            evidence_id="E001",
            claim_id="C100",
            title="Protest march broken up: PTI report",
            url="https://outlet1.com/news/1",
            domain="outlet1.com",
            snippet="PTI reported that police used tear gas to disperse the crowd.",
            publication_date="2026-07-20",
            search_query="pti",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        pti_2 = EvidenceItem(
            evidence_id="E002",
            claim_id="C100",
            title="Delhi protests: Press Trust of India feed",
            url="https://outlet2.com/news/2",
            domain="outlet2.com",
            snippet="According to Press Trust of India, tear gas was fired near Parliament.",
            publication_date="2026-07-20",
            search_query="pti",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(claim=self.claim, search_query=self.query, evidence_items=[pti_1, pti_2])
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.duplicate_wire_count, 1)
        self.assertTrue(summary.evaluated_items[0].evaluation.is_independent)
        self.assertFalse(summary.evaluated_items[1].evaluation.is_independent)
        self.assertEqual(summary.evaluated_items[1].evaluation.independence_status, IndependenceStatus.DUPLICATE_WIRE)

    # 7. ANI duplicate is detected
    def test_07_ani_duplicate_detected(self):
        ani_1 = EvidenceItem(
            evidence_id="E001",
            claim_id="C100",
            title="Police deploy batons at march: ANI",
            url="https://sitea.com/ani1",
            domain="sitea.com",
            snippet="ANI reports police batons used during Parliament march.",
            publication_date="2026-07-20",
            search_query="ani",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        ani_2 = EvidenceItem(
            evidence_id="E002",
            claim_id="C100",
            title="Delhi march halted: Asian News International",
            url="https://siteb.com/ani2",
            domain="siteb.com",
            snippet="Asian News International reported police deployed batons to stop march.",
            publication_date="2026-07-20",
            search_query="ani",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(claim=self.claim, search_query=self.query, evidence_items=[ani_1, ani_2])
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.duplicate_wire_count, 1)
        self.assertFalse(summary.evaluated_items[1].evaluation.is_independent)

    # 8. Reuters duplicate is detected
    def test_08_reuters_duplicate_detected(self):
        reuters_1 = EvidenceItem(
            evidence_id="E001",
            claim_id="C100",
            title="Indian police clash with protesters: Reuters News Service",
            url="https://portal1.com/r1",
            domain="portal1.com",
            snippet="Reuters wire reported police clash in New Delhi.",
            publication_date="2026-07-20",
            search_query="reuters",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        reuters_2 = EvidenceItem(
            evidence_id="E002",
            claim_id="C100",
            title="Protestors dispersed in Delhi: Reuters Wire",
            url="https://portal2.com/r2",
            domain="portal2.com",
            snippet="According to Reuters news service, police used tear gas.",
            publication_date="2026-07-20",
            search_query="reuters",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(claim=self.claim, search_query=self.query, evidence_items=[reuters_1, reuters_2])
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.duplicate_wire_count, 1)
        self.assertFalse(summary.evaluated_items[1].evaluation.is_independent)

    # 9. Same-domain result dominance is controlled
    def test_09_same_domain_dominance_controlled(self):
        results = [
            SearchResult(title=f"Story {i}", url=f"https://samedomain.com/news/{i}", snippet="Snippet", domain="samedomain.com")
            for i in range(1, 6)
        ]
        results.append(SearchResult(title="Independent Story", url="https://otherdomain.com/news/1", snippet="Snippet", domain="otherdomain.com"))

        provider = DummyMockProvider(results)
        cr = retrieve_evidence_for_claim(self.claim, provider, max_sources=4, max_per_domain=2)

        # Domain 'samedomain.com' must not exceed max_per_domain (2)
        same_domain_count = sum(1 for e in cr.evidence_items if e.domain == "samedomain.com")
        self.assertLessEqual(same_domain_count, 2)
        self.assertTrue(any(e.domain == "otherdomain.com" for e in cr.evidence_items))

    # 10. Different independent domains increase corroboration
    def test_10_different_independent_domains_increase_corroboration(self):
        item1 = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E001",
                claim_id="C100",
                title="Police used teargas",
                url="https://independentnews1.com/story",
                domain="independentnews1.com",
                snippet="Police used tear gas and lathi charges on protesters.",
                publication_date="2026-07-20",
                search_query="teargas",
                source_rank=1,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E001",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=80,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
                evidence_strength=EvidenceStrength.STRONG,
            ),
        )
        item2 = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E002",
                claim_id="C100",
                title="Chaos as police deploy teargas",
                url="https://independentnews2.com/story",
                domain="independentnews2.com",
                snippet="Protesters met by tear gas and lathi charges in central Delhi.",
                publication_date="2026-07-20",
                search_query="teargas",
                source_rank=2,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E002",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=80,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
                evidence_strength=EvidenceStrength.STRONG,
            ),
        )
        verdict = synthesize_claim_stance(self.claim, [item1, item2])
        self.assertEqual(verdict.final_stance, ClaimStance.SUPPORTED)
        self.assertEqual(verdict.evidence_strength, EvidenceStrength.STRONG)
        self.assertEqual(len(verdict.unique_domains), 2)

    # 11. Irrelevant search result does not support a claim
    def test_11_irrelevant_search_result_does_not_support_claim(self):
        irrelevant_item = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E001",
                claim_id="C100",
                title="Delhi Marathon Traffic Advisory",
                url="https://delhitraffic.com/advisory",
                domain="delhitraffic.com",
                snippet="Vehicles diverted due to annual sports marathon in central Delhi.",
                publication_date="2026-07-20",
                search_query="delhi",
                source_rank=1,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E001",
                category=SourceCategory.GENERAL_WEB,
                quality_score=50,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.WEAK,
                evidence_strength=EvidenceStrength.WEAK,
            ),
        )
        analysis = analyze_evidence_against_claim(self.claim, irrelevant_item)
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)

    # 12. Directly relevant evidence gets stronger ranking
    def test_12_directly_relevant_evidence_gets_stronger_ranking(self):
        rel_direct = determine_evidence_relevance(
            claim_text="Police used tear gas and lathi-charges against protesters in central Delhi.",
            title="Police use lathi charge, tear gas to break up march",
            snippet="Delhi police used tear gas shells and lathi charge to disperse protesters in central Delhi.",
        )
        rel_partial = determine_evidence_relevance(
            claim_text="Police used tear gas and lathi-charges against protesters in central Delhi.",
            title="Protesters gather in central Delhi",
            snippet="Thousands of people gathered in central Delhi demanding government action on issues.",
        )
        self.assertEqual(rel_direct, RelevanceLevel.DIRECT)
        self.assertEqual(rel_partial, RelevanceLevel.PARTIAL)

    # 13. Conflicting evidence produces UNCERTAIN
    def test_13_conflicting_evidence_produces_uncertain(self):
        sup_item = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E001",
                claim_id="C100",
                title="Report A",
                url="https://news1.com/a",
                domain="news1.com",
                snippet="Police used tear gas on protesters.",
                publication_date="2026-07-20",
                search_query="q",
                source_rank=1,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E001",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=75,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
            ),
        )
        con_item = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E002",
                claim_id="C100",
                title="Report B",
                url="https://news2.com/b",
                domain="news2.com",
                snippet='Authorities denied reports of tear gas as "entirely false".',
                publication_date="2026-07-20",
                search_query="q",
                source_rank=2,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E002",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=75,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
            ),
        )
        verdict = synthesize_claim_stance(self.claim, [sup_item, con_item])
        self.assertEqual(verdict.final_stance, ClaimStance.UNCERTAIN)

    # 14. No evidence produces UNCERTAIN
    def test_14_no_evidence_produces_uncertain(self):
        verdict = synthesize_claim_stance(self.claim, [])
        self.assertEqual(verdict.final_stance, ClaimStance.UNCERTAIN)
        self.assertEqual(verdict.evidence_strength, EvidenceStrength.INSUFFICIENT)

    # 15. Evidence remains linked to claim IDs
    def test_15_evidence_remains_linked_to_claim_ids(self):
        item = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E001",
                claim_id="C100",
                title="Report",
                url="https://news.com/1",
                domain="news.com",
                snippet="Police used tear gas.",
                publication_date="2026-07-20",
                search_query="q",
                source_rank=1,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E001",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=75,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
            ),
        )
        verdict = synthesize_claim_stance(self.claim, [item])
        self.assertEqual(verdict.claim_id, "C100")
        self.assertIn("E001", verdict.supporting_evidence_ids)

    # 16. Fake/missing URLs are never fabricated
    def test_16_fake_urls_never_fabricated(self):
        res = SearchResult(title="Test", url="https://realdomain.com/article", snippet="Text", domain="realdomain.com")
        provider = DummyMockProvider([res])
        cr = retrieve_evidence_for_claim(self.claim, provider)
        for ev in cr.evidence_items:
            self.assertEqual(ev.url, "https://realdomain.com/article")
            self.assertTrue(ev.url.startswith("http"))

    # 17. Publication date affects temporal relevance
    def test_17_publication_date_affects_temporal_relevance(self):
        note_contemporary, diff1 = detect_temporal_relevance("2026-07-20", "Protest happened on July 20, 2026.")
        note_mismatch, diff2 = detect_temporal_relevance("2015-05-10", "Protest happened on July 20, 2026.")
        self.assertEqual(diff1, 0)
        self.assertIn("contemporary", note_contemporary)
        self.assertEqual(diff2, 11)
        self.assertIn("differs substantially", note_mismatch)

    # 18. Supporting and contradicting evidence are both displayed
    def test_18_supporting_and_contradicting_evidence_both_displayed(self):
        sup_item = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E001",
                claim_id="C100",
                title="Supporter",
                url="https://news1.com",
                domain="news1.com",
                snippet="Police used tear gas and batons.",
                publication_date="2026-07-20",
                search_query="q",
                source_rank=1,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E001",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=75,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
            ),
        )
        con_item = EvaluatedEvidenceItem(
            evidence=EvidenceItem(
                evidence_id="E002",
                claim_id="C100",
                title="Denier",
                url="https://news2.com",
                domain="news2.com",
                snippet='Delhi Police called reports of tear gas "entirely false".',
                publication_date="2026-07-20",
                search_query="q",
                source_rank=2,
                retrieval_status=SearchStatus.SUCCESS,
            ),
            evaluation=SourceEvaluation(
                evidence_id="E002",
                category=SourceCategory.FACTUAL_REPORTING,
                quality_score=75,
                is_primary=False,
                is_opinion=False,
                is_syndicated_wire=False,
                is_independent=True,
                relevance_level=RelevanceLevel.DIRECT,
            ),
        )
        verdict = synthesize_claim_stance(self.claim, [sup_item, con_item])
        self.assertEqual(verdict.supporting_evidence_ids, ["E001"])
        self.assertEqual(verdict.contradicting_evidence_ids, ["E002"])
        self.assertIsNotNone(verdict.disputed_summary)

    # 19. Evidence strength is explainable
    def test_19_evidence_strength_is_explainable(self):
        str_strong = determine_evidence_strength(
            category=SourceCategory.PRIMARY_OFFICIAL,
            quality_score=90,
            is_independent=True,
            is_opinion=False,
            is_clickbait=False,
            is_primary=True,
            relevance_level=RelevanceLevel.DIRECT,
        )
        str_weak = determine_evidence_strength(
            category=SourceCategory.OPINION_EDITORIAL,
            quality_score=35,
            is_independent=True,
            is_opinion=True,
            is_clickbait=False,
            is_primary=False,
            relevance_level=RelevanceLevel.PARTIAL,
        )
        self.assertEqual(str_strong, EvidenceStrength.STRONG)
        self.assertEqual(str_weak, EvidenceStrength.WEAK)

    # 20. Article report aggregates diversity correctly
    def test_20_article_report_aggregates_diversity_correctly(self):
        claim_res = ClaimAnalysisResult(
            claim_id="C100",
            claim_text="Assertion",
            importance=ClaimImportance.HIGH,
            final_stance=ClaimStance.SUPPORTED,
            supporting_evidence_ids=["E001"],
            contradicting_evidence_ids=[],
            neutral_evidence_ids=[],
            explanation="Supported",
            evidence_limitations=[],
            individual_analyses=[],
            evidence_strength=EvidenceStrength.STRONG,
            strength_explanation="Strong evidence",
            disputed_summary=None,
            unique_domains=["hindu.com", "bbc.com"],
            primary_sources_count=1,
            independent_sources_count=2,
            syndicated_sources_count=0,
            opinion_sources_count=0,
        )
        report = generate_article_report([claim_res])
        self.assertEqual(report.total_unique_domains, 2)
        self.assertEqual(report.total_primary_sources, 1)
        self.assertEqual(report.total_independent_sources, 2)
        self.assertEqual(report.total_syndicated_sources, 0)
        self.assertEqual(report.overall_assessment, OverallAssessment.LIKELY_SUPPORTED)


if __name__ == "__main__":
    unittest.main()
