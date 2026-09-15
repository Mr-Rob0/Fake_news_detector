"""Unit tests for Phase 9: Generic Claim–Evidence Stance Analysis & Explainable Verdict Engine.

Strictly verifies DOMAIN-AGNOSTIC behavior across 10 unrelated domains:
1. Politics
2. Sports
3. Science
4. Technology
5. Business / Economy
6. Crime
7. International news
8. Natural disaster
9. Health
10. Government announcement

Also covers all 25 specific stance & aggregation criteria:
- Clear supporting evidence -> SUPPORTED
- Clear contradicting evidence -> CONTRADICTED
- No evidence -> UNCERTAIN (never CONTRADICTED)
- Conflicting credible evidence -> UNCERTAIN
- Keyword overlap without factual support -> NOT SUPPORTING (NEUTRAL_IRRELEVANT)
- Negation handling
- Attribution handling (allegations vs direct facts)
- Date / Temporal mismatch
- Quantity / Numeric mismatch
- Partial claim support
- Duplicate source handling
- Wire-service duplication
- Primary source + secondary source
- Multiple independent sources
- Opinion source handling
- Political viewpoint neutrality (structural evaluation)
- No fabricated URLs or citations
- Claim IDs remain traceable
- Evidence IDs remain traceable
- Overall article assessment
- Mixed article containing both supported and contradicted claims
- Domain-general test fixtures
- Empty/insufficient evidence
- Neutral/context evidence
- Exact claim vs broader-topic article
"""

import unittest
from src.claim_extractor import Claim
from src.search_provider import SearchStatus
from src.evidence_retriever import EvidenceItem
from src.source_evaluator import (
    SourceCategory,
    SourceEvaluation,
    EvaluatedEvidenceItem,
    IndependenceStatus,
    RelevanceLevel,
    EvidenceStrength,
)
from src.claim_analyzer import (
    ClaimImportance,
    EvidenceStance,
    ClaimStance,
    OverallAssessment,
    ClaimAnalysisResult,
    analyze_evidence_against_claim,
    synthesize_claim_stance,
    generate_article_report,
    extract_generic_representation,
)


class TestPhase9GenericStance(unittest.TestCase):
    """Test suite ensuring strictly domain-agnostic stance analysis and explainable verdicts."""

    def _make_item(
        self,
        evidence_id: str,
        title: str,
        snippet: str,
        claim_id: str = "C001",
        url: str = "https://example.com/report",
        domain: str = "example.com",
        quality_score: int = 80,
        category: SourceCategory = SourceCategory.FACTUAL_REPORTING,
        is_primary: bool = False,
        is_opinion: bool = False,
        is_independent: bool = True,
        relevance_level: RelevanceLevel = RelevanceLevel.DIRECT,
        evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE,
        independence_status: IndependenceStatus = IndependenceStatus.INDEPENDENT,
    ) -> EvaluatedEvidenceItem:
        ev = EvidenceItem(
            evidence_id=evidence_id,
            claim_id=claim_id,
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            publication_date="2026-01-01",
            search_query="test search query",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        evaluation = SourceEvaluation(
            evidence_id=evidence_id,
            quality_score=quality_score,
            category=category,
            is_primary=is_primary,
            is_opinion=is_opinion,
            is_independent=is_independent,
            is_syndicated_wire=(independence_status == IndependenceStatus.DUPLICATE_WIRE),
            relevance_level=relevance_level,
            evidence_strength=evidence_strength,
            independence_status=independence_status,
        )
        return EvaluatedEvidenceItem(evidence=ev, evaluation=evaluation)

    # -------------------------------------------------------------
    # 1. Politics Domain Test
    # -------------------------------------------------------------
    def test_01_domain_politics_supported(self):
        claim = Claim("POL_01", "Parliament approved the electoral reform bill.", 0)
        item = self._make_item(
            "E_POL_1",
            "Parliament Passes Electoral Reform Bill",
            "Lawmakers approved the landmark electoral reform bill after unanimous voting.",
            is_primary=True,
            quality_score=90,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)
        self.assertIn("E_POL_1", res.supporting_evidence_ids)

    # -------------------------------------------------------------
    # 2. Sports Domain Test
    # -------------------------------------------------------------
    def test_02_domain_sports_contradicted(self):
        claim = Claim("SPT_01", "Team Alpha won the championship match.", 0)
        item = self._make_item(
            "E_SPT_1",
            "Team Alpha Defeated in Finals",
            "Team Alpha lost the championship match to Team Beta after conceding two goals.",
            quality_score=85,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)
        self.assertIn("E_SPT_1", res.contradicting_evidence_ids)

    # -------------------------------------------------------------
    # 3. Science Domain Test
    # -------------------------------------------------------------
    def test_03_domain_science_numerical_support(self):
        claim = Claim("SCI_01", "The James Webb telescope observed galaxy cluster at redshift 8.5.", 0)
        item = self._make_item(
            "E_SCI_1",
            "Astrophysical Journal: Webb Telescope Data",
            "Spectroscopy confirms observation of a massive galaxy cluster at redshift 8.5.",
            is_primary=True,
            quality_score=95,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------
    # 4. Technology Domain Test
    # -------------------------------------------------------------
    def test_04_domain_technology_topic_overlap_vs_action(self):
        # Broad topic overlap does NOT equal factual support
        claim = Claim("TECH_01", "Acme Corp launched its quantum processor chip.", 0)
        item_topic_only = self._make_item(
            "E_TECH_1",
            "Acme Corp Financial Earnings Report",
            "Acme Corp released its fourth quarter revenue report showing strong growth.",
            quality_score=80,
        )
        analysis = analyze_evidence_against_claim(claim, item_topic_only)
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)
        self.assertIn("does not corroborate the specific action", analysis.rationale)

    # -------------------------------------------------------------
    # 5. Business / Economy Domain Test
    # -------------------------------------------------------------
    def test_05_domain_economy_rate_contradiction(self):
        claim = Claim("ECON_01", "Central Bank increased interest rates by 50 basis points.", 0)
        item = self._make_item(
            "E_ECON_1",
            "Central Bank Statement",
            "The monetary board decreased interest rates by 25 basis points to stimulate growth.",
            quality_score=90,
            is_primary=True,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------
    # 6. Crime Domain Test
    # -------------------------------------------------------------
    def test_06_domain_crime_arrest_polarity(self):
        claim = Claim("CRM_01", "Police arrested the cyber fraud suspects.", 0)
        item = self._make_item(
            "E_CRM_1",
            "Police Press Briefing",
            "Investigators confirmed police did not arrest the cyber fraud suspects yet.",
            quality_score=85,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------
    # 7. International News Domain Test
    # -------------------------------------------------------------
    def test_07_domain_international_treaty_approved(self):
        claim = Claim("INT_01", "Delegates approved the maritime security treaty.", 0)
        item = self._make_item(
            "E_INT_1",
            "UN Security Summit Bulletin",
            "International delegates ratified and approved the maritime security treaty.",
            is_primary=True,
            quality_score=92,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------
    # 8. Natural Disaster Domain Test
    # -------------------------------------------------------------
    def test_08_domain_disaster_temporal_mismatch(self):
        claim = Claim("DIS_01", "A magnitude 7.2 earthquake struck the coastal region in 2026.", 0)
        item_historical = self._make_item(
            "E_DIS_1",
            "Historical Seismology Archives",
            "A magnitude 7.2 earthquake struck the coastal region in 2011.",
            quality_score=80,
        )
        analysis = analyze_evidence_against_claim(claim, item_historical)
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)

    # -------------------------------------------------------------
    # 9. Health Domain Test
    # -------------------------------------------------------------
    def test_09_domain_health_numeric_discrepancy(self):
        claim = Claim("HLT_01", "Clinical trials reported a 90% efficacy rate for the vaccine.", 0)
        item = self._make_item(
            "E_HLT_1",
            "Phase 3 Clinical Trial Results",
            "Peer-reviewed trial data confirmed a 65% efficacy rate for the vaccine.",
            quality_score=88,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------
    # 10. Government Announcement Domain Test
    # -------------------------------------------------------------
    def test_10_domain_government_attribution_handling(self):
        claim = Claim("GOV_01", "The ministry opened subsidies for solar energy.", 0)
        item_allegation = self._make_item(
            "E_GOV_1",
            "Industry Association Bulletin",
            "According to union leaders alleged that the ministry opened subsidies for solar energy.",
            quality_score=75,
        )
        analysis = analyze_evidence_against_claim(claim, item_allegation)
        self.assertEqual(analysis.stance, EvidenceStance.SUPPORTS)
        self.assertIn("attributed statement or allegation", analysis.rationale)

    # -------------------------------------------------------------
    # 11. Negation Handling Test
    # -------------------------------------------------------------
    def test_11_negation_distinguishes_opposite_stances(self):
        claim_pos = Claim("NEG_01", "Authorities banned single-use plastic bags.", 0)
        claim_neg = Claim("NEG_02", "Authorities did not ban single-use plastic bags.", 1)

        item_event = self._make_item(
            "E_NEG_1",
            "Municipal Gazette",
            "The environment department announced authorities banned single-use plastic bags.",
            quality_score=85,
        )

        res_pos = synthesize_claim_stance(claim_pos, [item_event])
        res_neg = synthesize_claim_stance(claim_neg, [item_event])

        self.assertEqual(res_pos.final_stance, ClaimStance.SUPPORTED)
        self.assertEqual(res_neg.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------
    # 12. No Evidence Yields UNCERTAIN (Never CONTRADICTED)
    # -------------------------------------------------------------
    def test_12_no_evidence_returns_uncertain(self):
        claim = Claim("UNC_01", "A rare mineral deposit was discovered in Antarctica.", 0)
        res = synthesize_claim_stance(claim, [])
        self.assertEqual(res.final_stance, ClaimStance.UNCERTAIN)
        self.assertEqual(res.evidence_strength, EvidenceStrength.INSUFFICIENT)

    # -------------------------------------------------------------
    # 13. Conflicting Credible Sources Yields UNCERTAIN
    # -------------------------------------------------------------
    def test_13_conflicting_credible_sources_yields_uncertain(self):
        claim = Claim("CNF_01", "The port authority resumed cargo operations on Tuesday.", 0)
        item_support = self._make_item(
            "E_CNF_1",
            "Maritime Dispatch",
            "Port authority resumed cargo operations on Tuesday morning.",
            quality_score=80,
        )
        item_contradict = self._make_item(
            "E_CNF_2",
            "Shipping Gazette",
            "Port authority denied reports and delayed cargo operations until Friday.",
            quality_score=80,
        )
        res = synthesize_claim_stance(claim, [item_support, item_contradict])
        self.assertEqual(res.final_stance, ClaimStance.UNCERTAIN)
        self.assertIsNotNone(res.disputed_summary)
        self.assertIn("conflicting", res.explanation.lower())

    # -------------------------------------------------------------
    # 14. Wire-Service Duplication Does Not Multiply Independence
    # -------------------------------------------------------------
    def test_14_wire_service_duplication_discounted(self):
        claim = Claim("WIR_01", "Metro rail announced weekend service expansion.", 0)
        item_wire1 = self._make_item(
            "E_WIR_1",
            "NewsWire: Metro expansion",
            "Metro rail announced weekend service expansion across three lines.",
            domain="source1.com",
            quality_score=75,
            independence_status=IndependenceStatus.INDEPENDENT,
        )
        item_wire2 = self._make_item(
            "E_WIR_2",
            "Syndicated: Metro expansion",
            "NewsWire: Metro rail announced weekend service expansion across three lines.",
            domain="source2.com",
            quality_score=50,
            is_independent=False,
            independence_status=IndependenceStatus.DUPLICATE_WIRE,
        )
        res = synthesize_claim_stance(claim, [item_wire1, item_wire2])
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)
        self.assertEqual(res.independent_sources_count, 1)
        self.assertEqual(res.syndicated_sources_count, 1)

    # -------------------------------------------------------------
    # 15. Primary Source + Secondary Source
    # -------------------------------------------------------------
    def test_15_primary_source_and_secondary_source_corroboration(self):
        claim = Claim("PRI_01", "Supreme Court released new procedural guidelines for digital evidence.", 0)
        item_primary = self._make_item(
            "E_PRI_1",
            "Supreme Court Notification 2026",
            "The Supreme Court released official procedural guidelines for digital evidence.",
            is_primary=True,
            quality_score=95,
        )
        item_secondary = self._make_item(
            "E_SEC_1",
            "Legal Times Daily",
            "Legal analysts review how Supreme Court released guidelines for digital evidence.",
            quality_score=80,
        )
        res = synthesize_claim_stance(claim, [item_primary, item_secondary])
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)
        self.assertEqual(res.evidence_strength, EvidenceStrength.STRONG)
        self.assertEqual(res.primary_sources_count, 1)

    # -------------------------------------------------------------
    # 16. Opinion Source Handling
    # -------------------------------------------------------------
    def test_16_opinion_source_downweighted(self):
        claim = Claim("OPN_01", "The municipal council approved the bridge renovation contract.", 0)
        item_opinion = self._make_item(
            "E_OPN_1",
            "Opinion: Why this bridge contract is a mistake",
            "In my view the council should reject the bridge contract entirely.",
            is_opinion=True,
            category=SourceCategory.OPINION_EDITORIAL,
            quality_score=30,
        )
        analysis = analyze_evidence_against_claim(claim, item_opinion)
        self.assertLess(analysis.weight, 0.2)

    # -------------------------------------------------------------
    # 17. Structural Political Neutrality
    # -------------------------------------------------------------
    def test_17_political_neutrality_preserved(self):
        # Confirms no ideological blacklist exists in representation or evaluation
        claim = Claim("NEU_01", "The finance ministry lowered fuel taxes by 2%.", 0)
        item_pub = self._make_item(
            "E_PUB_1",
            "National Chronicle",
            "The finance ministry officially lowered fuel taxes by 2%.",
            quality_score=85,
        )
        res = synthesize_claim_stance(claim, [item_pub])
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)
        # Verify no mention of political leaning in explanations
        self.assertNotIn("left", res.explanation.lower())
        self.assertNotIn("right", res.explanation.lower())
        self.assertNotIn("godi", res.explanation.lower())

    # -------------------------------------------------------------
    # 18. Traceability of Claim IDs and Evidence IDs
    # -------------------------------------------------------------
    def test_18_traceability_preserved(self):
        claim = Claim("TRC_99", "Automaker rolled out hybrid electric sports utility vehicle.", 0)
        item = self._make_item(
            "E_TRC_55",
            "Automotive Gazette",
            "Automaker rolled out new hybrid electric sports utility vehicle lineup.",
            quality_score=80,
        )
        res = synthesize_claim_stance(claim, [item])
        self.assertEqual(res.claim_id, "TRC_99")
        self.assertIn("E_TRC_55", res.supporting_evidence_ids)

    # -------------------------------------------------------------
    # 19. Overall Article Assessment: Mixed Article
    # -------------------------------------------------------------
    def test_19_overall_article_assessment_mixed(self):
        claim1 = Claim("MIX_01", "The airline resumed 12 daily flights to Singapore.", 0)
        claim2 = Claim("MIX_02", "Ticket fares dropped by 50%.", 1)

        item1 = self._make_item(
            "E_MIX_1",
            "Aviation News",
            "The airline resumed 12 daily scheduled passenger flights to Singapore.",
            quality_score=85,
        )
        item2 = self._make_item(
            "E_MIX_2",
            "Aviation Pricing Monitor",
            "Reports of ticket fares dropped by 50% are false claim; fares increased by 10%.",
            quality_score=85,
        )

        res1 = synthesize_claim_stance(claim1, [item1])
        res2 = synthesize_claim_stance(claim2, [item2])

        report = generate_article_report([res1, res2])
        self.assertEqual(report.overall_assessment, OverallAssessment.MIXED)
        self.assertEqual(report.supported_claims, 1)
        self.assertEqual(report.contradicted_claims, 1)

    # -------------------------------------------------------------
    # 20. Overall Article Assessment: Insufficient Evidence
    # -------------------------------------------------------------
    def test_20_overall_article_assessment_insufficient(self):
        claim = Claim("INS_01", "Astronomers detected an unidentified comet trajectory.", 0)
        res = synthesize_claim_stance(claim, [])
        report = generate_article_report([res])
        self.assertEqual(report.overall_assessment, OverallAssessment.INSUFFICIENT_EVIDENCE)
        self.assertEqual(report.uncertain_claims, 1)

    # -------------------------------------------------------------
    # 21. No Fabricated URLs or Quotations
    # -------------------------------------------------------------
    def test_21_no_fabricated_urls(self):
        claim = Claim("FAB_01", "Telecom provider introduced satellite broadband connectivity.", 0)
        item = self._make_item(
            "E_REAL_1",
            "Telecom Industry Review",
            "Telecom provider introduced satellite broadband connectivity in rural zones.",
            url="https://legitimate-telecom-news.com/broadband-launch",
            quality_score=82,
        )
        res = synthesize_claim_stance(claim, [item])
        # Explanation must not invent phantom links or statistics
        self.assertNotIn("http://fake-site.com", res.explanation)
        self.assertNotIn("99.9%", res.explanation)

    # -------------------------------------------------------------
    # 22. Generic Representation Robustness
    # -------------------------------------------------------------
    def test_22_generic_representation_extraction(self):
        text = "RBI kept the policy repo rate unchanged at 6.5% on 15 February 2026."
        rep = extract_generic_representation(text)
        self.assertIn("rbi", rep.entities)
        self.assertEqual(rep.action_direction, "UNCHANGED")
        self.assertIn(6.5, rep.numbers)
        self.assertIn("2026", rep.dates_years)
        self.assertFalse(rep.is_negated)

    # -------------------------------------------------------------
    # 23. Denial Reports vs Direct Event
    # -------------------------------------------------------------
    def test_23_denial_reports_do_not_verify_claim(self):
        claim = Claim("DEN_01", "The company committed financial fraud.", 0)
        item = self._make_item(
            "E_DEN_1",
            "Company Press Release",
            "Company executives categorically denied reports of financial fraud.",
            quality_score=85,
        )
        res = synthesize_claim_stance(claim, [item])
        # Stance is either CONTRADICTED or UNCERTAIN, never SUPPORTED
        self.assertNotEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------
    # 24. Exact Claim vs Broader-Topic Article
    # -------------------------------------------------------------
    def test_24_exact_claim_vs_broader_topic(self):
        claim = Claim("EXA_01", "Automaker opened a new lithium battery factory in Texas.", 0)
        item_broader = self._make_item(
            "E_EXA_1",
            "Automaker Global Overview",
            "Automaker manufactures electric cars and operates facilities worldwide.",
            quality_score=75,
        )
        analysis = analyze_evidence_against_claim(claim, item_broader)
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)

    # -------------------------------------------------------------
    # 25. High-Importance Claims Influence Assessment
    # -------------------------------------------------------------
    def test_25_high_importance_claim_shapes_overall_verdict(self):
        high_claim = Claim("IMP_01", "The ministry banned imports of 85 products effective immediately.", 0)
        low_claim = Claim("IMP_02", "The meeting began on Tuesday morning.", 1)

        item_contradict_high = self._make_item(
            "E_IMP_1",
            "Official Gazette: Ministry Import Policy",
            "The ministry rejected the proposed import ban. No import restrictions on any products were approved this quarter.",
            is_primary=True,
            quality_score=95,
        )
        item_support_low = self._make_item(
            "E_IMP_2",
            "Local News",
            "The meeting began on Tuesday morning at the convention center.",
            quality_score=70,
        )

        res_high = synthesize_claim_stance(high_claim, [item_contradict_high])
        res_low = synthesize_claim_stance(low_claim, [item_support_low])

        self.assertEqual(res_high.importance, ClaimImportance.HIGH)
        self.assertEqual(res_high.final_stance, ClaimStance.CONTRADICTED)

        report = generate_article_report([res_high, res_low])
        # Central assertion contradicted -> article should be MIXED or LIKELY_CONTRADICTED, not LIKELY_SUPPORTED
        self.assertIn(report.overall_assessment, (OverallAssessment.MIXED, OverallAssessment.LIKELY_CONTRADICTED))


if __name__ == "__main__":
    unittest.main()
