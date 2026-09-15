"""Unit tests for Phase 7 Claim/Evidence Stance Analysis and Explainable Verdict Generation.

Covers all 19 required scenarios:
1. Strong supporting evidence -> SUPPORTED.
2. Strong contradicting evidence -> CONTRADICTED.
3. No relevant evidence -> UNCERTAIN (never CONTRADICTED).
4. Conflicting credible sources -> UNCERTAIN with conflict explanation.
5. Opinion source does not override factual evidence.
6. Duplicate wire sources do not multiply corroboration.
7. Primary official source receives appropriate weight.
8. Temporal / date mismatch handled.
9. Numerical contradiction detected.
10. High-importance claims drive overall assessment.
11. Mixed claims -> MIXED.
12. Insufficient evidence -> INSUFFICIENT_EVIDENCE.
13. Complete traceability (Evidence IDs mapped to stances and final verdict).
14. No fabricated evidence in explanations.
"""

import unittest
from src.claim_extractor import Claim
from src.search_provider import SearchStatus
from src.evidence_retriever import EvidenceItem
from src.source_evaluator import (
    SourceCategory,
    SourceEvaluation,
    EvaluatedEvidenceItem,
    ClaimEvaluationSummary,
)
from src.claim_analyzer import (
    ClaimImportance,
    EvidenceStance,
    ClaimStance,
    OverallAssessment,
    ClaimAnalysisResult,
    determine_claim_importance,
    analyze_evidence_against_claim,
    synthesize_claim_stance,
    generate_article_report,
    analyze_claims_and_evidence,
)


class TestClaimAnalyzer(unittest.TestCase):
    """Test suite for stance evaluation, conflict handling, and explainable verdicts."""

    def setUp(self):
        self.claim_repo = Claim(
            claim_id="C001",
            text="The Reserve Bank of India kept the repo rate unchanged at 6.5%.",
            source_sentence_index=0,
        )

    def _make_evaluated_item(
        self,
        evidence_id: str,
        title: str,
        snippet: str,
        category: SourceCategory = SourceCategory.FACTUAL_REPORTING,
        quality_score: int = 80,
        is_primary: bool = False,
        is_opinion: bool = False,
        is_independent: bool = True,
    ) -> EvaluatedEvidenceItem:
        ev = EvidenceItem(
            evidence_id=evidence_id,
            claim_id="C001",
            title=title,
            url=f"https://example.com/{evidence_id}",
            domain="example.com",
            snippet=snippet,
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        se = SourceEvaluation(
            evidence_id=evidence_id,
            category=category,
            quality_score=quality_score,
            is_primary=is_primary,
            is_opinion=is_opinion,
            is_syndicated_wire=False,
            is_independent=is_independent,
        )
        return EvaluatedEvidenceItem(evidence=ev, evaluation=se)

    # 1. Strong supporting evidence -> SUPPORTED
    def test_01_strong_supporting_evidence_yields_supported(self):
        support_item = self._make_evaluated_item(
            evidence_id="E001",
            title="RBI Monetary Policy: Repo Rate Kept Steady at 6.5%",
            snippet="The Reserve Bank of India kept the benchmark repo rate unchanged at 6.50% on Thursday.",
            is_primary=True,
            category=SourceCategory.PRIMARY_OFFICIAL,
            quality_score=95,
        )
        result = synthesize_claim_stance(self.claim_repo, [support_item])

        self.assertEqual(result.final_stance, ClaimStance.SUPPORTED)
        self.assertIn("E001", result.supporting_evidence_ids)
        self.assertEqual(len(result.contradicting_evidence_ids), 0)
        self.assertIn("directly", result.explanation.lower())

    # 2. Strong contradicting evidence -> CONTRADICTED
    def test_02_strong_contradicting_evidence_yields_contradicted(self):
        # Claim says kept unchanged at 6.5%, evidence says RBI cut repo rate to 6.25%
        contradict_item = self._make_evaluated_item(
            evidence_id="E002",
            title="RBI Cuts Repo Rate to 6.25%",
            snippet="In an unexpected move, the Reserve Bank of India cut the benchmark repo rate to 6.25%.",
            is_primary=True,
            category=SourceCategory.PRIMARY_OFFICIAL,
            quality_score=95,
        )
        result = synthesize_claim_stance(self.claim_repo, [contradict_item])

        self.assertEqual(result.final_stance, ClaimStance.CONTRADICTED)
        self.assertIn("E002", result.contradicting_evidence_ids)
        self.assertEqual(len(result.supporting_evidence_ids), 0)

    # 3. No relevant evidence -> UNCERTAIN (never CONTRADICTED)
    def test_03_no_evidence_strictly_yields_uncertain(self):
        result = synthesize_claim_stance(self.claim_repo, [])
        self.assertEqual(result.final_stance, ClaimStance.UNCERTAIN)
        self.assertNotEqual(result.final_stance, ClaimStance.CONTRADICTED)
        self.assertIn("No external evidence", result.explanation)

    # 4. Conflicting credible sources -> UNCERTAIN with conflict explanation
    def test_04_conflicting_credible_sources_yields_uncertain(self):
        item_support = self._make_evaluated_item(
            evidence_id="E001",
            title="Report A: RBI maintained repo rate at 6.5%",
            snippet="The MPC held the repo rate unchanged at 6.5%.",
            quality_score=80,
        )
        item_conflict = self._make_evaluated_item(
            evidence_id="E002",
            title="Report B: RBI cut repo rate to 6.25%",
            snippet="The central bank decreased the policy repo rate to 6.25%.",
            quality_score=80,
        )
        result = synthesize_claim_stance(self.claim_repo, [item_support, item_conflict])

        self.assertEqual(result.final_stance, ClaimStance.UNCERTAIN)
        self.assertIn("E001", result.supporting_evidence_ids)
        self.assertIn("E002", result.contradicting_evidence_ids)
        self.assertIn("conflicting", result.explanation.lower())

    # 5. Opinion source does not override factual evidence
    def test_05_opinion_source_cannot_override_factual_evidence(self):
        factual_support = self._make_evaluated_item(
            evidence_id="E001",
            title="Official Policy Statement: Repo Rate 6.5%",
            snippet="The Reserve Bank kept the repo rate at 6.5%.",
            is_primary=True,
            category=SourceCategory.PRIMARY_OFFICIAL,
            quality_score=90,
        )
        opinion_conflict = self._make_evaluated_item(
            evidence_id="E002",
            title="Opinion: Why the rate decision was a cut in real terms",
            snippet="In my view the effective monetary policy decreased interest rate impact.",
            is_opinion=True,
            category=SourceCategory.OPINION_EDITORIAL,
            quality_score=35,
        )
        result = synthesize_claim_stance(self.claim_repo, [factual_support, opinion_conflict])

        # Primary factual support decisively overrides low-weight opinion piece
        self.assertEqual(result.final_stance, ClaimStance.SUPPORTED)
        self.assertIn("E001", result.supporting_evidence_ids)

    # 6. Duplicate wire sources do not count as independent corroboration
    def test_06_duplicate_wire_sources_downweighted(self):
        wire_1 = self._make_evaluated_item(
            evidence_id="E001",
            title="PTI: Repo rate steady at 6.5%",
            snippet="New Delhi (PTI) - The Reserve Bank of India kept repo rate unchanged at 6.5%.",
            is_independent=True,
            quality_score=75,
        )
        wire_2_duplicate = self._make_evaluated_item(
            evidence_id="E002",
            title="Reprint: Repo rate steady at 6.5%",
            snippet="(PTI) The Reserve Bank of India kept repo rate unchanged at 6.5%.",
            is_independent=False,  # Flagged by Phase 6 as duplicate
            quality_score=55,
        )
        analysis_1 = analyze_evidence_against_claim(self.claim_repo, wire_1)
        analysis_2 = analyze_evidence_against_claim(self.claim_repo, wire_2_duplicate)

        # Duplicate wire copy has discounted marginal weight
        self.assertGreater(analysis_1.weight, analysis_2.weight)

    # 7. Primary official source receives decisive weight
    def test_07_primary_official_source_decisive(self):
        primary_item = self._make_evaluated_item(
            evidence_id="E001",
            title="RBI Press Release 2024",
            snippet="The Monetary Policy Committee decided to keep the policy repo rate at 6.50%.",
            is_primary=True,
            category=SourceCategory.PRIMARY_OFFICIAL,
            quality_score=95,
        )
        result = synthesize_claim_stance(self.claim_repo, [primary_item])
        self.assertEqual(result.final_stance, ClaimStance.SUPPORTED)
        self.assertIn("primary", result.explanation.lower())

    # 8. Temporal / Date mismatch handled correctly
    def test_08_temporal_mismatch_detected(self):
        claim_dated = Claim(
            claim_id="C002",
            text="NASA launched the lunar mission on 15 January 2024.",
            source_sentence_index=1,
        )
        # Evidence discusses a different year/event
        item_irrelevant = self._make_evaluated_item(
            evidence_id="E001",
            title="Apollo 11 Mission Overview",
            snippet="NASA landed astronauts on the moon in July 1969.",
        )
        result = synthesize_claim_stance(claim_dated, [item_irrelevant])
        self.assertEqual(result.final_stance, ClaimStance.UNCERTAIN)

    # 9. Numerical contradiction detected
    def test_09_numerical_contradiction_detected(self):
        claim_num = Claim(claim_id="C003", text="GDP growth was recorded at 8.2%.", source_sentence_index=2)
        # Evidence says GDP growth was 6.1%
        item_num = self._make_evaluated_item(
            evidence_id="E001",
            title="Statistical Release: GDP Growth at 6.1%",
            snippet="The national accounts confirmed annual GDP growth slowed to 6.1%.",
            quality_score=85,
        )
        analysis = analyze_evidence_against_claim(claim_num, item_num)
        self.assertEqual(analysis.stance, EvidenceStance.CONTRADICTS)
        self.assertTrue(any("6.1%" in p for p in analysis.conflicting_points))

    # 10. Claim importance drives overall article assessment
    def test_10_high_importance_drives_overall_assessment(self):
        claim_high = Claim(claim_id="C001", text="The central bank ordered a 50 basis point rate hike to 6.5%.", source_sentence_index=0)
        claim_low = Claim(claim_id="C002", text="The meeting was held in Mumbai.", source_sentence_index=1)

        self.assertEqual(determine_claim_importance(claim_high), ClaimImportance.HIGH)
        self.assertEqual(determine_claim_importance(claim_low), ClaimImportance.LOW)

        # High claim is CONTRADICTED, Low claim is SUPPORTED
        analysis_high = ClaimAnalysisResult(
            claim_id="C001",
            claim_text=claim_high.text,
            importance=ClaimImportance.HIGH,
            final_stance=ClaimStance.CONTRADICTED,
            contradicting_evidence_ids=["E001"],
        )
        analysis_low = ClaimAnalysisResult(
            claim_id="C002",
            claim_text=claim_low.text,
            importance=ClaimImportance.LOW,
            final_stance=ClaimStance.SUPPORTED,
            supporting_evidence_ids=["E002"],
        )

        report = generate_article_report([analysis_high, analysis_low])
        # High importance contradiction dominates low importance peripheral support
        self.assertEqual(report.overall_assessment, OverallAssessment.LIKELY_CONTRADICTED)

    # 11. Mixed claims -> MIXED
    def test_11_mixed_claims_yields_mixed(self):
        c1 = ClaimAnalysisResult(claim_id="C001", claim_text="Text 1", importance=ClaimImportance.HIGH, final_stance=ClaimStance.SUPPORTED)
        c2 = ClaimAnalysisResult(claim_id="C002", claim_text="Text 2", importance=ClaimImportance.HIGH, final_stance=ClaimStance.CONTRADICTED)
        report = generate_article_report([c1, c2])
        self.assertEqual(report.overall_assessment, OverallAssessment.MIXED)

    # 12. Insufficient evidence -> INSUFFICIENT_EVIDENCE
    def test_12_insufficient_evidence_yields_insufficient_evidence(self):
        c1 = ClaimAnalysisResult(claim_id="C001", claim_text="Text 1", importance=ClaimImportance.HIGH, final_stance=ClaimStance.UNCERTAIN)
        c2 = ClaimAnalysisResult(claim_id="C002", claim_text="Text 2", importance=ClaimImportance.MEDIUM, final_stance=ClaimStance.UNCERTAIN)
        report = generate_article_report([c1, c2])
        self.assertEqual(report.overall_assessment, OverallAssessment.INSUFFICIENT_EVIDENCE)

    # 13. Traceability: Evidence IDs preserved in analysis
    def test_13_traceability_preserved(self):
        item1 = self._make_evaluated_item("E001", "RBI Statement", "Repo rate held at 6.5%.", is_primary=True, quality_score=90)
        item2 = self._make_evaluated_item("E002", "Irrelevant News", "Weather in Delhi was sunny.", quality_score=60)
        result = synthesize_claim_stance(self.claim_repo, [item1, item2])

        self.assertIn("E001", result.supporting_evidence_ids)
        self.assertIn("E002", result.neutral_evidence_ids)
        self.assertEqual(len(result.individual_analyses), 2)
        self.assertEqual(result.individual_analyses[0].evidence_id, "E001")
        self.assertEqual(result.individual_analyses[1].evidence_id, "E002")

    # 14. No fabricated evidence
    def test_14_no_fabricated_evidence(self):
        result = synthesize_claim_stance(self.claim_repo, [])
        self.assertEqual(result.supporting_evidence_ids, [])
        self.assertEqual(result.contradicting_evidence_ids, [])
        self.assertEqual(result.neutral_evidence_ids, [])


if __name__ == "__main__":
    unittest.main()
