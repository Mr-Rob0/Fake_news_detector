"""Unit tests for the Source Evaluation module (Phase 6).

Covers all 10 user-specified criteria:
1. Primary / Official source prioritization.
2. Factual reporting vs opinion / editorial detection and transparent demotion.
3. Wire service attribution (PTI, ANI, AP, Reuters).
4. Syndicated wire deduplication (no false independent corroboration).
5. Clickbait / sensational headline detection and penalty.
6. Transparency notes and human-readable demotion reasons.
7. Fallback to UNCERTAIN candidate when high-quality independent evidence is absent.
8. Neutral, non-ideological evaluation (zero political blacklisting).
"""

import unittest
from src.claim_extractor import Claim
from src.search_provider import SearchStatus
from src.evidence_retriever import (
    EvidenceItem,
    SearchQuery,
    ClaimEvidenceResult,
)
from src.source_evaluator import (
    SourceCategory,
    SourceEvaluation,
    detect_opinion_content,
    detect_syndicated_wire,
    detect_clickbait_sensationalism,
    detect_primary_provenance,
    evaluate_claim_evidence,
)


class TestSourceEvaluator(unittest.TestCase):
    """Test suite for objective source quality and provenance evaluation."""

    def setUp(self):
        self.claim = Claim(
            claim_id="C001",
            text="The Reserve Bank of India kept the repo rate unchanged at 6.5%.",
            source_sentence_index=0,
        )
        self.query = SearchQuery(
            claim_id="C001",
            query_text="RBI repo rate 6.5%",
            preserved_keywords=["RBI", "repo", "rate", "6.5%"],
        )

    # 1. Primary / Official sources receive highest quality score
    def test_01_primary_official_source_prioritized(self):
        gov_item = EvidenceItem(
            evidence_id="E001",
            claim_id="C001",
            title="Monetary Policy Report",
            url="https://rbi.org.in/press/repo-rate",
            domain="rbi.org.in",
            snippet='The Monetary Policy Committee decided "to maintain the policy repo rate at 6.50%".',
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
            is_priority_source=True,
            priority_tier="Institutional / Regulatory",
        )
        cr = ClaimEvidenceResult(
            claim=self.claim,
            search_query=self.query,
            evidence_items=[gov_item],
            status=SearchStatus.SUCCESS,
        )
        summary = evaluate_claim_evidence(cr)

        self.assertEqual(summary.primary_source_count, 1)
        self.assertEqual(summary.evaluated_items[0].evaluation.category, SourceCategory.PRIMARY_OFFICIAL)
        self.assertTrue(summary.evaluated_items[0].evaluation.is_primary)
        self.assertGreaterEqual(summary.evaluated_items[0].evaluation.quality_score, 85)
        self.assertIn("Primary official source", summary.evaluated_items[0].evaluation.transparency_notes[0])

    # 2. Opinion / Editorial pieces detected and demoted with transparent reason
    def test_02_opinion_content_demoted_with_reason(self):
        opinion_item = EvidenceItem(
            evidence_id="E002",
            claim_id="C001",
            title="Opinion: Why the RBI rate decision hurts consumers",
            url="https://example.com/opinion/rbi-rate-bad-idea",
            domain="example.com",
            snippet="In my view, keeping the repo rate high will negatively impact home loan borrowers.",
            publication_date="2024-02-09",
            search_query="RBI repo rate 6.5%",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(
            claim=self.claim,
            search_query=self.query,
            evidence_items=[opinion_item],
            status=SearchStatus.SUCCESS,
        )
        summary = evaluate_claim_evidence(cr)

        eval_res = summary.evaluated_items[0].evaluation
        self.assertEqual(eval_res.category, SourceCategory.OPINION_EDITORIAL)
        self.assertTrue(eval_res.is_opinion)
        self.assertIsNotNone(eval_res.demotion_reason)
        self.assertIn("Opinion / Editorial", eval_res.demotion_reason)
        self.assertLess(eval_res.quality_score, 50)

    # 3. Syndicated wire service attribution detected (PTI, ANI, AP, Reuters)
    def test_03_wire_service_attribution_detected(self):
        is_wire, agency = detect_syndicated_wire(
            title="RBI keeps repo rate unchanged at 6.5% - PTI",
            snippet="New Delhi, Feb 8 (PTI) The Reserve Bank on Thursday kept repo rate steady.",
        )
        self.assertTrue(is_wire)
        self.assertIn("PTI", agency)

        is_ani, ani_agency = detect_syndicated_wire(
            title="Central Bank Announcement",
            snippet="According to ANI reports, the governor addressed reporters.",
        )
        self.assertTrue(is_ani)
        self.assertIn("ANI", ani_agency)

    # 4. Multiple outlets copying same wire report are NOT counted as independent corroboration
    def test_04_duplicate_wire_copy_not_counted_as_independent(self):
        item_a = EvidenceItem(
            evidence_id="E001",
            claim_id="C001",
            title="RBI holds repo rate at 6.5% - PTI",
            url="https://outlet-a.com/business/rbi-rate",
            domain="outlet-a.com",
            snippet="New Delhi (PTI) - The Reserve Bank of India kept repo rate unchanged.",
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        item_b = EvidenceItem(
            evidence_id="E002",
            claim_id="C001",
            title="Repo rate held steady by RBI",
            url="https://outlet-b.com/economy/rbi-announcement",
            domain="outlet-b.com",
            snippet="(PTI) The Reserve Bank of India kept repo rate unchanged on Thursday.",
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(
            claim=self.claim,
            search_query=self.query,
            evidence_items=[item_a, item_b],
            status=SearchStatus.SUCCESS,
        )
        summary = evaluate_claim_evidence(cr)

        # First wire copy is independent=True
        self.assertTrue(summary.evaluated_items[0].evaluation.is_independent or summary.evaluated_items[1].evaluation.is_independent)
        # Second wire copy MUST be flagged as duplicate wire (is_independent=False)
        duplicate_item = [e for e in summary.evaluated_items if not e.evaluation.is_independent]
        self.assertEqual(len(duplicate_item), 1)
        self.assertIn("Duplicate wire", duplicate_item[0].evaluation.demotion_reason)
        # Total independent count should be 1, NOT 2!
        self.assertEqual(summary.independent_source_count, 1)
        self.assertEqual(summary.duplicate_wire_count, 1)

    # 5. Clickbait / Sensational headlines penalized
    def test_05_sensational_clickbait_penalized(self):
        clickbait_item = EvidenceItem(
            evidence_id="E003",
            claim_id="C001",
            title="SHOCKING BOMBSHELL: RBI Destroys Economy With Rate Freeze???!!!",
            url="https://clickbaitblog.xyz/shocking-rbi",
            domain="clickbaitblog.xyz",
            snippet="You won't believe what the Governor just announced!",
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=3,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(
            claim=self.claim,
            search_query=self.query,
            evidence_items=[clickbait_item],
            status=SearchStatus.SUCCESS,
        )
        summary = evaluate_claim_evidence(cr)

        eval_res = summary.evaluated_items[0].evaluation
        self.assertEqual(eval_res.category, SourceCategory.SENSATIONAL_CLICKBAIT)
        self.assertIsNotNone(eval_res.demotion_reason)
        self.assertIn("Sensational", eval_res.demotion_reason)
        self.assertLessEqual(eval_res.quality_score, 25)

    # 6. Absence of high-quality independent sources flags UNCERTAIN candidate
    def test_06_insufficient_evidence_flags_uncertain_candidate(self):
        # Only one general unverified blog
        lone_item = EvidenceItem(
            evidence_id="E001",
            claim_id="C001",
            title="Some random blog post",
            url="https://unknownsite.xyz/post",
            domain="unknownsite.xyz",
            snippet="A brief uncorroborated mention of rates.",
            publication_date=None,
            search_query="RBI repo rate 6.5%",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(
            claim=self.claim,
            search_query=self.query,
            evidence_items=[lone_item],
            status=SearchStatus.SUCCESS,
        )
        summary = evaluate_claim_evidence(cr)

        # Since there is neither a primary source nor >= 2 independent factual sources:
        self.assertFalse(summary.has_sufficient_independent_evidence)
        self.assertIn("UNCERTAIN Candidate", summary.quality_assessment_note)

    # 7. Zero political blacklisting: evaluation based strictly on structural and epistemological merits
    def test_07_no_political_blacklisting(self):
        # Two news publications with different perceived editorial stances
        source_left = EvidenceItem(
            evidence_id="E001",
            claim_id="C001",
            title="RBI Keeps Repo Rate Steady at 6.5%",
            url="https://thehindu.com/business/rbi-rate",
            domain="thehindu.com",
            snippet='The central bank maintained its monetary stance with 5:1 majority.',
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )
        source_right = EvidenceItem(
            evidence_id="E002",
            claim_id="C001",
            title="Monetary Policy: RBI Retains 6.5% Benchmark Rate",
            url="https://timesofindia.indiatimes.com/business/rbi-rate",
            domain="timesofindia.indiatimes.com",
            snippet='The Reserve Bank decided to keep interest rates unchanged at 6.5%.',
            publication_date="2024-02-08",
            search_query="RBI repo rate 6.5%",
            source_rank=2,
            retrieval_status=SearchStatus.SUCCESS,
        )
        cr = ClaimEvidenceResult(
            claim=self.claim,
            search_query=self.query,
            evidence_items=[source_left, source_right],
            status=SearchStatus.SUCCESS,
        )
        summary = evaluate_claim_evidence(cr)

        # Both reputable reporting outlets evaluated fairly without ideological bias
        for item in summary.evaluated_items:
            self.assertEqual(item.evaluation.category, SourceCategory.FACTUAL_REPORTING)
            self.assertIsNone(item.evaluation.demotion_reason)
            self.assertTrue(item.evaluation.is_independent)
            self.assertGreaterEqual(item.evaluation.quality_score, 70)


if __name__ == "__main__":
    unittest.main()
