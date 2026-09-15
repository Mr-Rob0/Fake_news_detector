"""Regression tests for claim-level independent-evidence consensus.

These use neutral synthetic entities so calibration rules cannot encode a
particular article, person, outlet, or political topic.
"""
import unittest

from src.claim_extractor import Claim
from src.evidence_retriever import EvidenceItem
from src.search_provider import SearchStatus
from src.source_evaluator import (
    EvidenceStrength, EvaluatedEvidenceItem, RelevanceLevel, SourceCategory,
    SourceEvaluation,
)
from src.claim_analyzer import ClaimStance, EvidenceStance, analyze_evidence_against_claim, synthesize_claim_stance


class TestConsensusAggregation(unittest.TestCase):
    claim = Claim("C1", "Avery Stone was found dead in a hotel in Metro City in January 2024.", 0)

    def item(self, eid, domain, text, *, quality=75, strength=EvidenceStrength.STRONG,
             independent=True, relevance=RelevanceLevel.DIRECT, primary=False):
        return EvaluatedEvidenceItem(
            EvidenceItem(eid, "C1", text, f"https://{domain}/{eid}", domain, text,
                         "2024-01-22", "query", 1, SearchStatus.SUCCESS),
            SourceEvaluation(eid, SourceCategory.PRIMARY_OFFICIAL if primary else SourceCategory.FACTUAL_REPORTING,
                             quality, primary, False, not independent, is_independent=independent,
                             relevance_level=relevance, evidence_strength=strength),
        )

    def test_two_independent_direct_reports_are_supported(self):
        a = self.item("E1", "report-one.example", "Avery Stone was found dead at a hotel in Metro City in January 2024.")
        b = self.item("E2", "report-two.example", "Police said Avery Stone was found dead in a Metro City hotel in January 2024.")
        result = synthesize_claim_stance(self.claim, [a, b])
        self.assertEqual(result.final_stance, ClaimStance.SUPPORTED)
        self.assertEqual(result.independent_supporting_streams, 2)
        self.assertEqual(result.direct_supporting_evidence_count, 2)

    def test_weak_or_incomplete_result_does_not_cancel_support(self):
        support_a = self.item("E1", "report-one.example", "Avery Stone was found dead at a hotel in Metro City in January 2024.")
        support_b = self.item("E2", "report-two.example", "Avery Stone was found dead in a Metro City hotel in January 2024.")
        incomplete = self.item("E3", "archive.example", "Avery Stone attended an event in Metro City.", quality=45,
                               strength=EvidenceStrength.WEAK, relevance=RelevanceLevel.WEAK)
        result = synthesize_claim_stance(self.claim, [support_a, support_b, incomplete])
        self.assertEqual(result.final_stance, ClaimStance.SUPPORTED)
        self.assertIn("E3", result.neutral_evidence_ids)
        self.assertNotIn("E3", result.contradicting_evidence_ids)

    def test_strong_evidence_on_both_sides_is_uncertain(self):
        support = self.item("E1", "report-one.example", "Avery Stone was found dead at a hotel in Metro City in January 2024.")
        denial = self.item("E2", "official.example", "Officials said reports that Avery Stone was found dead in a hotel were entirely false.")
        result = synthesize_claim_stance(self.claim, [support, denial])
        self.assertEqual(result.final_stance, ClaimStance.UNCERTAIN)

    def test_duplicate_copies_do_not_create_independent_corroboration(self):
        original = self.item("E1", "report-one.example", "Avery Stone was found dead at a hotel in Metro City in January 2024.")
        duplicate = self.item("E2", "copy.example", "Avery Stone was found dead at a hotel in Metro City in January 2024.",
                              independent=False)
        result = synthesize_claim_stance(self.claim, [original, duplicate])
        self.assertEqual(result.independent_supporting_streams, 1)
        self.assertEqual(result.syndicated_duplicate_count, 1)

    def test_historical_statement_is_not_contradicted_by_earlier_life_event(self):
        evidence = self.item("E1", "archive.example", "Avery Stone was alive in Metro City in 2023.")
        analysis = analyze_evidence_against_claim(self.claim, evidence)
        self.assertNotEqual(analysis.stance, EvidenceStance.CONTRADICTS)


if __name__ == "__main__":
    unittest.main()
