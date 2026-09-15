"""Adversarial Relevance and False Contradiction Prevention Test Suite.

Contains 17 adversarial test scenarios specifically designed to stress-test
the evidence retrieval, relevance gating, and stance aggregation pipeline:

1. Same keyword, different event (The Agrasar study critical bug scenario)
2. Same person, different event
3. Same organization, different study
4. Same location, different year
5. Same number, different topic
6. Related topic but not the same claim
7. Source contains claim words but contradicts context
8. Source discusses background only
9. Irrelevant high-quality source
10. Low-quality source that directly addresses the claim
11. Two syndicated wire copies appearing independent
12. Conflicting numbers on the same metric / study
13. Negation handling (affirmative vs negative)
14. Attributed allegation vs established fact
15. Outdated evidence vs historical claims
16. Partial evidence
17. Headline matches but article body discusses sidebar / different content
"""

import unittest
from src.claim_extractor import Claim
from src.search_provider import SearchStatus
from src.evidence_retriever import EvidenceItem, ClaimEvidenceResult, SearchQuery
from src.source_evaluator import (
    RelevanceLevel,
    SourceCategory,
    IndependenceStatus,
    evaluate_claim_evidence,
    evaluate_result_relevance,
)
from src.claim_analyzer import (
    ClaimStance,
    EvidenceStance,
    synthesize_claim_stance,
    analyze_evidence_against_claim,
)


class TestAdversarialRelevance(unittest.TestCase):
    """Adversarial tests specifically crafted to probe false relevance and false contradiction vulnerabilities."""

    def _make_item(
        self,
        evidence_id: str,
        claim_id: str,
        title: str,
        snippet: str,
        domain: str = "reputable-news.org",
        url: str = "https://reputable-news.org/article",
        published_date: str = "2024-01-01",
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=evidence_id,
            claim_id=claim_id,
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            publication_date=published_date,
            search_query="adversarial test query",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )

    # -------------------------------------------------------------------------
    # 1. Same keyword, different event: The Agrasar Study Critical Bug Scenario
    # -------------------------------------------------------------------------
    def test_01_same_keyword_different_event_agrasar_bug(self):
        """CRITICAL: An article about 'The Nonprofit Starvation Cycle' sharing generic

        words ('nonprofit', 'study') must NEVER be marked relevant or contradict
        the Agrasar Gurugram study claim.
        """
        claim = Claim(
            "C_AGRASAR",
            "A 2018 study by Agrasar, a nonprofit, found that nearly 80 percent of low-income schoolchildren surveyed in Gurugram city near New Delhi reported being beaten several times a week, and a majority of their parents approved of it and used it themselves at home.",
            0,
        )

        # Irrelevant source A: Stanford Social Innovation Review on Nonprofit Starvation Cycle
        item_starvation = self._make_item(
            "E_IRREL_1",
            claim.claim_id,
            "The Nonprofit Starvation Cycle",
            "A landmark study shows that 53% of nonprofit organizations face overhead starvation and lack adequate operating reserves.",
            domain="ssir.org",
        )

        # Irrelevant source B: National Center for Charitable Statistics
        item_nccs = self._make_item(
            "E_IRREL_2",
            claim.claim_id,
            "National Center for Charitable Statistics: Annual Nonprofit Overview",
            "The study provides an analytical overview of over 1.5 million nonprofit organizations registered across the United States.",
            domain="nccs.urban.org",
        )

        cr = ClaimEvidenceResult(
            claim,
            SearchQuery("Agrasar 2018 Gurugram study 80 percent", []),
            [item_starvation, item_nccs],
            SearchStatus.SUCCESS,
        )

        summary = evaluate_claim_evidence(cr)

        # Both sources must be classified as IRRELEVANT and filtered at the Relevance Gate
        self.assertEqual(len(summary.evaluated_items), 0)
        self.assertEqual(len(summary.filtered_irrelevant_items), 2)
        self.assertEqual(summary.filtered_irrelevant_items[0].evaluation.relevance_level, RelevanceLevel.IRRELEVANT)
        self.assertEqual(summary.filtered_irrelevant_items[1].evaluation.relevance_level, RelevanceLevel.IRRELEVANT)

        # Synthesized verdict must NEVER be a false dispute! Must be UNCERTAIN (insufficient evidence)
        res = synthesize_claim_stance(claim, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertEqual(res.final_stance, ClaimStance.UNCERTAIN)
        self.assertNotIn("conflicting evidence", res.explanation.lower())
        self.assertIn("sufficiently relevant", res.explanation.lower())
        self.assertEqual(len(res.contradicting_evidence_ids), 0)
        self.assertEqual(len(res.supporting_evidence_ids), 0)

    # -------------------------------------------------------------------------
    # 2. Same person, different event
    # -------------------------------------------------------------------------
    def test_02_same_person_different_event(self):
        """Mentions the same political leader but describes a completely different event."""
        claim = Claim("C_PERS", "Prime Minister Modi inaugurated the new semiconductor plant in Dholera.", 0)
        item_diff_event = self._make_item(
            "E_PERS_DIFF",
            claim.claim_id,
            "Prime Minister Modi Addresses Rally in Varanasi",
            "Prime Minister Narendra Modi addressed a massive gathering in Varanasi highlighting spiritual tourism and infrastructure projects.",
            domain="pib.gov.in",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("Modi semiconductor Dholera", []), [item_diff_event], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)

        # Must not be DIRECT, must not produce SUPPORTED
        res = synthesize_claim_stance(claim, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertNotEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------------------
    # 3. Same organization, different study
    # -------------------------------------------------------------------------
    def test_03_same_organization_different_study(self):
        """World Bank study on clean energy should not verify a World Bank study on rural poverty."""
        claim = Claim("C_ORG", "The World Bank published a study indicating rural poverty dropped to 11%.", 0)
        item_diff_study = self._make_item(
            "E_ORG_DIFF",
            claim.claim_id,
            "World Bank Global Energy Transition Report",
            "The World Bank released a comprehensive research paper examining renewable electricity investments in sub-Saharan Africa.",
            domain="worldbank.org",
        )

        analysis = analyze_evidence_against_claim(
            claim,
            evaluate_claim_evidence(
                ClaimEvidenceResult(claim, SearchQuery("World Bank rural poverty 11%", []), [item_diff_study], SearchStatus.SUCCESS)
            ).evaluated_items[0] if evaluate_claim_evidence(
                ClaimEvidenceResult(claim, SearchQuery("World Bank rural poverty 11%", []), [item_diff_study], SearchStatus.SUCCESS)
            ).evaluated_items else evaluate_claim_evidence(
                ClaimEvidenceResult(claim, SearchQuery("World Bank rural poverty 11%", []), [item_diff_study], SearchStatus.SUCCESS)
            ).filtered_irrelevant_items[0]
        )
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)

    # -------------------------------------------------------------------------
    # 4. Same location, different year
    # -------------------------------------------------------------------------
    def test_04_same_location_different_year(self):
        """A source discussing a 2014 flood cannot verify or contradict a 2024 flood claim."""
        claim = Claim("C_LOC_YR", "Flash floods in Uttarakhand displaced 12,000 residents in July 2024.", 0)
        item_old_year = self._make_item(
            "E_OLD_YR",
            claim.claim_id,
            "Uttarakhand Disaster Memorial",
            "In July 2013, devastating cloudbursts and flash floods across Uttarakhand caused catastrophic destruction in Kedarnath.",
            published_date="2014-06-15",
            domain="thehindu.com",
        )

        analysis = analyze_evidence_against_claim(
            claim,
            evaluate_claim_evidence(
                ClaimEvidenceResult(claim, SearchQuery("Uttarakhand flash floods 12,000 2024", []), [item_old_year], SearchStatus.SUCCESS)
            ).evaluated_items[0]
        )
        # Should be flagged as temporal mismatch or neutral
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)
        self.assertTrue(any("temporal" in p.lower() for p in [analysis.rationale]))

    # -------------------------------------------------------------------------
    # 5. Same number, different topic
    # -------------------------------------------------------------------------
    def test_05_same_number_different_topic(self):
        """A source with 80% on another topic must not be treated as supporting evidence."""
        claim = Claim("C_NUM_DIFF", "Survey found that 80% of urban commuters prefer metro transit over buses.", 0)
        item_num_unrelated = self._make_item(
            "E_NUM_UNRELATED",
            claim.claim_id,
            "Smartphone Battery Usage Survey",
            "Research shows that 80% of smartphone users charge their mobile devices overnight.",
            domain="techradar.com",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("80% urban commuters metro", []), [item_num_unrelated], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)

        # Smartphone battery study is irrelevant to metro transit
        self.assertEqual(len(summary.evaluated_items), 0)
        self.assertEqual(len(summary.filtered_irrelevant_items), 1)

    # -------------------------------------------------------------------------
    # 6. Related topic but not the same claim
    # -------------------------------------------------------------------------
    def test_06_related_topic_not_same_claim(self):
        """Company quarterly revenue versus launching an AI model."""
        claim = Claim("C_TOPIC", "Anthropic released the Claude 3.5 Sonnet language model.", 0)
        item_topic_only = self._make_item(
            "E_TOPIC_ONLY",
            claim.claim_id,
            "Anthropic Raises Series C Financing Round",
            "AI safety research lab Anthropic announced it raised fresh capital from institutional tech investors.",
            domain="techcrunch.com",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("Anthropic released Claude 3.5 Sonnet", []), [item_topic_only], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertEqual(res.final_stance, ClaimStance.UNCERTAIN)

    # -------------------------------------------------------------------------
    # 7. Source contains claim words but contradicts context
    # -------------------------------------------------------------------------
    def test_07_claim_words_in_different_context(self):
        """Words 'police', 'arrested', 'hospital' in an article about a hospital donation."""
        claim = Claim("C_WORDS", "Police arrested the hospital administrator for fraud.", 0)
        item_context = self._make_item(
            "E_WORDS",
            claim.claim_id,
            "Community Police Foundation Donates Equipment to Hospital",
            "Local police representatives visited the memorial hospital to donate diagnostic medical equipment.",
            domain="localnews.org",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("police arrested hospital administrator fraud", []), [item_context], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertNotEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------------------
    # 8. Source discusses background only
    # -------------------------------------------------------------------------
    def test_08_background_only_source(self):
        """General background on inflation history cannot verify a specific rate decision."""
        claim = Claim("C_BG", "The Federal Reserve cut interest rates by 50 basis points.", 0)
        item_bg = self._make_item(
            "E_BG",
            claim.claim_id,
            "Explainer: How the Federal Reserve Operates",
            "An educational guide describing the history and governance structure of the Federal Reserve system.",
            domain="investopedia.com",
        )

        analysis = analyze_evidence_against_claim(
            claim,
            evaluate_claim_evidence(
                ClaimEvidenceResult(claim, SearchQuery("Federal Reserve rate cut 50 bps", []), [item_bg], SearchStatus.SUCCESS)
            ).evaluated_items[0]
        )
        self.assertEqual(analysis.stance, EvidenceStance.NEUTRAL_IRRELEVANT)

    # -------------------------------------------------------------------------
    # 9. Irrelevant high-quality source
    # -------------------------------------------------------------------------
    def test_09_irrelevant_high_quality_source(self):
        """A prestigious WHO report about diabetes is still irrelevant to an Ebola outbreak claim."""
        claim = Claim("C_WHO", "The WHO declared the end of the Ebola virus outbreak in Uganda.", 0)
        item_who_diabetes = self._make_item(
            "E_WHO_DIAB",
            claim.claim_id,
            "WHO Global Diabetes Compact: Key Milestones",
            "World Health Organization report highlighting global initiatives to increase insulin access for diabetes patients.",
            domain="who.int",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("WHO declared end Ebola outbreak Uganda", []), [item_who_diabetes], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)

        # High quality does NOT override irrelevance!
        self.assertEqual(len(summary.filtered_irrelevant_items), 1)
        self.assertEqual(summary.filtered_irrelevant_items[0].evaluation.relevance_level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # 10. Low-quality source that directly addresses the claim
    # -------------------------------------------------------------------------
    def test_10_low_quality_source_direct_address(self):
        """A personal blog directly corroborating an assertion receives low evidentiary weight."""
        claim = Claim("C_LOW_Q", "The municipal council approved the new bicycle lane project.", 0)
        item_blog = self._make_item(
            "E_BLOG",
            claim.claim_id,
            "My City Diary: Bicycle Lane Approved",
            "The municipal council voted today to approve the new bicycle lane project across downtown.",
            domain="randomblogger.blogspot.com",
            url="http://randomblogger.blogspot.com/post/123",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("municipal council bicycle lane", []), [item_blog], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        # Relevant, but quality score reflects general web / blog
        self.assertLess(summary.evaluated_items[0].evaluation.quality_score, 65)

    # -------------------------------------------------------------------------
    # 11. Two syndicated copies appearing independent
    # -------------------------------------------------------------------------
    def test_11_syndicated_wire_copies_deduplication(self):
        """Two outlets publishing the identical Reuters wire feed should be identified as syndicated."""
        claim = Claim("C_WIRE", "Oil prices settled at $82 a barrel following OPEC production cuts.", 0)
        item_1 = self._make_item(
            "E_WIRE_1",
            claim.claim_id,
            "Oil settles at $82 on OPEC output cuts (Reuters)",
            "Brent crude futures settled at $82 per barrel on Tuesday following OPEC+ supply restrictions, Reuters reports.",
            domain="news-portal-a.com",
        )
        item_2 = self._make_item(
            "E_WIRE_2",
            claim.claim_id,
            "Oil settles at $82 on OPEC output cuts - Reuters News",
            "Brent crude futures settled at $82 per barrel on Tuesday following OPEC+ supply restrictions, Reuters reports.",
            domain="news-portal-b.com",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("Oil prices $82 OPEC", []), [item_1, item_2], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)

        # Independence analysis must catch wire syndication
        syndicated_items = [it for it in summary.evaluated_items if it.evaluation.is_syndicated_wire]
        self.assertGreaterEqual(len(syndicated_items), 1)

    # -------------------------------------------------------------------------
    # 12. Conflicting numbers on the same metric / study
    # -------------------------------------------------------------------------
    def test_12_conflicting_numbers_same_study(self):
        """Evidence stating 20% on the same study must produce genuine contradiction."""
        claim = Claim("C_NUM_CONF", "The national education survey found that 80% of schools lacked libraries.", 0)
        item_diff_num = self._make_item(
            "E_NUM_CONF",
            claim.claim_id,
            "Education Ministry Survey Findings",
            "The national education survey concluded that only 20% of schools lacked dedicated libraries.",
            domain="education.gov.in",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("education survey 80% schools libraries", []), [item_diff_num], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------------------
    # 13. Negation handling (affirmative claim vs negative evidence)
    # -------------------------------------------------------------------------
    def test_13_negation_handling(self):
        """Claim asserts police did not use water cannons, evidence asserts police used them."""
        claim_neg = Claim("C_NEG", "Police did not use water cannons during the demonstration.", 0)
        item_pos = self._make_item(
            "E_POS",
            claim_neg.claim_id,
            "Protest Report: Security Forces Deploy Water Cannons",
            "Police used water cannons and tear gas during the demonstration to disperse crowds.",
            domain="reuters.com",
        )

        cr = ClaimEvidenceResult(claim_neg, SearchQuery("police water cannons demonstration", []), [item_pos], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim_neg, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------------------
    # 14. Attributed allegation vs established fact
    # -------------------------------------------------------------------------
    def test_14_attributed_allegation_handling(self):
        """Distinguish attributed statement from objective proven fact."""
        claim = Claim("C_ATTRIB", "Opposition leaders alleged the government manipulated procurement bids.", 0)
        item_allegation = self._make_item(
            "E_ATTRIB",
            claim.claim_id,
            "Opposition Press Briefing on Procurement",
            "Opposition leaders alleged that the government manipulated public procurement tenders during recent bidding.",
            domain="thehindu.com",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("opposition alleged manipulated procurement bids", []), [item_allegation], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        # Verifies that the allegation was made, but notes attribution
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------------------
    # 15. Outdated evidence vs historical claims
    # -------------------------------------------------------------------------
    def test_15_historical_claims_preserved(self):
        """Historical claims are legitimate and should not be discarded merely for referencing the past."""
        claim_hist = Claim("C_HIST", "Apollo 11 landed astronauts on the Moon in July 1969.", 0)
        item_hist = self._make_item(
            "E_HIST",
            claim_hist.claim_id,
            "NASA History: Apollo 11 Mission Details",
            "On July 20, 1969, American astronauts Neil Armstrong and Buzz Aldrin landed the Apollo 11 lunar module on the Moon.",
            domain="nasa.gov",
            published_date="2019-07-20",
        )

        cr = ClaimEvidenceResult(claim_hist, SearchQuery("Apollo 11 landed astronauts Moon July 1969", []), [item_hist], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim_hist, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------------------
    # 16. Partial evidence
    # -------------------------------------------------------------------------
    def test_16_partial_evidence_handling(self):
        """A source providing partial background cannot unilaterally verify a complex predicate."""
        claim_complex = Claim("C_PARTIAL", "Government sanctioned 500 million dollars to build 40 new solar parks across Rajasthan.", 0)
        item_partial = self._make_item(
            "E_PARTIAL",
            claim_complex.claim_id,
            "Ministry of New and Renewable Energy Update",
            "The ministry announced ongoing renewable energy development plans for the state of Rajasthan.",
            domain="mnre.gov.in",
        )

        cr = ClaimEvidenceResult(claim_complex, SearchQuery("sanctioned 500 million 40 solar parks Rajasthan", []), [item_partial], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        # Should be classified as PARTIAL or WEAK, resulting in UNCERTAIN
        res = synthesize_claim_stance(claim_complex, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertEqual(res.final_stance, ClaimStance.UNCERTAIN)

    # -------------------------------------------------------------------------
    # 17. Headline matches but article body discusses sidebar / different content
    # -------------------------------------------------------------------------
    def test_17_headline_matches_body_is_sidebar(self):
        """Headline contains keywords but snippet shows unrelated search directory text."""
        claim = Claim("C_SIDEBAR", "Supreme Court upheld the validity of the electoral bond scheme.", 0)
        item_sidebar = self._make_item(
            "E_SIDEBAR",
            claim.claim_id,
            "Supreme Court Cases Overview - Legal Directory",
            "Search our legal database for legal advice, divorce attorneys, property law, and Supreme Court judgments.",
            domain="legal-directory-search.com",
        )

        cr = ClaimEvidenceResult(claim, SearchQuery("Supreme Court electoral bond scheme validity", []), [item_sidebar], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        self.assertEqual(len(summary.filtered_irrelevant_items), 1)
        self.assertEqual(summary.filtered_irrelevant_items[0].evaluation.relevance_level, RelevanceLevel.IRRELEVANT)


if __name__ == "__main__":
    unittest.main()
