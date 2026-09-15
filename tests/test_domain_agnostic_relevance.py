"""Domain-Agnostic Relevance and Verification Tests.

Verifies that the evidence retrieval, relevance evaluation, and claim verification
pipeline operates robustly across 18 distinct news and factual domains without
hardcoded heuristics or domain bias:

1. Indian politics
2. International politics
3. Science
4. Space
5. Technology
6. Economy
7. Business
8. Sports
9. Crime
10. Health
11. Environment
12. Education
13. Government schemes
14. History
15. Climate
16. Defense
17. Social issues
18. Viral / social-media claims
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


class TestDomainAgnosticRelevance(unittest.TestCase):
    """Verifies pipeline behavior across 18 diverse news and factual domains."""

    def _make_item(
        self,
        evidence_id: str,
        claim_id: str,
        title: str,
        snippet: str,
        url: str = "https://reputable-news.org/article",
        domain: str = "reputable-news.org",
        published_date: str = "2024-03-01",
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=evidence_id,
            claim_id=claim_id,
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            publication_date=published_date,
            search_query="test query",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )

    # 1. Indian Politics Domain
    def test_01_indian_politics_direct_corroboration(self):
        claim = Claim("POL_IN_01", "The Election Commission of India announced general elections in seven phases.", 0)
        item = self._make_item(
            "E_POL_IN",
            claim.claim_id,
            "ECI Press Conference: Lok Sabha Polls in 7 Phases",
            "The Election Commission of India on Saturday announced that the general elections will be conducted in seven phases starting April.",
            domain="eci.gov.in"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("ECI seven phases", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        self.assertEqual(len(summary.evaluated_items), 1)
        self.assertEqual(summary.evaluated_items[0].evaluation.relevance_level, RelevanceLevel.DIRECT)
        res = synthesize_claim_stance(claim, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 2. International Politics Domain
    def test_02_international_politics_diplomatic_accord(self):
        claim = Claim("POL_INT_01", "France and Germany signed a bilateral defense cooperation treaty in Aachen.", 0)
        item = self._make_item(
            "E_POL_INT",
            claim.claim_id,
            "Franco-German Friendship: Aachen Treaty Signed",
            "Leaders of France and Germany convened in Aachen to formally sign a renewed bilateral defense cooperation treaty.",
            domain="dw.com"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("France Germany Aachen treaty", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        self.assertEqual(summary.evaluated_items[0].evaluation.relevance_level, RelevanceLevel.DIRECT)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 3. Science Domain
    def test_03_science_cern_higgs_boson(self):
        claim = Claim("SCI_01", "CERN physicists observed a rare decay mode of the Higgs boson.", 0)
        item = self._make_item(
            "E_SCI_01",
            claim.claim_id,
            "CERN Collaboration Discovers Rare Higgs Boson Decay",
            "Physicists working on the ATLAS and CMS experiments at CERN have observed evidence of a rare decay channel of the Higgs boson.",
            domain="cern.ch"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("CERN Higgs boson rare decay", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        self.assertEqual(summary.evaluated_items[0].evaluation.relevance_level, RelevanceLevel.DIRECT)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 4. Space Domain
    def test_04_space_james_webb_exoplanet(self):
        claim = Claim("SPC_01", "James Webb Space Telescope detected water vapor in the atmosphere of exoplanet WASP-96b.", 0)
        item = self._make_item(
            "E_SPC_01",
            claim.claim_id,
            "NASA Webb Reveals Steamy Atmosphere on Distant Planet",
            "The James Webb Space Telescope has detected unambiguous signatures of water vapor in the atmosphere of exoplanet WASP-96b.",
            domain="nasa.gov"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("James Webb water vapor WASP-96b", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        self.assertEqual(summary.evaluated_items[0].evaluation.relevance_level, RelevanceLevel.DIRECT)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 5. Technology Domain
    def test_05_technology_semiconductor_fab(self):
        claim = Claim("TECH_01", "TSMC started commercial production of 3-nanometer semiconductor chips in Tainan.", 0)
        item = self._make_item(
            "E_TECH_01",
            claim.claim_id,
            "TSMC Kicks Off Mass Production of 3nm Chips in Tainan",
            "Taiwan Semiconductor Manufacturing Company (TSMC) announced it has initiated volume production of 3-nanometer chips at its Fab 18 in Tainan.",
            domain="reuters.com"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("TSMC 3-nanometer commercial production Tainan", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        self.assertEqual(summary.evaluated_items[0].evaluation.relevance_level, RelevanceLevel.DIRECT)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 6. Economy Domain
    def test_06_economy_inflation_contradiction(self):
        claim = Claim("ECON_01", "Consumer price index inflation dropped to 3.2% in November.", 0)
        item = self._make_item(
            "E_ECON_01",
            claim.claim_id,
            "Bureau of Labor Statistics: Consumer Price Index Summary",
            "The Consumer Price Index rose to 4.8% on an annual basis in November, defying expectations of a slowdown.",
            domain="bls.gov"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("CPI inflation 3.2% November", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.CONTRADICTED)

    # 7. Business Domain
    def test_07_business_merger_acquisition(self):
        claim = Claim("BIZ_01", "Microsoft completed its $68.7 billion acquisition of Activision Blizzard.", 0)
        item = self._make_item(
            "E_BIZ_01",
            claim.claim_id,
            "Microsoft Closes Activision Blizzard Deal",
            "Microsoft has officially finalized its $68.7 billion purchase of video game publisher Activision Blizzard after regulatory clearance.",
            domain="bloomberg.com"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("Microsoft completed Activision Blizzard 68.7 billion", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 8. Sports Domain
    def test_08_sports_world_cup_final(self):
        claim = Claim("SPT_01", "Argentina won the 2022 FIFA World Cup final in Qatar by defeating France in a penalty shootout.", 0)
        item = self._make_item(
            "E_SPT_01",
            claim.claim_id,
            "World Cup 2022 Final: Argentina Beats France on Penalties",
            "Argentina claimed the 2022 FIFA World Cup title in Qatar after beating France 4-2 on penalties following a thrilling 3-3 draw.",
            domain="fifa.com"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("Argentina won 2022 FIFA World Cup Qatar France penalties", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 9. Crime Domain
    def test_09_crime_interpol_red_notice(self):
        claim = Claim("CRM_01", "Interpol issued a Red Notice against the fugitive financier in Zurich.", 0)
        item = self._make_item(
            "E_CRM_01",
            claim.claim_id,
            "Interpol Wanted Bulletin: Red Notice Published",
            "Interpol has published an international Red Notice request for the provisional arrest of the fugitive financier who fled Zurich.",
            domain="interpol.int"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("Interpol Red Notice fugitive Zurich", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 10. Health Domain
    def test_10_health_malaria_vaccine_who(self):
        claim = Claim("HLT_01", "The World Health Organization recommended the R21/Matrix-M vaccine for malaria prevention in children.", 0)
        item = self._make_item(
            "E_HLT_01",
            claim.claim_id,
            "WHO Recommends R21/Matrix-M Vaccine for Malaria Prevention",
            "The World Health Organization has officially recommended the programmatic use of the R21/Matrix-M malaria vaccine to protect vulnerable children.",
            domain="who.int"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("WHO recommended R21 Matrix-M malaria vaccine", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 11. Environment Domain
    def test_11_environment_coral_bleaching(self):
        claim = Claim("ENV_01", "NOAA confirmed the fourth global mass coral bleaching event across 53 countries.", 0)
        item = self._make_item(
            "E_ENV_01",
            claim.claim_id,
            "NOAA Coral Reef Watch: Global Bleaching Declared",
            "Scientists at NOAA and international partners confirmed that the fourth global coral bleaching event has impacted reefs across 53 nations.",
            domain="noaa.gov"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("NOAA fourth global coral bleaching 53 countries", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 12. Education Domain
    def test_12_education_literacy_program(self):
        claim = Claim("EDU_01", "UNESCO reported that the global adult literacy rate reached 87% in 2023.", 0)
        item = self._make_item(
            "E_EDU_01",
            claim.claim_id,
            "UNESCO Institute for Statistics: Global Literacy Trends",
            "New survey findings released by UNESCO demonstrate that the worldwide adult literacy rate reached 87% during 2023.",
            domain="unesco.org"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("UNESCO global adult literacy rate 87% 2023", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 13. Government Schemes Domain
    def test_13_government_scheme_beneficiaries(self):
        claim = Claim("SCHEME_01", "The PM-KISAN program transferred 2,000 rupees to 80 million eligible farmers.", 0)
        item = self._make_item(
            "E_SCHEME_01",
            claim.claim_id,
            "Ministry of Agriculture: PM-KISAN Installment Released",
            "The direct benefit transfer under the PM-KISAN scheme disbursed 2,000 rupees directly to the bank accounts of 80 million farmer households.",
            domain="pib.gov.in"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("PM-KISAN 2,000 rupees 80 million farmers", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 14. History Domain
    def test_14_history_archaeological_excavation(self):
        claim = Claim("HIST_01", "Archaeologists discovered a 3,000-year-old Bronze Age settlement in Wiltshire.", 0)
        item = self._make_item(
            "E_HIST_01",
            claim.claim_id,
            "Historic England: Bronze Age Village Unearthed in Wiltshire",
            "Archaeological excavations in Wiltshire have uncovered the exceptionally well-preserved remnants of a 3,000-year-old Bronze Age settlement.",
            domain="historicengland.org.uk"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("archaeologists 3,000-year-old Bronze Age settlement Wiltshire", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 15. Climate Domain
    def test_15_climate_cop28_loss_and_damage(self):
        claim = Claim("CLM_01", "Delegates at COP28 approved the operationalization of the Loss and Damage climate fund in Dubai.", 0)
        item = self._make_item(
            "E_CLM_01",
            claim.claim_id,
            "UN Climate Press Release: COP28 Opens with Historic Agreement",
            "On the opening day of COP28 in Dubai, world leaders formally adopted and operationalized the Loss and Damage climate assistance fund.",
            domain="unfccc.int"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("COP28 operationalization Loss and Damage fund Dubai", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 16. Defense Domain
    def test_16_defense_air_defense_system(self):
        claim = Claim("DEF_01", "Sweden officially joined the NATO military alliance in Washington.", 0)
        item = self._make_item(
            "E_DEF_01",
            claim.claim_id,
            "NATO Welcomes Sweden as 32nd Allied Member",
            "Sweden officially completed its accession process and became the 32nd member of the NATO defensive alliance in Washington DC.",
            domain="nato.int"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("Sweden officially joined NATO alliance Washington", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 17. Social Issues Domain
    def test_17_social_issues_gender_pay_gap(self):
        claim = Claim("SOC_01", "The national statistical agency reported that the median gender wage gap widened to 14%.", 0)
        item = self._make_item(
            "E_SOC_01",
            claim.claim_id,
            "National Labour Force Survey: Gender Earnings Disparity",
            "Official statistical tables published by the labor agency reveal that the median gender wage gap expanded to 14% over the preceding fiscal year.",
            domain="ons.gov.uk"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("median gender wage gap widened 14%", []), [item], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        res = synthesize_claim_stance(claim, summary.evaluated_items)
        self.assertEqual(res.final_stance, ClaimStance.SUPPORTED)

    # 18. Viral / Social-Media Claims Domain
    def test_18_viral_social_media_hoax_debunked(self):
        claim = Claim("VIRAL_01", "Social media posts claimed the Eiffel Tower in Paris was destroyed by fire.", 0)
        item_debunk = self._make_item(
            "E_VIRAL_01",
            claim.claim_id,
            "Fact Check: Viral Video of Eiffel Tower on Fire is Computer-Generated Hoax",
            "Paris police and monument authorities confirmed there was no fire at the Eiffel Tower; viral social media clips circulating online were debunked as AI CGI simulations.",
            domain="afp.com"
        )
        cr = ClaimEvidenceResult(claim, SearchQuery("Eiffel Tower Paris fire hoax debunked", []), [item_debunk], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)
        # Check that refutation marker correctly triggers contradiction of the viral fire assertion
        analysis = analyze_evidence_against_claim(claim, summary.evaluated_items[0])
        self.assertEqual(analysis.stance, EvidenceStance.CONTRADICTS)
        self.assertTrue(any("debunk" in p.lower() or "refutation" in p.lower() or "denial" in p.lower() for p in analysis.conflicting_points))


if __name__ == "__main__":
    unittest.main()
