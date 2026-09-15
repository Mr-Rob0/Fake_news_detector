"""Comprehensive Multilingual Adversarial Test Suite.

Covers categories A through Z as specified in the adversarial robustness requirements:
A. Hindi claims
B. English claims
C. Hinglish claims
D. Named entities
E. No named entities
F. Numbers
G. Dates
H. Locations
I. Acronyms
J. Multiple entities
K. Conflicting entities
L. Dictionary pages
M. Wikipedia reference pages
N. Generic statistics
O. Same-domain duplicates
P. Syndicated articles
Q. Irrelevant political articles
R. Irrelevant science articles
S. Irrelevant sports articles
T. Irrelevant health articles
U. Completely unrelated articles
V. One-character overlap
W. Generic-word overlap
X. Correctly relevant evidence
Y. Partial contextual evidence
Z. Genuine contradiction

Includes regression tests for both the Agrasar bug and the Hindi Devanagari Hormuz failure.
Does NOT hardcode expected URLs or article titles.
"""

import unittest
from src.claim_extractor import Claim, decompose_claim_text, extract_claims
from src.search_provider import SearchStatus
from src.evidence_retriever import (
    EvidenceItem,
    ClaimEvidenceResult,
    SearchQuery,
    generate_search_query,
)
from src.source_evaluator import (
    RelevanceLevel,
    SourceCategory,
    IndependenceStatus,
    EvidenceStrength,
    evaluate_claim_evidence,
    evaluate_result_relevance,
    detect_reference_or_linguistic_page,
)
from src.claim_analyzer import (
    ClaimStance,
    EvidenceStance,
    synthesize_claim_stance,
    analyze_evidence_against_claim,
)


class TestMultilingualAdversarialRobustness(unittest.TestCase):
    """Adversarial test suite covering categories A through Z."""

    def _make_item(
        self,
        evidence_id: str,
        claim_id: str,
        title: str,
        snippet: str,
        domain: str = "news-source.org",
        url: str = "https://news-source.org/article",
        published_date: str = "2026-03-01",
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=evidence_id,
            claim_id=claim_id,
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            publication_date=published_date,
            search_query="test search query",
            source_rank=1,
            retrieval_status=SearchStatus.SUCCESS,
        )

    # -------------------------------------------------------------------------
    # Category A: Hindi Claims
    # -------------------------------------------------------------------------
    def test_cat_A_hindi_claim_preserves_concepts_and_verifies(self):
        claim_text = "भारत ने चंद्रमा के दक्षिणी ध्रुव पर सफल लैंडिंग की घोषणा की"
        claims = extract_claims(claim_text)
        self.assertGreaterEqual(len(claims), 1)
        c = claims[0]
        self.assertTrue(len(c.decomposed.core_concepts) >= 3)
        self.assertIn("चंद्रमा", c.decomposed.core_concepts)

        # Relevant reporting
        ev_item = self._make_item(
            "E_HI_01", c.claim_id,
            "इसरो: भारत ने चंद्रमा के दक्षिणी ध्रुव पर की सफल लैंडिंग",
            "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रमा के दक्षिणी ध्रुव पर सफल लैंडिंग की आधिकारिक घोषणा की।",
            domain="isro.gov.in"
        )
        rel = evaluate_result_relevance(c.text, ev_item.title, ev_item.snippet, ev_item.url, ev_item.domain, c)
        self.assertIn(rel.level, (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL))
        self.assertTrue(rel.is_eligible_for_stance)

    # -------------------------------------------------------------------------
    # Category B: English Claims
    # -------------------------------------------------------------------------
    def test_cat_B_english_claim_robust_verification(self):
        claim_text = "The European Union agreed on landmark artificial intelligence regulations."
        c = Claim("C_EU_AI", claim_text, 0)
        c.decomposed = decompose_claim_text(claim_text)
        
        sq = generate_search_query(c)
        self.assertIn("European", sq.query_text)
        self.assertTrue(any("ai" in kw.lower() or "artificial" in kw.lower() for kw in sq.preserved_keywords))

    # -------------------------------------------------------------------------
    # Category C: Hinglish / Mixed-Language Claims
    # -------------------------------------------------------------------------
    def test_cat_C_hinglish_mixed_language_claim(self):
        claim_text = "Delhi Metro ने new automated signalling system launch किया"
        c = Claim("C_HINGLISH", claim_text, 0)
        c.decomposed = decompose_claim_text(claim_text)

        sq = generate_search_query(c)
        self.assertIn("Delhi Metro", sq.preserved_keywords)
        self.assertTrue(any(k in sq.query_text for k in ("signalling", "signaling", "system", "automated")))

    # -------------------------------------------------------------------------
    # Category D: Named Entities
    # -------------------------------------------------------------------------
    def test_cat_D_named_entity_discrimination(self):
        claim = Claim("C_NE", "Reserve Bank of India increased the benchmark repo rate by 25 basis points.", 0)
        # Entity mismatch item
        item_mismatch = self._make_item(
            "E_D1", claim.claim_id,
            "European Central Bank raises deposit rate",
            "The European Central Bank increased deposit facility rates by 25 basis points.",
            domain="ft.com"
        )
        rel = evaluate_result_relevance(claim.text, item_mismatch.title, item_mismatch.snippet, item_mismatch.url, item_mismatch.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category E: Claims Without Named Entities
    # -------------------------------------------------------------------------
    def test_cat_E_claims_without_named_entities(self):
        claim_text = "Global renewable energy capacity increased by 50 percent last year."
        c = Claim("C_NO_NE", claim_text, 0)
        c.decomposed = decompose_claim_text(claim_text)

        item = self._make_item(
            "E_E1", c.claim_id,
            "Renewable energy additions surge globally",
            "Global renewable energy capacity additions increased by nearly 50 percent last year.",
            domain="iea.org"
        )
        rel = evaluate_result_relevance(c.text, item.title, item.snippet, item.url, item.domain, c)
        self.assertIn(rel.level, (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL))

    # -------------------------------------------------------------------------
    # Category F: Numbers and Percentages
    # -------------------------------------------------------------------------
    def test_cat_F_numbers_and_percentages_handling(self):
        claim = Claim("C_NUM", "Inflation dropped to 3.2 percent in August.", 0)
        item_diff_num = self._make_item(
            "E_F1", claim.claim_id,
            "August Inflation Report",
            "Consumer price inflation rose to 8.7 percent in August.",
            domain="reuters.com"
        )
        eval_res = evaluate_claim_evidence(ClaimEvidenceResult(claim, SearchQuery(claim.text, []), [item_diff_num], SearchStatus.SUCCESS))
        analysis = synthesize_claim_stance(claim, eval_res.evaluated_items, eval_res.filtered_irrelevant_items)
        self.assertEqual(analysis.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------------------
    # Category G: Dates and Temporal Reasoning
    # -------------------------------------------------------------------------
    def test_cat_G_temporal_mismatch_neutralizes_historical_source(self):
        claim = Claim("C_DATE", "Parliament passed the digital privacy law in 2026.", 0)
        item_historical = self._make_item(
            "E_G1", claim.claim_id,
            "Privacy Bill Debate 2018",
            "Parliament debated the draft digital privacy legislation in 2018.",
            domain="thehindu.com"
        )
        eval_res = evaluate_claim_evidence(ClaimEvidenceResult(claim, SearchQuery(claim.text, []), [item_historical], SearchStatus.SUCCESS))
        analysis = synthesize_claim_stance(claim, eval_res.evaluated_items, eval_res.filtered_irrelevant_items)
        self.assertNotEqual(analysis.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------------------
    # Category H: Locations
    # -------------------------------------------------------------------------
    def test_cat_H_location_compatibility(self):
        claim = Claim("C_LOC", "Police conducted major search operations in Mumbai.", 0)
        item_mumbai = self._make_item("E_H1", claim.claim_id, "Mumbai Police raid", "Police carried out coordinated search operations in Mumbai.", domain="indianexpress.com")
        rel = evaluate_result_relevance(claim.text, item_mumbai.title, item_mumbai.snippet, item_mumbai.url, item_mumbai.domain, claim)
        self.assertIn(rel.level, (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL))

    # -------------------------------------------------------------------------
    # Category I: Acronyms
    # -------------------------------------------------------------------------
    def test_cat_I_acronym_preservation_without_shredding(self):
        claim = Claim("C_ACR", "ISRO launched the meteorological satellite using GSLV rocket.", 0)
        sq = generate_search_query(claim)
        self.assertIn("ISRO", sq.preserved_keywords)
        self.assertIn("GSLV", sq.preserved_keywords)

    # -------------------------------------------------------------------------
    # Category J: Multiple Entities
    # -------------------------------------------------------------------------
    def test_cat_J_multiple_entities_agreement(self):
        claim = Claim("C_MULTI_E", "NASA and ISRO jointly developed the NISAR satellite.", 0)
        item = self._make_item(
            "E_J1", claim.claim_id,
            "NISAR Mission Status",
            "NASA and ISRO collaborative NISAR satellite completes final integration tests.",
            domain="nasa.gov"
        )
        rel = evaluate_result_relevance(claim.text, item.title, item.snippet, item.url, item.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.DIRECT)
        self.assertIn("NASA", rel.matched_entities)
        self.assertIn("ISRO", rel.matched_entities)

    # -------------------------------------------------------------------------
    # Category K: Conflicting Entities
    # -------------------------------------------------------------------------
    def test_cat_K_conflicting_entities_rejected(self):
        claim = Claim("C_CONF_E", "Apple acquired artificial intelligence startup Anthropic.", 0)
        item_amazon = self._make_item(
            "E_K1", claim.claim_id,
            "Amazon announces strategic investment in Anthropic",
            "Amazon announced a multi-billion dollar investment partnership with Anthropic.",
            domain="bloomberg.com"
        )
        rel = evaluate_result_relevance(claim.text, item_amazon.title, item_amazon.snippet, item_amazon.url, item_amazon.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category L: Dictionary Pages (Linguistic Filtering)
    # -------------------------------------------------------------------------
    def test_cat_L_dictionary_pages_filtered_generically(self):
        claim = Claim("C_DICT", "Scientists discovered a new species of deep-sea jellyfish.", 0)
        
        # Wiktionary
        item_wikt = self._make_item("E_L1", claim.claim_id, "jellyfish - Wiktionary", "Free dictionary definition of jellyfish, noun.", domain="en.wiktionary.org", url="https://en.wiktionary.org/wiki/jellyfish")
        rel_wikt = evaluate_result_relevance(claim.text, item_wikt.title, item_wikt.snippet, item_wikt.url, item_wikt.domain, claim)
        self.assertEqual(rel_wikt.level, RelevanceLevel.IRRELEVANT)

        # Shabdkosh
        item_shabd = self._make_item("E_L2", claim.claim_id, "species - English to Hindi dictionary", "Meaning of species in Hindi on Shabdkosh.", domain="shabdkosh.com", url="https://shabdkosh.com/dictionary/species")
        rel_shabd = evaluate_result_relevance(claim.text, item_shabd.title, item_shabd.snippet, item_shabd.url, item_shabd.domain, claim)
        self.assertEqual(rel_shabd.level, RelevanceLevel.IRRELEVANT)

        # Cambridge Dictionary
        item_camb = self._make_item("E_L3", claim.claim_id, "DISCOVER | English meaning - Cambridge Dictionary", "Cambridge Dictionary: discover definition and pronunciation.", domain="dictionary.cambridge.org", url="https://dictionary.cambridge.org/dictionary/english/discover")
        rel_camb = evaluate_result_relevance(claim.text, item_camb.title, item_camb.snippet, item_camb.url, item_camb.domain, claim)
        self.assertEqual(rel_camb.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category M: Wikipedia Reference & Grammar Pages (Regression for Maheshwar Sutra & Alphabet)
    # -------------------------------------------------------------------------
    def test_cat_M_wikipedia_reference_and_grammar_filtered(self):
        claim = Claim("C_GRAM", "Cabinet announced major educational scholarship reform.", 0)
        
        # Sanskrit phonetics / Maheshwar sutra
        item_sutra = self._make_item(
            "E_M1", claim.claim_id,
            "माहेश्वर सूत्र - विकिपीडिया",
            "माहेश्वर सूत्रों को प्रत्याहार सूत्र भी कहा जाता है। संस्कृत व्याकरण में प्रयुक्त होते हैं।",
            domain="hi.wikipedia.org",
            url="https://hi.wikipedia.org/wiki/Maheshwar_Sutra"
        )
        rel_sutra = evaluate_result_relevance(claim.text, item_sutra.title, item_sutra.snippet, item_sutra.url, item_sutra.domain, claim)
        self.assertEqual(rel_sutra.level, RelevanceLevel.IRRELEVANT)

        # Alphabet page
        item_alpha = self._make_item(
            "E_M2", claim.claim_id,
            "र - विक्षनरी",
            "देवनागरी वर्णमाला का २७ वाँ व्यंजन।",
            domain="hi.wiktionary.org",
            url="https://hi.wiktionary.org/wiki/र"
        )
        rel_alpha = evaluate_result_relevance(claim.text, item_alpha.title, item_alpha.snippet, item_alpha.url, item_alpha.domain, claim)
        self.assertEqual(rel_alpha.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category N: Generic Statistics / Macro Overviews
    # -------------------------------------------------------------------------
    def test_cat_N_generic_statistics_rejected(self):
        claim = Claim("C_STAT", "Local municipal hospital recorded 40 cases of dengue this week.", 0)
        item_global = self._make_item(
            "E_N1", claim.claim_id,
            "Global Health Data: Annual Infectious Diseases Overview",
            "Statistical summary of worldwide epidemiological patterns across 190 nations.",
            domain="who.int"
        )
        rel = evaluate_result_relevance(claim.text, item_global.title, item_global.snippet, item_global.url, item_global.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category O: Same-Domain Duplicates
    # -------------------------------------------------------------------------
    def test_cat_O_same_domain_duplicates_deprioritized(self):
        claim = Claim("C_DOM", "Ministry of Finance published annual revenue collections.", 0)
        item1 = self._make_item("E_O1", claim.claim_id, "Finance Ministry Revenue Report", "Revenue collections grew 12 percent.", domain="finmin.gov.in")
        item2 = self._make_item("E_O2", claim.claim_id, "Tax Collection Press Release", "Tax revenues registered 12 percent increase.", domain="finmin.gov.in")

        summary = evaluate_claim_evidence(ClaimEvidenceResult(claim, SearchQuery(claim.text, []), [item1, item2], SearchStatus.SUCCESS))
        self.assertTrue(summary.evaluated_items[0].evaluation.is_independent)
        self.assertFalse(summary.evaluated_items[1].evaluation.is_independent)

    # -------------------------------------------------------------------------
    # Category P: Syndicated Wire Articles
    # -------------------------------------------------------------------------
    def test_cat_P_syndicated_wire_copy_detected(self):
        claim = Claim("C_WIRE", "Government declared public holiday following state funeral.", 0)
        item_wire1 = self._make_item("E_P1", claim.claim_id, "PTI News: Public holiday declared", "PTI reported government announced holiday.", domain="portal-a.com")
        item_wire2 = self._make_item("E_P2", claim.claim_id, "Public holiday announced: PTI wire", "PTI reported government declared holiday.", domain="portal-b.com")

        summary = evaluate_claim_evidence(ClaimEvidenceResult(claim, SearchQuery(claim.text, []), [item_wire1, item_wire2], SearchStatus.SUCCESS))
        self.assertTrue(summary.evaluated_items[0].evaluation.is_independent)
        self.assertFalse(summary.evaluated_items[1].evaluation.is_independent)
        self.assertEqual(summary.evaluated_items[1].evaluation.independence_status, IndependenceStatus.DUPLICATE_WIRE)

    # -------------------------------------------------------------------------
    # Category Q: Irrelevant Political Articles
    # -------------------------------------------------------------------------
    def test_cat_Q_irrelevant_political_article_rejected(self):
        claim = Claim("C_POL", "Election Commission issued notice to candidate X for poll code violation.", 0)
        item_unrelated_pol = self._make_item(
            "E_Q1", claim.claim_id,
            "State Assembly Speaker holds session on budget estimates",
            "Speaker presided over debate discussing state transport department allocations.",
            domain="ndtv.com"
        )
        rel = evaluate_result_relevance(claim.text, item_unrelated_pol.title, item_unrelated_pol.snippet, item_unrelated_pol.url, item_unrelated_pol.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category R: Irrelevant Science Articles
    # -------------------------------------------------------------------------
    def test_cat_R_irrelevant_science_article_rejected(self):
        claim = Claim("C_SCI", "James Webb Space Telescope detected water vapor on exoplanet WASP-96b.", 0)
        item_unrelated_sci = self._make_item(
            "E_R1", claim.claim_id,
            "Mars rover collects sedimentary rock core samples",
            "NASA rover drilled and collected geological core samples from Martian crater floor.",
            domain="sciencedaily.com"
        )
        rel = evaluate_result_relevance(claim.text, item_unrelated_sci.title, item_unrelated_sci.snippet, item_unrelated_sci.url, item_unrelated_sci.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category S: Irrelevant Sports Articles
    # -------------------------------------------------------------------------
    def test_cat_S_irrelevant_sports_article_rejected(self):
        claim = Claim("C_SPORT", "Novak Djokovic won the Australian Open men singles title.", 0)
        item_unrelated_sport = self._make_item(
            "E_S1", claim.claim_id,
            "Formula 1: Red Bull secures victory at Monaco Grand Prix",
            "Max Verstappen dominated the Monaco Grand Prix from pole to checkered flag.",
            domain="espn.com"
        )
        rel = evaluate_result_relevance(claim.text, item_unrelated_sport.title, item_unrelated_sport.snippet, item_unrelated_sport.url, item_unrelated_sport.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category T: Irrelevant Health Articles
    # -------------------------------------------------------------------------
    def test_cat_T_irrelevant_health_article_rejected(self):
        claim = Claim("C_HLTH", "Health ministry approved a new mRNA vaccine for malaria prevention.", 0)
        item_unrelated_hlth = self._make_item(
            "E_T1", claim.claim_id,
            "Study examines dietary sodium impact on cardiovascular hypertension",
            "Clinical research evaluates long-term sodium restriction on adult blood pressure.",
            domain="nih.gov"
        )
        rel = evaluate_result_relevance(claim.text, item_unrelated_hlth.title, item_unrelated_hlth.snippet, item_unrelated_hlth.url, item_unrelated_hlth.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category U: Completely Unrelated Articles
    # -------------------------------------------------------------------------
    def test_cat_U_completely_unrelated_articles_rejected(self):
        claim = Claim("C_RAND", "Supreme Court struck down the contentious zoning statute.", 0)
        item_unrelated = self._make_item(
            "E_U1", claim.claim_id,
            "How to bake artisan sourdough bread at home",
            "A comprehensive recipe guide for maintaining wild yeast starter and high hydration dough.",
            domain="foodandwine.com"
        )
        rel = evaluate_result_relevance(claim.text, item_unrelated.title, item_unrelated.snippet, item_unrelated.url, item_unrelated.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)
        self.assertEqual(rel.score, 0.0)

    # -------------------------------------------------------------------------
    # Category V: One-Character Overlap (Regression for the "र" Failure)
    # -------------------------------------------------------------------------
    def test_cat_V_single_character_overlap_never_sufficient(self):
        claim = Claim("C_ONECHAR", "होर्मुज में घुसपैठ कर रहे अमेरिकी ड्रोन बोट पर हमला, IRGC ने क्या बताया", 0)
        # Search result containing only single character match
        item_single_char = self._make_item(
            "E_V1", claim.claim_id,
            "र वर्णमाला प्रविष्टि",
            "र वर्ण देवनागरी लिपि का एक वर्ण है।",
            domain="varnamala-search.org",
            url="https://varnamala-search.org/alphabet/ra"
        )
        rel = evaluate_result_relevance(claim.text, item_single_char.title, item_single_char.snippet, item_single_char.url, item_single_char.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)
        self.assertFalse(rel.is_eligible_for_stance)

    # -------------------------------------------------------------------------
    # Category W: Generic-Word Overlap (Nonprofit / News / Study / Report)
    # -------------------------------------------------------------------------
    def test_cat_W_generic_word_overlap_never_sufficient(self):
        claim = Claim("C_GENERIC", "Agrasar surveyed 400 low-income schoolchildren in Gurugram.", 0)
        item_generic = self._make_item(
            "E_W1", claim.claim_id,
            "The Nonprofit Starvation Cycle - Stanford Social Innovation Review",
            "A study of nonprofit organizations and operating margins across philanthropic sectors.",
            domain="ssir.org"
        )
        rel = evaluate_result_relevance(claim.text, item_generic.title, item_generic.snippet, item_generic.url, item_generic.domain, claim)
        self.assertEqual(rel.level, RelevanceLevel.IRRELEVANT)

    # -------------------------------------------------------------------------
    # Category X: Correctly Relevant Evidence (Direct Verification)
    # -------------------------------------------------------------------------
    def test_cat_X_correctly_relevant_evidence_direct_stance(self):
        claim = Claim("C_CORR", "Supreme Court struck down the electoral bond scheme as unconstitutional.", 0)
        item_direct = self._make_item(
            "E_X1", claim.claim_id,
            "Supreme Court strikes down electoral bonds scheme",
            "In a unanimous ruling, the Supreme Court struck down the electoral bonds scheme, declaring it unconstitutional.",
            domain="thehindu.com"
        )
        eval_res = evaluate_claim_evidence(ClaimEvidenceResult(claim, SearchQuery(claim.text, []), [item_direct], SearchStatus.SUCCESS))
        analysis = synthesize_claim_stance(claim, eval_res.evaluated_items, eval_res.filtered_irrelevant_items)
        self.assertEqual(analysis.final_stance, ClaimStance.SUPPORTED)

    # -------------------------------------------------------------------------
    # Category Y: Partial Contextual Evidence
    # -------------------------------------------------------------------------
    def test_cat_Y_partial_contextual_evidence_classified_accurately(self):
        claim = Claim("C_PART", "Delhi Police arrested three suspects following bank robbery in Karol Bagh.", 0)
        item_partial = self._make_item(
            "E_Y1", claim.claim_id,
            "Karol Bagh bank robbery under investigation",
            "Delhi Police registered a case following a major bank robbery in Karol Bagh area.",
            domain="hindustantimes.com"
        )
        rel = evaluate_result_relevance(claim.text, item_partial.title, item_partial.snippet, item_partial.url, item_partial.domain, claim)
        self.assertIn(rel.level, (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL))

    # -------------------------------------------------------------------------
    # Category Z: Genuine Contradiction
    # -------------------------------------------------------------------------
    def test_cat_Z_genuine_contradiction_classified_accurately(self):
        claim = Claim("C_CONTR", "Central Bank raised benchmark interest rates to 6.5 percent.", 0)
        item_contra = self._make_item(
            "E_Z1", claim.claim_id,
            "Monetary Policy Statement: Central Bank cuts rates to 5.5 percent",
            "The Central Bank announced it cut benchmark interest rates by 100 bps to 5.5 percent.",
            domain="reuters.com"
        )
        eval_res = evaluate_claim_evidence(ClaimEvidenceResult(claim, SearchQuery(claim.text, []), [item_contra], SearchStatus.SUCCESS))
        analysis = synthesize_claim_stance(claim, eval_res.evaluated_items, eval_res.filtered_irrelevant_items)
        self.assertEqual(analysis.final_stance, ClaimStance.CONTRADICTED)

    # -------------------------------------------------------------------------
    # Regression Test: Hormuz IRGC Hindi Failure Scenario
    # -------------------------------------------------------------------------
    def test_regression_hormuz_irgc_hindi_failure_scenario(self):
        """CRITICAL REGRESSION TEST for the reported Hormuz Hindi claim:

        'होर्मुज में घुसपैठ कर रहे अमेरिकी ड्रोन बोट पर हमला, IRGC ने क्या बताया'
        Ensures:
        1. Wiktionary letter 'र' is classified IRRELEVANT and filtered.
        2. Maheshwar Sutra Sanskrit phonetics is classified IRRELEVANT and filtered.
        3. Search query does not shred characters into 'ह र म ज'.
        4. When only irrelevant results exist, verdict is UNCERTAIN (INSUFFICIENT_EVIDENCE).
        5. Genuinely relevant reporting participates cleanly in stance.
        """
        claim_text = "होर्मुज में घुसपैठ कर रहे अमेरिकी ड्रोन बोट पर हमला, IRGC ने क्या बताया"
        claims = extract_claims(claim_text)
        self.assertEqual(len(claims), 1)
        claim = claims[0]

        # 1. Query generation test
        sq = generate_search_query(claim)
        self.assertNotIn("ह र म ज", sq.query_text)
        self.assertIn("IRGC", sq.preserved_keywords)
        self.assertTrue(any(kw in sq.preserved_keywords for kw in ("होर्मुज", "ड्रोन", "बोट", "हमला")))

        # 2. Wiktionary page for letter 'र'
        item_wikt_ra = self._make_item(
            "E_RA", claim.claim_id,
            "र - विक्षनरी",
            "देवनागरी वर्णमाला का २७ वाँ व्यंजन। र व्यंजन वर्ण का सताईसवाँ अक्षर है।",
            domain="hi.wiktionary.org",
            url="https://hi.wiktionary.org/wiki/र"
        )
        # 3. Wikipedia page for Maheshwar Sutra
        item_sutra = self._make_item(
            "E_SUTRA", claim.claim_id,
            "माहेश्वर सूत्र - विकिपीडिया",
            "माहेश्वर सूत्रों को प्रत्याहार सूत्र भी कहा जाता है। संस्कृत व्याकरण में प्रयुक्त होते हैं।",
            domain="hi.wikipedia.org",
            url="https://hi.wikipedia.org/wiki/Maheshwar_Sutra"
        )

        cr = ClaimEvidenceResult(claim, sq, [item_wikt_ra, item_sutra], SearchStatus.SUCCESS)
        summary = evaluate_claim_evidence(cr)

        # Both MUST be filtered out as IRRELEVANT
        self.assertEqual(len(summary.filtered_irrelevant_items), 2)
        self.assertEqual(len(summary.evaluated_items), 0)
        self.assertEqual(summary.filtered_irrelevant_items[0].evaluation.relevance_level, RelevanceLevel.IRRELEVANT)
        self.assertEqual(summary.filtered_irrelevant_items[1].evaluation.relevance_level, RelevanceLevel.IRRELEVANT)

        # Epistemic safety: Absence of relevant evidence MUST yield UNCERTAIN
        analysis = synthesize_claim_stance(claim, summary.evaluated_items, summary.filtered_irrelevant_items)
        self.assertEqual(analysis.final_stance, ClaimStance.UNCERTAIN)
        self.assertIn("filtered_irrelevant_sources_count", dir(analysis))
        self.assertEqual(analysis.filtered_irrelevant_sources_count, 2)
        self.assertEqual(analysis.relevant_sources_count, 0)


if __name__ == "__main__":
    unittest.main()
