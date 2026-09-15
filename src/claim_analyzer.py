"""Claim stance analysis and explainable verdict generation engine (Phase 7).

Compares extracted factual claims against evaluated external evidence to determine
claim-level stances (SUPPORTED, CONTRADICTED, UNCERTAIN) and aggregate overall article
assessments (LIKELY_SUPPORTED, LIKELY_CONTRADICTED, MIXED, INSUFFICIENT_EVIDENCE).

Core Verification Guardrails:
1. No binary assumptions: absence of evidence or conflicting evidence yields UNCERTAIN.
2. Compares factual metrics, dates, and action polarities rather than relying on word overlap.
3. Weighs evidence by provenance and independence (Phase 6): primary sources carry high weight,
   opinion pieces cannot override factual reporting, duplicate wire copies do not multiply corroboration.
4. Claim importance classification (HIGH, MEDIUM, LOW) drives the overall article assessment.
5. 100% traceable: every verdict links to specific Evidence IDs.
6. Zero hallucinated facts or artificial confidence percentages.
7. Zero political bias or ideological blacklisting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set
import re

from src.claim_extractor import Claim, HINDI_STOPWORDS
from src.evidence_retriever import QUERY_STOPWORDS
from src.source_evaluator import (
    SourceCategory,
    SourceEvaluation,
    EvaluatedEvidenceItem,
    ClaimEvaluationSummary,
    IndependenceStatus,
    RelevanceLevel,
    EvidenceStrength,
)


class ClaimImportance(str, Enum):
    """Importance level of a factual claim within the article."""

    HIGH = "HIGH"        # Central assertions, quantitative metrics, regulatory rulings, key events
    MEDIUM = "MEDIUM"    # Corroborating factual assertions with specific dates or named entities
    LOW = "LOW"          # Secondary background or peripheral contextual assertions


class EvidenceStance(str, Enum):
    """Relationship of an individual evidence item to a factual claim."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL_IRRELEVANT = "NEUTRAL_IRRELEVANT"


class ClaimStance(str, Enum):
    """Synthesized verdict for an individual factual claim."""

    SUPPORTED = "SUPPORTED"          # Reliable evidence directly substantiates the claim
    CONTRADICTED = "CONTRADICTED"    # Reliable evidence directly refutes the claim
    UNCERTAIN = "UNCERTAIN"          # Evidence is insufficient, ambiguous, conflicting, or absent


class OverallAssessment(str, Enum):
    """Aggregate assessment for the entire article or story."""

    LIKELY_SUPPORTED = "LIKELY_SUPPORTED"          # Major claims supported, no major claims contradicted
    LIKELY_CONTRADICTED = "LIKELY_CONTRADICTED"    # Major claims directly contradicted by reliable evidence
    MIXED = "MIXED"                                # Some important claims supported, others contradicted
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE" # Inadequate reliable evidence across claims


@dataclass
class EvidenceStanceAnalysis:
    """Detailed stance evaluation of an individual evidence source against a claim."""

    evidence_id: str
    stance: EvidenceStance
    weight: float                     # 0.0 to 1.0 (derived from quality score & independence)
    rationale: str
    matched_points: List[str] = field(default_factory=list)
    conflicting_points: List[str] = field(default_factory=list)
    relevance_level: RelevanceLevel = RelevanceLevel.DIRECT
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    independence_status: IndependenceStatus = IndependenceStatus.INDEPENDENT


@dataclass
class ClaimAnalysisResult:
    """Complete stance analysis and explainable verdict for an individual claim."""

    claim_id: str
    claim_text: str
    importance: ClaimImportance
    final_stance: ClaimStance
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    neutral_evidence_ids: List[str] = field(default_factory=list)
    filtered_irrelevant_evidence_ids: List[str] = field(default_factory=list)
    explanation: str = ""
    evidence_limitations: List[str] = field(default_factory=list)
    individual_analyses: List[EvidenceStanceAnalysis] = field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    strength_explanation: str = ""
    disputed_summary: Optional[str] = None
    unique_domains: List[str] = field(default_factory=list)
    primary_sources_count: int = 0
    independent_sources_count: int = 0
    syndicated_sources_count: int = 0
    opinion_sources_count: int = 0
    relevant_sources_count: int = 0
    filtered_irrelevant_sources_count: int = 0
    relevance_audit_notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArticleVerificationReport:
    """Aggregate verification report across all extracted claims."""

    overall_assessment: OverallAssessment
    summary_explanation: str
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    uncertain_claims: int
    high_importance_claims: int
    claim_results: List[ClaimAnalysisResult] = field(default_factory=list)
    key_limitations: List[str] = field(default_factory=list)
    total_unique_domains: int = 0
    total_independent_sources: int = 0
    total_primary_sources: int = 0
    total_syndicated_sources: int = 0
    total_opinion_sources: int = 0
    total_filtered_irrelevant_sources: int = 0


# Domain-agnostic directional verb mappings to detect predicate agreement vs contradiction
DIRECTIONAL_VERB_MAP: Dict[str, Set[str]] = {
    "INCREASE": {"increase", "increased", "increasing", "raise", "raised", "raising", "hike", "hiked", "hiking", "elevated", "rose", "risen", "grew", "surged", "jumped", "up", "बढ़ा", "वृद्धि", "इजाफा", "तेजी", "बढ़ोतरी"},
    "DECREASE": {"decrease", "decreased", "decreasing", "cut", "cutting", "slash", "slashed", "slashing", "reduce", "reduced", "reducing", "lowered", "fell", "dropped", "slumped", "down", "घटा", "गिरावट", "कमी", "कटौती"},
    "UNCHANGED": {"unchanged", "steady", "hold", "held", "pause", "paused", "maintained", "status quo", "flat", "stagnant", "keep", "kept", "retained"},
    "APPROVE": {"approve", "approved", "approving", "pass", "passed", "passing", "sanction", "sanctioned", "ratified", "authorized", "cleared", "greenlighted", "adopted", "adopt", "adopting", "operationalized", "operationalize", "मंजूरी", "स्वीकार", "अनुमोदन"},
    "REJECT": {"reject", "rejected", "rejecting", "deny", "denied", "strike down", "struck down", "ban", "banned", "veto", "vetoed", "dismissed", "blocked", "खारिज", "अस्वीकार", "रद्द", "इनकार", "प्रतिबंध"},
    "LAUNCH": {"launch", "launched", "launching", "start", "started", "initiate", "initiated", "unveil", "unveiled", "introduced", "announced", "kicked", "kicks", "commenced", "commence", "शुरू", "प्रक्षेपण", "लॉन्च", "उद्घाटन"},
    "DELAY": {"delay", "delayed", "delaying", "postpone", "postponed", "cancel", "cancelled", "halt", "halted", "suspended", "called off"},
    "WIN": {"win", "won", "winning", "triumph", "triumphed", "victorious", "beat", "जीत", "जीता"},
    "LOSE": {"lose", "lost", "losing", "defeat", "defeated", "concede", "conceded", "fail", "failed", "eliminated", "beaten", "हार", "हारा"},
    "ARREST": {"arrest", "arrested", "arresting", "detain", "detained", "detaining", "nabbed", "apprehended", "jailed", "custody", "गिरफ्तार", "हिरासत", "कैद"},
    "FREE": {"release", "released", "releasing", "free", "freed", "acquitted", "cleared", "discharged", "exonerated", "bailed", "रिहा", "बरी"},
    "JOIN": {"join", "joined", "joining", "accede", "acceded", "accession", "member", "admitted", "enter", "entered", "शामिल"},
    "ISSUE": {"issue", "issued", "issuing", "publish", "published", "publishing", "notify", "notified", "bulletin", "release", "released", "जारी"},
    "DISCOVER": {"discover", "discovered", "discovering", "find", "found", "observe", "observed", "detect", "detected", "खोज", "पता"},
    "SIGN": {"sign", "signed", "signing", "ink", "inked", "conclude", "concluded", "finalize", "finalized", "हस्ताक्षर"},
}

# Explicit pairs of mutually opposing directions
OPPOSITE_DIRECTIONS: Set[Tuple[str, str]] = {
    ("INCREASE", "DECREASE"), ("DECREASE", "INCREASE"),
    ("UNCHANGED", "INCREASE"), ("UNCHANGED", "DECREASE"),
    ("INCREASE", "UNCHANGED"), ("DECREASE", "UNCHANGED"),
    ("APPROVE", "REJECT"), ("REJECT", "APPROVE"),
    ("LAUNCH", "DELAY"), ("DELAY", "LAUNCH"),
    ("WIN", "LOSE"), ("LOSE", "WIN"),
    ("ARREST", "FREE"), ("FREE", "ARREST"),
}

# Action and predicate synonym groups for domain-agnostic semantic alignment
ACTION_SYNONYM_GROUPS: List[Set[str]] = [
    {"start", "started", "starting", "launch", "launched", "launching", "initiate", "initiated", "initiating", "commenced", "commence", "kicked", "kicks", "शुरू"},
    {"approve", "approved", "approving", "pass", "passed", "passing", "adopt", "adopted", "adopting", "ratify", "ratified", "authorize", "authorized", "operationalize", "operationalized", "मंजूरी"},
    {"join", "joined", "joining", "accede", "acceded", "accession", "enter", "entered", "member", "admitted", "शामिल"},
    {"issue", "issued", "issuing", "publish", "published", "publishing", "release", "released", "releasing", "notify", "notified", "जारी"},
    {"arrest", "arrested", "arresting", "detain", "detained", "detaining", "apprehend", "apprehended", "custody", "गिरफ्तार", "हिरासत"},
    {"sign", "signed", "signing", "ink", "inked", "conclude", "concluded", "finalize", "finalized", "हस्ताक्षर"},
    {"find", "found", "finding", "observe", "observed", "observing", "discover", "discovered", "discovering", "detect", "detected", "detecting"},
    {"increase", "increased", "increasing", "raise", "raised", "raising", "hike", "hiked", "rose", "risen", "grew", "surged", "jumped", "वृद्धि"},
    {"decrease", "decreased", "decreasing", "cut", "cutting", "slash", "slashed", "reduce", "reduced", "lowered", "fell", "dropped", "slumped", "कमी"},
    {"hold", "held", "holding", "meet", "met", "meeting", "convene", "convened", "gather", "gathered"},
    {"ban", "banned", "banning", "prohibit", "prohibited", "prohibiting", "restrict", "restricted", "bar", "barred", "प्रतिबंध"},
]

# Domain-agnostic refutation and denial patterns
REFUTATION_MARKERS: List[str] = [
    "entirely false", "not true", "denied reports", "dismissed reports",
    "fake claim", "false claim", "misleading reports", "denies reports",
    "rejected claim", "no truth", "untrue", "debunked", "fabricated",
    "hoax", "fact check finds false", "refuted claims", "unfounded",
    "no evidence of", "baseless", "categorically denied",
    "झूठा दावा", "भ्रामक दावा", "खंडन किया", "गलत दावा", "खारिज किया", "फेक न्यूज", "इनकार किया"
]

# Negation indicators
NEGATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:not|no|never|neither|nor|cannot|can't|didn't|did not|won't|will not|wasn't|was not|weren't|were not|hasn't|has not|haven't|have not|isn't|is not|aren't|are not)\b", re.IGNORECASE),
    re.compile(r"\b(?:denied|refused|failed to|rejected|dismissed|refuted|debunked|prevented)\b", re.IGNORECASE),
    re.compile(r"(?:\bनहीं\b|\bना\b|\bइनकार\b|\bखारिज\b|\bगलत\b|\bझूठ\b)"),
]

# Attribution patterns (statements reported as allegations/claims vs direct events)
ATTRIBUTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:according to|alleged|claimed|stated|said|reported by|spokesperson said|official said|police said|leaders alleged|minister stated|sources said|cited as saying)\b", re.IGNORECASE),
]


@dataclass
class GenericClaimRepresentation:
    """Domain-agnostic structured factual representation of a claim or evidence statement."""

    raw_text: str
    entities: Set[str] = field(default_factory=set)        # Significant nouns/named entities
    action_direction: Optional[str] = None                 # Directional action type (INCREASE, LAUNCH, etc.)
    action_directions: Set[str] = field(default_factory=set) # All detected directional actions
    action_tokens: Set[str] = field(default_factory=set)   # Significant verbs/action words
    numbers: Dict[float, str] = field(default_factory=dict) # Numeric values & units (e.g. 5.0 -> '5%')
    dates_years: Set[str] = field(default_factory=set)     # Extracted 4-digit years or month names
    is_negated: bool = False                               # True if statement contains negation polarity
    attribution: Optional[str] = None                      # Extracted reporting attribution (e.g. 'police said')


def detect_negation(text: str) -> bool:
    """Detects whether text contains grammatical or lexical negation."""
    for pattern in NEGATION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def extract_attribution(text: str) -> Optional[str]:
    """Extracts attribution framing if present (e.g. 'alleged', 'according to official')."""
    for pattern in ATTRIBUTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip().lower()
    return None


def extract_dates_and_times(text: str) -> Set[str]:
    """Extracts 4-digit years and prominent calendar terms."""
    matches = set(re.findall(r"\b(?:19\d\d|20\d\d)\b", text))
    months = re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.IGNORECASE)
    for m in months:
        matches.add(m.lower())
    return matches


def parse_number_value(token: str) -> Optional[float]:
    """Extracts float value from a numeric or percentage token (e.g. '6.5%' -> 6.5)."""
    clean = token.strip("%,$€£₹ ")
    try:
        return float(clean)
    except Exception:
        return None


def extract_numbers_and_percentages(text: str) -> Dict[float, str]:
    """Extracts numbers and percentages, mapping numeric float value to raw token string."""
    matches = re.findall(r"(?:\$|€|£|₹)?\b\d+(?:\.\d+)?%?", text)
    result: Dict[float, str] = {}
    for m in matches:
        val = parse_number_value(m)
        if val is not None:
            result[val] = m
    return result


def detect_action_directions(text: str) -> Set[str]:
    """Detects all directional action types from verbs in text across generic categories."""
    tokens = set(re.findall(r"[\u0900-\u097F\w]+", text.lower()))
    found: Set[str] = set()
    for direction, verbs in DIRECTIONAL_VERB_MAP.items():
        if tokens & verbs:
            found.add(direction)
    return found


def detect_action_direction(text: str) -> Optional[str]:
    """Detects primary directional action type from verbs in text across generic categories."""
    dirs = detect_action_directions(text)
    return next(iter(dirs), None) if dirs else None


def extract_generic_representation(text: str) -> GenericClaimRepresentation:
    """Extracts a domain-agnostic structured representation from any text snippet."""
    text_lower = text.lower()
    
    # 1. Action direction & action tokens
    directions = detect_action_directions(text_lower)
    primary_direction = next(iter(directions), None) if directions else None
    all_action_verbs = set().union(*DIRECTIONAL_VERB_MAP.values())
    raw_words = set(re.findall(r"[\u0900-\u097F\w]+", text_lower))
    action_tokens = raw_words & all_action_verbs
    
    # Also collect generic past-participle / verb tokens (ends with -ed, -ing, or common irregulars)
    for w in raw_words:
        if len(w) >= 4 and (w.endswith("ed") or w.endswith("ing")):
            if w not in QUERY_STOPWORDS:
                action_tokens.add(w)

    # Hindi action vocabulary
    HINDI_ACTION_WORDS = {
        "हमला", "घुसपैठ", "गिरफ्तार", "हिरासत", "घोषणा", "दावा", "खारिज", "पुष्टि",
        "प्रतिबंध", "आदेश", "फैसला", "शुरू", "हस्ताक्षर", "दर्ज", "जांच", "मंजूरी",
        "रद्द", "हताहत", "मौत", "बताया", "कहा"
    }
    for haw in HINDI_ACTION_WORDS:
        if haw in text_lower:
            action_tokens.add(haw)

    # 2. Significant entities (nouns, capitalized words, acronyms, substantive words across scripts)
    tokens = re.findall(r"[\u0900-\u097F\w\-\_]{2,}", text)
    entities: Set[str] = set()
    for tok in tokens:
        low = tok.lower()
        if low.isdigit():
            continue
        is_dev = any('\u0900' <= c <= '\u097F' for c in tok)
        if is_dev:
            if low not in HINDI_STOPWORDS and len(low) >= 2 and low not in HINDI_ACTION_WORDS:
                entities.add(low)
        else:
            if low not in QUERY_STOPWORDS and len(low) >= 3 and low not in all_action_verbs:
                entities.add(low)

    # 3. Numbers & Percentages
    numbers = extract_numbers_and_percentages(text)

    # 4. Dates & Years
    dates_years = extract_dates_and_times(text)

    # 5. Negation Polarity
    is_negated = detect_negation(text)

    # 6. Attribution
    attrib = extract_attribution(text)

    return GenericClaimRepresentation(
        raw_text=text,
        entities=entities,
        action_direction=primary_direction,
        action_directions=directions,
        action_tokens=action_tokens,
        numbers=numbers,
        dates_years=dates_years,
        is_negated=is_negated,
        attribution=attrib,
    )


def determine_claim_importance(claim: Claim) -> ClaimImportance:
    """Classifies claim importance (HIGH, MEDIUM, LOW) based on domain-agnostic factual density."""
    text = claim.text

    # High importance: specific numbers, percentages, currencies, or institutional rulings/official actions
    has_metrics = bool(re.search(r"\b\d+(?:\.\d+)?%\b", text) or any(c in text for c in "$€£₹"))
    has_specific_numbers = bool(re.search(r"\b\d{2,}\b", text))  # Numbers with 2+ digits (not just "a" or "1")

    # Use a stricter set of institutional action verbs (excluding generic "reported", "announced")
    has_official_action = bool(re.search(
        r"\b(?:ordered|banned|approved|rejected|arrested|sentenced|convicted|acquitted|launched|signed|ratified|enacted|repealed|impeached|indicted|declared|recalled|sanctioned|fined|suspended)\b",
        text,
        re.IGNORECASE,
    ))

    # HIGH requires both specific quantitative data AND institutional action
    if (has_metrics or has_specific_numbers) and has_official_action:
        return ClaimImportance.HIGH

    # MEDIUM: has either quantitative data OR institutional action, or both general metrics and generic actions
    has_general_action = bool(re.search(
        r"\b(?:announced|increased|decreased|reported|confirmed|court|ministry|agency|bank|government|parliament|police)\b",
        text,
        re.IGNORECASE,
    ))

    if has_official_action or (has_metrics and has_general_action) or (has_specific_numbers and has_general_action):
        return ClaimImportance.MEDIUM

    # Medium importance: specific dates, years, or named entities
    has_dates = bool(re.search(r"\b(?:19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.IGNORECASE))
    words = text.split()
    capitalized_words = [w for w in words[1:] if w and w[0].isupper() and w.isalpha()]

    if has_dates or len(capitalized_words) >= 2:
        return ClaimImportance.MEDIUM

    return ClaimImportance.LOW


def analyze_evidence_against_claim(
    claim: Claim,
    evaluated_item: EvaluatedEvidenceItem,
) -> EvidenceStanceAnalysis:
    """Compares an individual evidence item against the claim's factual substance in a domain-agnostic manner."""
    evidence = evaluated_item.evidence
    eval_data = evaluated_item.evaluation
    evidence_text = f"{evidence.title}. {evidence.snippet}"
    evidence_text_lower = evidence_text.lower()
    claim_text_lower = claim.text.lower()

    # Weight factor: derived from quality score, penalized for opinion pieces or syndicated duplication
    weight_factor = eval_data.quality_score / 100.0
    if eval_data.is_opinion:
        weight_factor *= 0.25
    if not eval_data.is_independent:
        weight_factor *= 0.5  # Duplicate wire copy has reduced marginal weight

    matched_points: List[str] = []
    conflicting_points: List[str] = []

    # Domain-agnostic generic representations
    claim_rep = extract_generic_representation(claim.text)
    evidence_rep = extract_generic_representation(evidence_text)

    # 0. Relevance Gate Check: IRRELEVANT sources MUST NOT participate in stance analysis!
    if eval_data.relevance_level == RelevanceLevel.IRRELEVANT or not eval_data.is_relevant:
        return EvidenceStanceAnalysis(
            evidence_id=evidence.evidence_id,
            stance=EvidenceStance.NEUTRAL_IRRELEVANT,
            weight=0.0,
            rationale=f"Filtered at Relevance Gate: {eval_data.demotion_reason or 'Source does not address claim entity or subject.'}",
            matched_points=[],
            conflicting_points=[],
            relevance_level=RelevanceLevel.IRRELEVANT,
            evidence_strength=EvidenceStrength.INSUFFICIENT,
            independence_status=eval_data.independence_status,
        )

    # 1. Topical / Entity Relevance Guardrail
    # Compute overlap between substantive entities
    shared_entities = claim_rep.entities & evidence_rep.entities
    entity_overlap_ratio = len(shared_entities) / max(1, len(claim_rep.entities))

    # Basic relevance threshold: must share key entities
    is_topically_relevant = len(shared_entities) >= min(2, len(claim_rep.entities)) or bool(
        claim_rep.entities & {"rbi", "gdp", "nasa", "who", "un", "isro", "imf", "eu", "bcci", "fifa"}
    ) or (eval_data.relevance_level in (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL))

    if not is_topically_relevant or eval_data.relevance_level == RelevanceLevel.WEAK:
        return EvidenceStanceAnalysis(
            evidence_id=evidence.evidence_id,
            stance=EvidenceStance.NEUTRAL_IRRELEVANT,
            weight=0.0,
            rationale="Source content does not contain sufficient topical or entity overlap with the claim.",
            matched_points=[],
            conflicting_points=[],
            relevance_level=RelevanceLevel.WEAK,
            evidence_strength=EvidenceStrength.WEAK,
            independence_status=eval_data.independence_status,
        )

    # 2. Check for Explicit Official Refutations or Denials in Evidence
    has_refutation = any(marker in evidence_text_lower for marker in REFUTATION_MARKERS)
    if has_refutation and is_topically_relevant:
        conflicting_points.append("Source contains explicit refutation, denial, or debunking of the reported assertion.")

    # 3. Negation & Polarity Analysis
    # If one asserts the positive and the other asserts the negation (with substantive entity overlap)
    polarity_conflict = False
    if claim_rep.is_negated != evidence_rep.is_negated:
        # One is negated, one is affirmative
        # Check if they share predicate or key action/entities
        shared_actions = claim_rep.action_tokens & evidence_rep.action_tokens
        claim_dirs_chk = claim_rep.action_directions or ({claim_rep.action_direction} if claim_rep.action_direction else set())
        ev_dirs_chk = evidence_rep.action_directions or ({evidence_rep.action_direction} if evidence_rep.action_direction else set())
        shared_dirs_chk = claim_dirs_chk & ev_dirs_chk
        has_synonym_act = any((claim_rep.action_tokens & grp) and (evidence_rep.action_tokens & grp) for grp in ACTION_SYNONYM_GROUPS)

        if shared_actions or shared_dirs_chk or has_synonym_act or (len(shared_entities) >= 2):
            polarity_conflict = True
            claim_pol = "negated" if claim_rep.is_negated else "affirmative"
            ev_pol = "negated" if evidence_rep.is_negated else "affirmative"
            conflicting_points.append(
                f"Polarity conflict: Claim is framed as {claim_pol}, whereas source states the {ev_pol} event."
            )

    # 4. Action Direction / Predicate Comparison
    direction_conflict = False
    direction_support = False

    claim_dirs = claim_rep.action_directions or ({claim_rep.action_direction} if claim_rep.action_direction else set())
    ev_dirs = evidence_rep.action_directions or ({evidence_rep.action_direction} if evidence_rep.action_direction else set())

    # Check for direct conflicts in directions
    for cdir in claim_dirs:
        for edir in ev_dirs:
            if (cdir, edir) in OPPOSITE_DIRECTIONS:
                direction_conflict = True
                conflicting_points.append(
                    f"Action polarity conflict: Claim indicates '{cdir}', but source reports '{edir}'."
                )
                break
        if direction_conflict:
            break

    # Check for direction agreement
    shared_dirs = claim_dirs & ev_dirs
    if shared_dirs:
        direction_support = True
        matched_points.append(f"Consistent action/predicate: '{', '.join(sorted(shared_dirs))}'.")

    # 5. Numerical / Quantity Comparison (False Contradiction Prevention)
    numerical_conflict = False
    numerical_support = False
    shared_actions = claim_rep.action_tokens & evidence_rep.action_tokens
    has_action_match = bool(shared_actions or direction_support)
    has_numerical_context_match = bool(
        has_action_match
        or (len(shared_entities) >= 2)
        or bool(shared_entities & {
            "gdp", "growth", "inflation", "rate", "unemployment", "revenue", "profit",
            "deficit", "efficacy", "mortality", "poverty", "tax", "cases", "deaths",
            "toll", "score", "temperature", "poll", "votes", "share", "budget"
        })
    )

    if claim_rep.numbers:
        for val, token in claim_rep.numbers.items():
            if val in evidence_rep.numbers:
                numerical_support = True
                matched_points.append(f"Matching numerical value: {token} (found {evidence_rep.numbers[val]})")
            else:
                # Discrepancy check: ONLY compare numbers if the source is DIRECT/PARTIAL relevance
                # AND shares the specific action/predicate or topic metric!
                # An unrelated percentage in another topic must NEVER contradict this claim.
                if is_topically_relevant and has_numerical_context_match and eval_data.relevance_level in (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL):
                    if "%" in token:
                        diff_percents = [raw for fval, raw in evidence_rep.numbers.items() if "%" in raw and abs(fval - val) > 0.001]
                        if diff_percents:
                            numerical_conflict = True
                            conflicting_points.append(
                                f"Numerical conflict: Claim mentions {token}, but evidence reports {', '.join(diff_percents)}."
                            )
                    else:
                        diff_counts = [raw for fval, raw in evidence_rep.numbers.items() if "%" not in raw and abs(fval - val) >= 1.0]
                        if diff_counts:
                            numerical_conflict = True
                            conflicting_points.append(
                                f"Quantitative discrepancy: Claim states {token}, but source specifies {', '.join(diff_counts)}."
                            )

    # 6. Temporal / Date Reasoning
    # If claim specifies a distinct year (e.g. 2026) and evidence discusses a different historical year (e.g. 2013)
    temporal_mismatch_points: List[str] = []
    claim_years = {y for y in claim_rep.dates_years if y.isdigit()}
    ev_years = {y for y in evidence_rep.dates_years if y.isdigit()}
    if claim_years and ev_years and not (claim_years & ev_years):
        temporal_mismatch_points.append(
            f"Temporal mismatch: Claim refers to {', '.join(sorted(claim_years))}, but source references event from {', '.join(sorted(ev_years))}."
        )

    # 7. Action / Predicate Coverage
    # Shared action tokens across any domain (already computed in shared_actions and has_action_match)

    # 8. Attribution Handling
    # If evidence is merely reporting an unverified allegation ("Person A alleged that X happened")
    attribution_note = ""
    if evidence_rep.attribution and not claim_rep.attribution:
        attribution_note = f"Source reports assertion as an attributed statement or allegation ({evidence_rep.attribution}), not direct confirmation."

    # 9. Synthesize Individual Evidence Stance
    has_numerical_req = bool(claim_rep.numbers)
    can_support_numerics = not has_numerical_req or numerical_support

    if temporal_mismatch_points:
        stance = EvidenceStance.NEUTRAL_IRRELEVANT
        rationale = f"Source is from a different time period ({'; '.join(temporal_mismatch_points)}) and cannot verify current claim."
    elif conflicting_points or direction_conflict or polarity_conflict or (has_refutation and is_topically_relevant):
        stance = EvidenceStance.CONTRADICTS
        rationale = f"Source contradicts the claim: {'; '.join(conflicting_points)}"
    elif (numerical_support or (direction_support and can_support_numerics and (has_action_match or not claim_rep.action_tokens))) and is_topically_relevant:
        stance = EvidenceStance.SUPPORTS
        rationale = f"Source supports the claim: {'; '.join(matched_points)}"
        if attribution_note:
            rationale += f" ({attribution_note})"
    elif is_topically_relevant and shared_actions and len(shared_entities) >= 2 and can_support_numerics and not conflicting_points:
        stance = EvidenceStance.SUPPORTS
        rationale = f"Source substantively corroborates key context and assertions ({', '.join(list(shared_entities)[:3])})."
        if attribution_note:
            rationale += f" ({attribution_note})"
    elif is_topically_relevant and not has_action_match and claim_rep.action_tokens:
        # Shared entity but missing the key predicate/action (e.g. "Company X announced product" vs "Company X quarterly results")
        stance = EvidenceStance.NEUTRAL_IRRELEVANT
        rationale = (
            f"Source discusses related subject/entities ({', '.join(list(shared_entities)[:3])}), "
            "but does not corroborate the specific action or event asserted in the claim."
        )
    else:
        stance = EvidenceStance.NEUTRAL_IRRELEVANT
        rationale = "Source is topically related but does not provide direct corroboration or refutation of the specific action."

    return EvidenceStanceAnalysis(
        evidence_id=evidence.evidence_id,
        stance=stance,
        weight=weight_factor,
        rationale=rationale,
        matched_points=matched_points,
        conflicting_points=conflicting_points,
        relevance_level=eval_data.relevance_level,
        evidence_strength=eval_data.evidence_strength,
        independence_status=eval_data.independence_status,
    )


def synthesize_claim_stance(
    claim: Claim,
    evaluated_items: List[EvaluatedEvidenceItem],
    filtered_items: Optional[List[EvaluatedEvidenceItem]] = None,
) -> ClaimAnalysisResult:
    """Synthesizes all individual evidence evaluations into an explainable claim verdict.

    Enforces strict False Contradiction Prevention:
    - Irrelevant or weak sources never become supporting or contradicting evidence.
    - Contradictions are recognized only when credible relevant sources materially conflict
      on the same factual predicate.
    - If all retrieved evidence is irrelevant, outputs UNCERTAIN / INSUFFICIENT_EVIDENCE
      without manufacturing false disputes.
    """
    importance = determine_claim_importance(claim)
    filtered_list = filtered_items or []
    filtered_ids = [it.evidence.evidence_id for it in filtered_list]

    # Collect audit notes for all sources
    audit_notes: Dict[str, str] = {}
    for it in evaluated_items:
        if it.evaluation.relevance_assessment and it.evaluation.relevance_assessment.explanation:
            audit_notes[it.evidence.evidence_id] = it.evaluation.relevance_assessment.explanation
        else:
            audit_notes[it.evidence.evidence_id] = f"Relevant ({it.evaluation.relevance_level.value})"

    for it in filtered_list:
        audit_notes[it.evidence.evidence_id] = it.evaluation.demotion_reason or "Filtered at Relevance Gate (Irrelevant)"

    if not evaluated_items and not filtered_list:
        return ClaimAnalysisResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            importance=importance,
            final_stance=ClaimStance.UNCERTAIN,
            supporting_evidence_ids=[],
            contradicting_evidence_ids=[],
            neutral_evidence_ids=[],
            filtered_irrelevant_evidence_ids=[],
            explanation="No external evidence was retrieved. The system cannot establish whether the claim is true or false.",
            evidence_limitations=["Zero search results retrieved from configured search provider."],
            individual_analyses=[],
            evidence_strength=EvidenceStrength.INSUFFICIENT,
            strength_explanation="Zero search results retrieved from configured search provider.",
            disputed_summary=None,
            unique_domains=[],
            primary_sources_count=0,
            independent_sources_count=0,
            syndicated_sources_count=0,
            opinion_sources_count=0,
            relevant_sources_count=0,
            filtered_irrelevant_sources_count=0,
            relevance_audit_notes={},
        )

    individual_analyses: List[EvidenceStanceAnalysis] = []
    support_ids: List[str] = []
    contradict_ids: List[str] = []
    neutral_ids: List[str] = []

    total_support_weight = 0.0
    total_contradict_weight = 0.0

    has_primary_support = False
    has_primary_contradict = False

    for item in evaluated_items:
        analysis = analyze_evidence_against_claim(claim, item)
        individual_analyses.append(analysis)

        # Relevance Gate Guardrail: Only DIRECT or PARTIAL relevant sources can support or contradict
        is_source_relevant = (
            item.evaluation.is_relevant
            and item.evaluation.relevance_level in (RelevanceLevel.DIRECT, RelevanceLevel.PARTIAL)
        )

        if analysis.stance == EvidenceStance.SUPPORTS and is_source_relevant:
            support_ids.append(item.evidence.evidence_id)
            total_support_weight += analysis.weight
            if item.evaluation.is_primary:
                has_primary_support = True
        elif analysis.stance == EvidenceStance.CONTRADICTS and is_source_relevant:
            contradict_ids.append(item.evidence.evidence_id)
            total_contradict_weight += analysis.weight
            if item.evaluation.is_primary:
                has_primary_contradict = True
        else:
            neutral_ids.append(item.evidence.evidence_id)

    limitations: List[str] = [
        "Evaluation is based on retrieved search snippets and metadata. Full document auditing may provide additional context."
    ]

    disputed_summary: Optional[str] = None

    # Stance synthesis rules
    # Case 1: Conflicting evidence from credible relevant sources (Genuine Conflict)
    if support_ids and contradict_ids:
        # Check if primary source decisively breaks the tie
        if has_primary_support and not has_primary_contradict and total_support_weight > (total_contradict_weight * 2):
            final_stance = ClaimStance.SUPPORTED
            explanation = (
                f"Although conflicting mentions were noted in {', '.join(contradict_ids)}, "
                f"official primary evidence ({', '.join(support_ids)}) authoritatively supports the claim."
            )
        elif has_primary_contradict and not has_primary_support and total_contradict_weight > (total_support_weight * 2):
            final_stance = ClaimStance.CONTRADICTED
            explanation = (
                f"Although some reports supported the claim ({', '.join(support_ids)}), "
                f"official primary evidence ({', '.join(contradict_ids)}) directly refutes it."
            )
        else:
            final_stance = ClaimStance.UNCERTAIN
            explanation = (
                f"Credible sources provide conflicting evidence: {', '.join(support_ids)} support the claim, "
                f"while {', '.join(contradict_ids)} conflict with it. Available evidence is insufficient to resolve this conflict conclusively."
            )
            disputed_summary = f"Contested assertion: supported by {', '.join(support_ids)} but contradicted by {', '.join(contradict_ids)}."
            limitations.append("Conflicting reporting detected across external sources.")

    # Case 2: Uncontested support
    elif support_ids and not contradict_ids:
        if total_support_weight >= 0.5:
            final_stance = ClaimStance.SUPPORTED
            sources_summary = f"supported by reliable evidence ({', '.join(support_ids)})"
            if has_primary_support:
                sources_summary += " including official primary documentation"
            explanation = f"The claim's core assertions are directly {sources_summary}."
        else:
            final_stance = ClaimStance.UNCERTAIN
            explanation = (
                f"Weak topical support noted in {', '.join(support_ids)}, but evidence quality or depth is "
                "insufficient to substantiate the claim with high confidence."
            )
            limitations.append("Supporting sources had low factual weight or were limited to opinion/commentary.")

    # Case 3: Uncontested contradiction
    elif contradict_ids and not support_ids:
        if total_contradict_weight >= 0.5:
            final_stance = ClaimStance.CONTRADICTED
            sources_summary = f"directly refuted by reliable evidence ({', '.join(contradict_ids)})"
            if has_primary_contradict:
                sources_summary += " including official primary sources"
            explanation = f"The claim is {sources_summary}."
        else:
            final_stance = ClaimStance.UNCERTAIN
            explanation = (
                f"Potential conflict noted in {', '.join(contradict_ids)}, but the evidence is insufficient "
                "or too low-weight to firmly contradict the statement."
            )
            limitations.append("Contradicting sources had low factual weight.")

    # Case 4: Only neutral or irrelevant sources (No substantive evidence)
    else:
        final_stance = ClaimStance.UNCERTAIN
        if filtered_ids and not evaluated_items:
            explanation = (
                f"Search results did not contain sufficiently relevant evidence for this claim "
                f"({len(filtered_ids)} irrelevant result(s) filtered out). The system cannot verify or refute the statement."
            )
            limitations.append(f"External search results were filtered as irrelevant ({', '.join(filtered_ids)}).")
        else:
            explanation = (
                f"Available sources ({', '.join(neutral_ids) if neutral_ids else 'none'}) "
                "do not contain direct evidence to either confirm or refute the specific substance of the claim."
            )
            limitations.append("Retrieved sources were neutral, peripheral, or did not address the specific factual assertion.")

    # Synthesize claim-level evidence strength
    if final_stance == ClaimStance.SUPPORTED:
        if has_primary_support or any(a.evidence_strength == EvidenceStrength.STRONG for a in individual_analyses if a.stance == EvidenceStance.SUPPORTS):
            claim_strength = EvidenceStrength.STRONG
            strength_explanation = "Substantiated by strong independent or primary documentation directly addressing the assertion."
        elif total_support_weight >= 0.6:
            claim_strength = EvidenceStrength.MODERATE
            strength_explanation = "Substantiated by credible factual reporting with standard evidentiary depth."
        else:
            claim_strength = EvidenceStrength.WEAK
            strength_explanation = "Topical support present, but relying on lower-weight or syndicated sources."
    elif final_stance == ClaimStance.CONTRADICTED:
        if has_primary_contradict or any(a.evidence_strength == EvidenceStrength.STRONG for a in individual_analyses if a.stance == EvidenceStance.CONTRADICTS):
            claim_strength = EvidenceStrength.STRONG
            strength_explanation = "Directly refuted by authoritative primary or high-quality independent evidence."
        else:
            claim_strength = EvidenceStrength.MODERATE
            strength_explanation = "Refuted by external reporting."
    else:  # UNCERTAIN
        if support_ids and contradict_ids:
            claim_strength = EvidenceStrength.MODERATE
            strength_explanation = "Contested: multiple credible sources provide conflicting accounts."
        elif individual_analyses and any(a.stance != EvidenceStance.NEUTRAL_IRRELEVANT for a in individual_analyses):
            claim_strength = EvidenceStrength.WEAK
            strength_explanation = "Evidence is peripheral, conflicting, or lacks sufficient independent depth."
        else:
            claim_strength = EvidenceStrength.INSUFFICIENT
            strength_explanation = "Zero direct relevant evidence found to confirm or refute the assertion."

    # Domain metrics
    unique_domains = list({it.evidence.domain for it in evaluated_items if it.evidence.domain})
    primary_count = sum(1 for it in evaluated_items if it.evaluation.is_primary)
    independent_count = sum(1 for it in evaluated_items if it.evaluation.is_independent and not it.evaluation.is_opinion)
    syndicated_count = sum(1 for it in evaluated_items if it.evaluation.is_syndicated_wire)
    opinion_count = sum(1 for it in evaluated_items if it.evaluation.is_opinion)

    return ClaimAnalysisResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        importance=importance,
        final_stance=final_stance,
        supporting_evidence_ids=support_ids,
        contradicting_evidence_ids=contradict_ids,
        neutral_evidence_ids=neutral_ids,
        filtered_irrelevant_evidence_ids=filtered_ids,
        explanation=explanation,
        evidence_limitations=limitations,
        individual_analyses=individual_analyses,
        evidence_strength=claim_strength,
        strength_explanation=strength_explanation,
        disputed_summary=disputed_summary,
        unique_domains=unique_domains,
        primary_sources_count=primary_count,
        independent_sources_count=independent_count,
        syndicated_sources_count=syndicated_count,
        opinion_sources_count=opinion_count,
        relevant_sources_count=len(evaluated_items),
        filtered_irrelevant_sources_count=len(filtered_list),
        relevance_audit_notes=audit_notes,
    )


def generate_article_report(
    claim_results: List[ClaimAnalysisResult],
) -> ArticleVerificationReport:
    """Aggregates claim-level verdicts into an overall article assessment."""
    if not claim_results:
        return ArticleVerificationReport(
            overall_assessment=OverallAssessment.INSUFFICIENT_EVIDENCE,
            summary_explanation="No claims were analyzed. Evidence is insufficient to verify the article.",
            total_claims=0,
            supported_claims=0,
            contradicted_claims=0,
            uncertain_claims=0,
            high_importance_claims=0,
            claim_results=[],
            key_limitations=["No factual claims provided or extracted."],
            total_unique_domains=0,
            total_independent_sources=0,
            total_primary_sources=0,
            total_syndicated_sources=0,
            total_opinion_sources=0,
            total_filtered_irrelevant_sources=0,
        )

    total_claims = len(claim_results)
    supported = [c for c in claim_results if c.final_stance == ClaimStance.SUPPORTED]
    contradicted = [c for c in claim_results if c.final_stance == ClaimStance.CONTRADICTED]
    uncertain = [c for c in claim_results if c.final_stance == ClaimStance.UNCERTAIN]

    high_importance = [c for c in claim_results if c.importance == ClaimImportance.HIGH]
    high_supported = [c for c in high_importance if c.final_stance == ClaimStance.SUPPORTED]
    high_contradicted = [c for c in high_importance if c.final_stance == ClaimStance.CONTRADICTED]

    key_limitations: List[str] = []
    for c in claim_results:
        for lim in c.evidence_limitations:
            if lim not in key_limitations:
                key_limitations.append(lim)

    # Calculate domain and provenance diversity totals
    all_domains: Set[str] = set()
    for c in claim_results:
        for d in c.unique_domains:
            if d:
                all_domains.add(d)

    total_primary = sum(c.primary_sources_count for c in claim_results)
    total_independent = sum(c.independent_sources_count for c in claim_results)
    total_syndicated = sum(c.syndicated_sources_count for c in claim_results)
    total_opinion = sum(c.opinion_sources_count for c in claim_results)
    total_filtered_irrelevant = sum(c.filtered_irrelevant_sources_count for c in claim_results)

    # Aggregation logic respecting claim importance
    if high_contradicted:
        if high_supported:
            overall = OverallAssessment.MIXED
            summary = (
                f"The article contains mixed veracity: high-importance assertions are supported ({len(high_supported)}), "
                f"while other central assertions are contradicted by external evidence ({len(high_contradicted)})."
            )
        else:
            overall = OverallAssessment.LIKELY_CONTRADICTED
            summary = (
                f"The article is likely contradicted: central high-importance claim(s) "
                f"({', '.join(c.claim_id for c in high_contradicted)}) are directly refuted by reliable evidence."
            )
    elif contradicted:
        if supported:
            overall = OverallAssessment.MIXED
            summary = (
                f"The article presents a mixed picture: {len(supported)} claim(s) are supported, "
                f"but {len(contradicted)} claim(s) conflict with reliable external sources."
            )
        else:
            overall = OverallAssessment.LIKELY_CONTRADICTED
            summary = f"Key claim(s) ({', '.join(c.claim_id for c in contradicted)}) are contradicted by external evidence."
    elif supported and not contradicted:
        if high_importance and len(high_supported) >= len(high_importance) / 2:
            overall = OverallAssessment.LIKELY_SUPPORTED
            summary = (
                f"The article's core factual assertions are likely supported: {len(supported)} claim(s) "
                "are substantiated by external evidence, and no major claims were contradicted."
            )
        elif len(supported) >= total_claims / 2:
            overall = OverallAssessment.LIKELY_SUPPORTED
            summary = f"The article's assertions are likely supported by external evidence ({len(supported)} of {total_claims} claims verified)."
        else:
            overall = OverallAssessment.INSUFFICIENT_EVIDENCE
            summary = (
                "Some claims are supported, but the remaining claims lack sufficient external corroboration. "
                "Overall evidence is insufficient for a firm conclusion."
            )
    else:
        overall = OverallAssessment.INSUFFICIENT_EVIDENCE
        summary = (
            "Available external evidence is insufficient, ambiguous, or uncorroborated across claims. "
            "The system cannot verify the article conclusively."
        )

    return ArticleVerificationReport(
        overall_assessment=overall,
        summary_explanation=summary,
        total_claims=total_claims,
        supported_claims=len(supported),
        contradicted_claims=len(contradicted),
        uncertain_claims=len(uncertain),
        high_importance_claims=len(high_importance),
        claim_results=claim_results,
        key_limitations=key_limitations,
        total_unique_domains=len(all_domains),
        total_independent_sources=total_independent,
        total_primary_sources=total_primary,
        total_syndicated_sources=total_syndicated,
        total_opinion_sources=total_opinion,
        total_filtered_irrelevant_sources=total_filtered_irrelevant,
    )


def analyze_claims_and_evidence(
    claim_evaluations: List[ClaimEvaluationSummary],
    claims: List[Claim],
) -> ArticleVerificationReport:
    """End-to-end analyzer: takes evaluated claims evidence and produces the full verification report."""
    claim_dict = {c.claim_id: c for c in claims}
    claim_analyses: List[ClaimAnalysisResult] = []

    for eval_summary in claim_evaluations:
        claim = claim_dict.get(eval_summary.claim_id)
        if not claim:
            continue
        res = synthesize_claim_stance(
            claim=claim,
            evaluated_items=eval_summary.evaluated_items,
            filtered_items=eval_summary.filtered_irrelevant_items,
        )
        claim_analyses.append(res)

    return generate_article_report(claim_analyses)
