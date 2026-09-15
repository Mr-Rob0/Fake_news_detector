"""Claim extraction module for the AI News Verification System.

Extracts verifiable factual claims from article content or user text.
Filters out subjective opinions, conversational filler, and rhetorical questions,
and isolates distinct assertions for downstream evidence retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re

# Minimum length for a candidate claim sentence
MIN_CLAIM_CHAR_LENGTH = 20
MIN_CLAIM_WORD_COUNT = 4

# Sentence splitting pattern preserving common abbreviations
ABBREVIATIONS = (
    r"(?:e\.g|i\.e|mr|mrs|ms|dr|prof|inc|ltd|co|dept|est|vs|u\.s|u\.k|u\.n|"
    r"no|approx|rs|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
)

# Patterns indicative of subjective opinions or non-factual statements
OPINION_MARKERS = re.compile(
    r"\b(?:in\s+my\s+opinion|i\s+think|i\s+believe|i\s+feel|we\s+believe|in\s+our\s+view|"
    r"personally|it\s+seems\s+to\s+me|who\s+knows|should\s+be|ought\s+to|what\s+a\s+shame|"
    r"thankfully|unfortunately|luckily|amazingly|hopefully)\b",
    re.IGNORECASE,
)

# Comprehensive domain-agnostic Hindi grammatical stopwords
HINDI_STOPWORDS = {
    "में", "पर", "से", "को", "का", "की", "के", "ने", "तक", "द्वारा", "हेतु", "प्रति",
    "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे", "होना", "होने", "होता", "होती", "होते",
    "रहा", "रही", "रहे", "कर", "करके", "किया", "किए", "दिया", "दिए", "लिया", "लिए",
    "यह", "वह", "इस", "उस", "इन", "उन", "इसे", "उसे", "इन्हें", "उन्हें", "इसका", "उसका",
    "इनका", "उनका", "जिसका", "जिसके", "जिसकी", "कौन", "क्या", "किसे", "किसका", "किसने",
    "किसके", "कुछ", "कोई", "सब", "सभी", "और", "तथा", "एवं", "या", "अथवा", "भी", "तो",
    "ही", "लेकिन", "किन्तु", "परन्तु", "मगर", "क्योंकि", "चूंकि", "इसलिए", "यदि", "अगर",
    "जब", "तब", "जहां", "वहां", "जैसे", "वैसे", "कि", "अपने", "अपनी", "अपना", "वाले", "वाली", "वाला",
    "जा", "जाता", "जाती", "जाते", "गया", "गई", "गए", "हुए", "हुआ", "हुई"
}

# Patterns indicating factual, testable assertions (quantities, dates, entities, regulatory actions)
FACTUAL_INDICATORS = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?%?\b|"  # Numbers, percentages, decimals
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\b(?:19\d\d|20\d\d)\b|"  # Years
    r"\b(?:dollar|dollars|rupee|rupees|crore|lakh|billion|million|trillion|percent)\b|"
    r"[\$€£₹]|"  # Currency symbols
    r"\b(?:announced|increased|decreased|approved|reported|confirmed|signed|passed|launched|"
    r"stated|declared|discovered|appointed|decided|published|ordered|ruled|banned|rejected)\b|"
    r"\b(?:government|ministry|court|bank|president|minister|governor|commission|agency|"
    r"parliament|senate|department|police|spokesperson|organization|who|un|rbi|nasa|fda|cdc|sec|isro)\b|"
    r"(?:प्रतिशत|करोड़|लाख|अरब|घोषणा|फैसला|आदेश|दावा|पुष्टि|बैन|प्रतिबंध|सरकार|मंत्रालय|अदालत|पुलिस|बैंक|राष्ट्रपति|मंत्री|सेना)"
    r")",
    re.IGNORECASE,
)


@dataclass
class DecomposedClaim:
    """Domain-agnostic structured representation of a factual claim's components."""

    subject: str = ""
    action: str = ""
    object: str = ""
    quantity: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    attribution: Optional[str] = None
    qualification: Optional[str] = None
    is_negated: bool = False
    named_entities: List[str] = field(default_factory=list)
    core_concepts: List[str] = field(default_factory=list)


@dataclass
class Claim:
    """Represents an extracted verifiable factual claim."""

    claim_id: str
    text: str
    source_sentence_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    decomposed: Optional[DecomposedClaim] = None
    sub_claims: List[str] = field(default_factory=list)


def split_into_sentences(text: str) -> List[str]:
    """Splits plain text into clean sentences while respecting common abbreviations."""
    cleaned = text.strip()
    if not cleaned:
        return []

    # Protect abbreviations by temporarily replacing their periods
    def replace_abbrev(match: re.Match) -> str:
        return match.group(0).replace(".", "@@DOT@@")

    protected = re.sub(
        rf"\b{ABBREVIATIONS}\.",
        replace_abbrev,
        cleaned,
        flags=re.IGNORECASE,
    )

    # Protect decimal numbers like 6.5 or 3.14
    protected = re.sub(r"(\d+)\.(\d+)", r"\1@@DOT@@\2", protected)

    # Split on sentence terminals followed by whitespace or end of string (including Hindi danda U+0964 and double danda U+0965)
    raw_sentences = re.split(r"(?<=[.!?।॥])\s+|\n+", protected)

    sentences: List[str] = []
    for s in raw_sentences:
        restored = s.replace("@@DOT@@", ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def is_viable_claim(sentence: str) -> bool:
    """Determines whether a candidate sentence constitutes a verifiable factual claim."""
    cleaned = sentence.strip()

    # Rule 1: Length bounds
    if len(cleaned) < MIN_CLAIM_CHAR_LENGTH:
        return False
    words = cleaned.split()
    if len(words) < MIN_CLAIM_WORD_COUNT:
        return False

    # Rule 2: Exclude questions and pure exclamations
    if cleaned.endswith("?") or "?" in cleaned or cleaned.endswith("؟"):
        return False
    if cleaned.endswith("!") and not FACTUAL_INDICATORS.search(cleaned):
        return False

    # Rule 3: Exclude heavy opinion markers unless strongly counterbalanced by factual metrics
    has_opinion = bool(OPINION_MARKERS.search(cleaned))
    has_factual = bool(FACTUAL_INDICATORS.search(cleaned))

    if has_opinion and not has_factual:
        return False

    return True


def score_claim_priority(sentence: str) -> int:
    """Scores a claim sentence by factual density (numbers, entities, action verbs)."""
    score = 0
    # Count factual indicator matches
    matches = FACTUAL_INDICATORS.findall(sentence)
    score += len(matches) * 2

    # Bonus for numbers and percentages
    if re.search(r"\b\d+(?:\.\d+)?%?\b", sentence):
        score += 3

    # Bonus for capitalized proper nouns/acronyms (excluding sentence start)
    words = sentence.split()[1:]
    capitalized = [w for w in words if w and w[0].isupper() and w.isalpha()]
    score += len(capitalized)

    # Bonus for substantive Devanagari content words
    dev_words = [
        w for w in words
        if any('\u0900' <= c <= '\u097F' for c in w)
        and len(w) >= 3
        and w not in HINDI_STOPWORDS
    ]
    score += min(len(dev_words), 3)

    return score


def decompose_claim_text(text: str) -> DecomposedClaim:
    """Decomposes a factual claim into domain-agnostic structured components.

    Extracts: subject, action, object, quantity, location, time, attribution,
    negation status, and isolates proper named entities from generic concepts.
    """
    cleaned = text.strip()
    
    # 1. Detect source attribution (English and Hindi attribution markers)
    attribution = None
    statement_body = cleaned
    attr_match = re.search(
        r"^(?:according to ([^,\.;]+)|(?:a\s+)?(?:\d{4}\s+)?(?:study|report|survey)\s+by\s+([^,\.;]+,\s*[^,\.;]+|[^,\.;]+)|([A-Z][A-Za-z0-9\s\.\-]+?)\s+(?:said|stated|reported|announced|claimed|confirmed|found))\s+(?:that\s+|,)?",
        cleaned,
        re.IGNORECASE,
    )
    if attr_match:
        matched_str = attr_match.group(0).strip(" ,")
        attribution = matched_str
        remainder = cleaned[attr_match.end():].strip()
        if len(remainder) >= 15:
            statement_body = remainder
    else:
        # Hindi attribution patterns (e.g. 'X के अनुसार', 'X ने बताया/कहा/दावा किया')
        hi_attr = re.search(
            r"^(?:([^\,\;\।]+?)\s+के\s+अनुसार|([^\,\;\।]+?)\s+ने\s+(?:बताया|कहा|दावा किया|पुष्टि की))\s*(?:कि|,)?",
            cleaned,
        )
        if hi_attr:
            attribution = hi_attr.group(0).strip(" ,।")
            remainder = cleaned[hi_attr.end():].strip()
            if len(remainder) >= 10:
                statement_body = remainder

    # 2. Negation detection
    negation_patterns = re.compile(
        r"\b(?:not|no|never|neither|nor|cannot|can't|didn't|did not|won't|will not|"
        r"wasn't|was not|weren't|were not|hasn't|has not|haven't|have not|"
        r"denied|rejected|failed to|ruled out|dismissed)\b|"
        r"(?:\bनहीं\b|\bना\b|\bइनकार\b|\bखारिज\b)",
        re.IGNORECASE,
    )
    is_negated = bool(negation_patterns.search(statement_body))

    # 3. Quantities & Metrics
    quantity_matches = re.findall(
        r"(?:nearly|approximately|about|over|around|up to)?\s*(?:[\$€£₹]\s*)?\b\d+(?:\.\d+)?%?\b(?:\s*(?:percent|crore|lakh|billion|million|trillion|students|people|officers|times a week|प्रतिशत|करोड़|लाख|अरब))?",
        statement_body,
        re.IGNORECASE,
    )
    quantities = [q.strip() for q in quantity_matches if q.strip()]
    quantity_str = ", ".join(quantities) if quantities else None

    # 4. Temporal / Date / Time expressions
    time_matches = re.findall(
        r"\b(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|in\s+(?:19\d\d|20\d\d)|on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?|\b(?:19\d\d|20\d\d)\b)\b",
        cleaned,
        re.IGNORECASE,
    )
    time_str = ", ".join(time_matches) if time_matches else None

    # 5. Location expressions
    loc_match = re.search(
        r"\b(?:in|at|near|across)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+city)?(?:\s+near\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?)\b",
        statement_body,
    )
    if loc_match:
        location_str = loc_match.group(1).strip()
    else:
        # Hindi locative postposition: word(s) preceding 'में' or 'पर'
        loc_hi = re.search(r"([\u0900-\u097F\w]{3,})\s+(?:में|के\s+पास)(?:\s|$)", statement_body)
        location_str = loc_hi.group(1).strip() if loc_hi else None

    # 6. Extract Named Entities (Proper nouns, multi-word capitalized phrases, acronyms)
    words = cleaned.split()
    named_entities: List[str] = []
    
    # Check multi-word capitalized phrases first (Latin)
    multi_caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", cleaned)
    for mc in multi_caps:
        if mc not in named_entities:
            named_entities.append(mc)

    # Check individual capitalized tokens or acronyms (Latin)
    for idx, w in enumerate(words):
        clean_w = w.strip(".,;:\"'()?![]{}।॥")
        if not clean_w:
            continue
        if clean_w.isupper() and len(clean_w) >= 2:
            if clean_w not in named_entities:
                named_entities.append(clean_w)
        elif clean_w[0].isupper() and len(clean_w) >= 3:
            COMMON_STARTERS = {
                "the", "this", "that", "these", "those", "police", "government",
                "survey", "study", "report", "research", "analysis", "data", "findings",
                "officials", "sources", "nearly", "almost", "about", "according", "after",
                "before", "during", "several", "many", "most", "some", "all", "new", "recent"
            }
            if idx == 0 and clean_w.lower() in COMMON_STARTERS:
                continue
            if clean_w.lower() in ("survey", "study", "report", "analysis", "nonprofit"):
                continue
            if not any(clean_w in entity for entity in named_entities):
                named_entities.append(clean_w)

    # 7. Core concept words (substantive non-stopword tokens across Latin & Devanagari scripts)
    generic_stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "shall", "should", "may", "might", "must", "can",
        "could", "it", "its", "their", "they", "them", "that", "this", "these",
        "those", "which", "who", "whom", "whose", "what", "where", "when", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }
    raw_tokens = re.findall(r"[\u0900-\u097F\w]+", statement_body)
    core_concepts: List[str] = []
    for tok in raw_tokens:
        tok_low = tok.lower()
        is_dev = any('\u0900' <= c <= '\u097F' for c in tok)
        # Single-character tokens must NEVER be treated as core concepts
        if len(tok_low) < 2:
            continue
        if is_dev:
            if tok_low in HINDI_STOPWORDS:
                continue
            if tok_low not in core_concepts and not any(tok_low == ne.lower() for ne in named_entities):
                core_concepts.append(tok_low)
        else:
            if tok_low in generic_stopwords or len(tok_low) < 3:
                continue
            if tok_low not in core_concepts and not any(tok_low == ne.lower() for ne in named_entities):
                core_concepts.append(tok_low)
    core_concepts = list(dict.fromkeys(core_concepts))[:10]

    # 8. Identify main action/verb and subject
    action_match = re.search(
        r"\b(?:detained|arrested|launched|increased|decreased|approved|rejected|"
        r"reported|confirmed|found|announced|ordered|banned|ruled|beaten|punished|"
        r"killed|signed|passed|held|kept|used|struck|filed|investigated)\b|"
        r"(?:हमला|घुसपैठ|गिरफ्तार|हिरासत|घोषणा|दावा|खारिज|पुष्टि|प्रतिबंध|आदेश|फैसला|"
        r"शुरू|हस्ताक्षर|दर्ज|जांच|मंजूरी|रद्द|हताहत|मौत|बताया|कहा)",
        statement_body,
        re.IGNORECASE,
    )
    action_str = action_match.group(0).lower() if action_match else ""

    # Subject extraction: words preceding the action or attribution
    subject_str = ""
    if action_match:
        before_action = statement_body[:action_match.start()].strip()
        words_before = before_action.split()
        if words_before:
            subject_str = " ".join(words_before[-4:])

    # Object extraction: words following the action
    object_str = ""
    if action_match:
        after_action = statement_body[action_match.end():].strip()
        words_after = after_action.split()
        if words_after:
            object_str = " ".join(words_after[:5])

    return DecomposedClaim(
        subject=subject_str,
        action=action_str,
        object=object_str,
        quantity=quantity_str,
        location=location_str,
        time=time_str,
        attribution=attribution,
        qualification=None,
        is_negated=is_negated,
        named_entities=named_entities,
        core_concepts=core_concepts,
    )


def extract_atomic_predicates(sentence: str) -> List[str]:
    """Decomposes compound sentences with multiple factual predicates into atomic assertions.

    Example:
        'Police detained 40 students at 5 PM and used tear gas.'
        -> ['Police detained 40 students at 5 PM', 'Police used tear gas']
    """
    cleaned = sentence.strip()
    if not cleaned:
        return []

    # Check for coordinated verb clauses joined by ' and ' or ' while '
    # Must have action verbs in both parts to justify splitting
    conjunction_pattern = re.compile(r"\s+(?:and|while|as well as)\s+", re.IGNORECASE)
    parts = conjunction_pattern.split(cleaned)

    if len(parts) < 2:
        return [cleaned]

    # Check if first part has a subject and verb
    action_verbs = re.compile(
        r"\b(?:detained|arrested|used|launched|increased|decreased|approved|rejected|"
        r"reported|found|announced|ordered|banned|ruled|beaten|fired|struck)\b",
        re.IGNORECASE,
    )

    p1 = parts[0].strip()
    p2 = parts[1].strip()

    if action_verbs.search(p1) and action_verbs.search(p2):
        # Determine if part 2 lacks an explicit subject
        first_word_p2 = p2.split()[0].lower() if p2.split() else ""
        if action_verbs.match(first_word_p2):
            # Extract subject from p1 (first few words)
            p1_words = p1.split()
            subject = " ".join(p1_words[:2]) if len(p1_words) >= 2 else p1_words[0]
            p2_reconstructed = f"{subject} {p2}"
            return [p1, p2_reconstructed]
        else:
            return [p1, p2]

    return [cleaned]


def extract_claims(text: str, max_claims: int = 5, decompose_compound: bool = False) -> List[Claim]:
    """Extracts prioritized, verifiable factual claims from text.

    Args:
        text: Article content or input text.
        max_claims: Maximum number of claims to extract (default: 5).
        decompose_compound: If True, compound claims with multiple independent
            predicates are separated into distinct atomic claims.

    Returns:
        List of Claim objects with unique identifiers (C001, C002, ...).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    candidates = []
    for idx, sentence in enumerate(sentences):
        if is_viable_claim(sentence):
            priority = score_claim_priority(sentence)
            candidates.append((idx, sentence, priority))

    # If no candidate passed the strict filter but text was provided, fallback to first substantive sentence
    if not candidates and sentences:
        for idx, sentence in enumerate(sentences):
            if len(sentence.strip()) >= MIN_CLAIM_CHAR_LENGTH and not sentence.strip().endswith("?"):
                candidates.append((idx, sentence.strip(), 1))
                break

    # Sort by priority score descending while preserving original order for ties
    candidates.sort(key=lambda item: (-item[2], item[0]))

    # Pick top candidates
    selected = candidates[:max_claims]
    selected.sort(key=lambda item: item[0])

    claims: List[Claim] = []
    claim_count = 0
    for idx, sentence, score in selected:
        sub_predicates = extract_atomic_predicates(sentence) if len(sentence) > 30 else [sentence]
        
        if decompose_compound and len(sub_predicates) > 1:
            for sub_p in sub_predicates:
                if claim_count >= max_claims:
                    break
                claim_count += 1
                claim_id = f"C{claim_count:03d}"
                decomposed = decompose_claim_text(sub_p)
                claims.append(
                    Claim(
                        claim_id=claim_id,
                        text=sub_p,
                        source_sentence_index=idx,
                        metadata={"priority_score": score, "parent_sentence": sentence},
                        decomposed=decomposed,
                        sub_claims=[sub_p],
                    )
                )
        else:
            claim_count += 1
            claim_id = f"C{claim_count:03d}"
            decomposed = decompose_claim_text(sentence)
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=sentence,
                    source_sentence_index=idx,
                    metadata={"priority_score": score},
                    decomposed=decomposed,
                    sub_claims=sub_predicates if len(sub_predicates) > 1 else [],
                )
            )

    return claims
