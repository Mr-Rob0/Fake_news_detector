"""Source evaluation and evidence quality scoring module (Phase 6).

Evaluates the epistemological quality, provenance, independence, and transparency
of retrieved evidence sources without political bias or ideological blacklists.

Core Evaluation Rules:
1. Primary / Official sources receive highest weight (government, courts, regulators, academic).
2. Opinion, editorial, and commentary content is identified and demoted for factual verification.
3. Syndicated wire stories (PTI, ANI, AP, Reuters wire) are flagged and grouped to prevent
   counting duplicated wire copy as independent corroboration.
4. Sensational and clickbait titles are penalized.
5. News organizations are treated as corroborating evidence sources, not absolute ground truth.
6. If high-quality independent evidence is absent, the claim is flagged as an UNCERTAIN candidate.
7. Every demoted or lower-ranked source stores an explicit human-readable reason.
8. No political or ideological blacklisting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set
from urllib.parse import urlparse
import re

from src.claim_extractor import HINDI_STOPWORDS
from src.search_provider import extract_domain_from_url
from src.evidence_retriever import (
    EvidenceItem,
    ClaimEvidenceResult,
    GOVERNMENT_DOMAINS,
    INSTITUTIONAL_DOMAINS,
    REPUTABLE_NEWS_DOMAINS,
    QUERY_STOPWORDS,
)


class SourceCategory(str, Enum):
    """Categorized provenance and genre of an evidence source."""

    PRIMARY_OFFICIAL = "Primary / Official Document"
    ACADEMIC_RESEARCH = "Academic / Peer-Reviewed Research"
    FACTUAL_REPORTING = "Factual News Reporting"
    SYNDICATED_WIRE = "Syndicated News Wire Copy"
    OPINION_EDITORIAL = "Opinion / Editorial / Commentary"
    SENSATIONAL_CLICKBAIT = "Sensational / Clickbait Framing"
    GENERAL_WEB = "General Web Content"


class IndependenceStatus(str, Enum):
    """Independence status of an evidence source."""

    INDEPENDENT = "Independent Reporting"
    SYNDICATED = "Syndicated News Feed"
    DUPLICATE_WIRE = "Duplicate Wire Copy"
    UNKNOWN = "Unknown / General"


class RelevanceLevel(str, Enum):
    """Degree of relevance between a retrieved source and a factual claim."""

    DIRECT = "Directly Relevant"
    PARTIAL = "Partially Relevant / Contextual"
    WEAK = "Weak / Peripheral"
    IRRELEVANT = "Irrelevant / Unrelated"


@dataclass
class RelevanceAssessment:
    """Structured assessment of factual relevance between claim and source."""

    level: RelevanceLevel
    score: float = 0.0
    is_eligible_for_stance: bool = True
    matched_entities: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    matched_predicates: List[str] = field(default_factory=list)
    matched_numbers: List[str] = field(default_factory=list)
    location_match: Optional[bool] = None
    temporal_match: Optional[bool] = None
    explanation: str = ""
    rejection_reasons: List[str] = field(default_factory=list)


class EvidenceStrength(str, Enum):
    """Overall evidentiary strength rating of an individual source."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"


# Academic / Research domain indicators
ACADEMIC_DOMAINS = {
    "edu", "ac.in", "ac.uk", "nature.com", "sciencemag.org", "sciencedirect.com",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "thelancet.com", "nejm.org",
    "ieee.org", "arxiv.org", "springer.com", "jstor.org"
}

# Common action and predicate synonym sets for domain-agnostic semantic matching
ACTION_SYNONYM_GROUPS = [
    {"start", "started", "starting", "launch", "launched", "launching", "initiate", "initiated", "initiating", "commenced", "commence", "kicked", "kicks"},
    {"approve", "approved", "approving", "pass", "passed", "passing", "adopt", "adopted", "adopting", "ratify", "ratified", "authorize", "authorized", "operationalize", "operationalized"},
    {"join", "joined", "joining", "accede", "acceded", "accession", "enter", "entered", "member", "admitted"},
    {"issue", "issued", "issuing", "publish", "published", "publishing", "release", "released", "releasing", "notify", "notified"},
    {"arrest", "arrested", "arresting", "detain", "detained", "detaining", "apprehend", "apprehended", "custody"},
    {"sign", "signed", "signing", "ink", "inked", "conclude", "concluded", "finalize", "finalized"},
    {"find", "found", "finding", "observe", "observed", "observing", "discover", "discovered", "discovering", "detect", "detected", "detecting"},
    {"increase", "increased", "increasing", "raise", "raised", "raising", "hike", "hiked", "rose", "risen", "grew", "surged", "jumped"},
    {"decrease", "decreased", "decreasing", "cut", "cutting", "slash", "slashed", "reduce", "reduced", "lowered", "fell", "dropped", "slumped"},
    {"hold", "held", "holding", "meet", "met", "meeting", "convene", "convened", "gather", "gathered"},
    {"lathi", "baton", "batons", "caning", "tear gas", "water cannon", "dispersed", "disperse", "dispersing"},
    {"protest", "protests", "protester", "protesters", "protestor", "protestors", "march", "marching", "rally", "demonstration", "demonstrators", "agitation"},
]

# Patterns indicating opinion, column, blog, or editorial content
OPINION_PATH_PATTERNS = re.compile(
    r"/(?:opinion|editorial|columns?|blogs?|commentary|viewpoint|analysis/opinion|voices?|perspectives?)/",
    re.IGNORECASE,
)

OPINION_TITLE_PATTERNS = re.compile(
    r"^\s*(?:"
    r"opinion\s*[:\-\u2013\u2014|]|"
    r"editorial\s*[:\-\u2013\u2014|]|"
    r"column\s*[:\-\u2013\u2014|]|"
    r"comment\s*[:\-\u2013\u2014|]|"
    r"viewpoint\s*[:\-\u2013\u2014|]|"
    r"guest\s+column\s*[:\-\u2013\u2014|]|"
    r"analysis\s*[:\-\u2013\u2014|]\s*why|"
    r"why\s+(?:we|i|you)\s+should|"
    r"here'?s?\s+why\s+i\s+think|"
    r"letter\s+to\s+(?:the\s+)?editor"
    r")",
    re.IGNORECASE,
)

# Patterns identifying wire service syndication and copy attribution
WIRE_ATTRIBUTION_PATTERNS = [
    (re.compile(r"\b(?:pti|press\s+trust\s+of\s+india)\b", re.IGNORECASE), "PTI (Press Trust of India)"),
    (re.compile(r"\b(?:ani|asian\s+news\s+international)\b", re.IGNORECASE), "ANI (Asian News International)"),
    (re.compile(r"\b(?:reuters(?:\s+wire|\s+news\s+service)?)\b", re.IGNORECASE), "Reuters Wire"),
    (re.compile(r"\b(?:associated\s+press|ap\s+wire|ap\s+news)\b", re.IGNORECASE), "Associated Press (AP) Wire"),
    (re.compile(r"\b(?:afp|agence\s+france[\s\-]presse)\b", re.IGNORECASE), "AFP (Agence France-Presse)"),
    (re.compile(r"\b(?:ians|indo[\s\-]asian\s+news\s+service)\b", re.IGNORECASE), "IANS (Indo-Asian News Service)"),
    (re.compile(r"\b(?:syndicated\s+feed|news\s+agency\s+feed|syndicated\s+from)\b", re.IGNORECASE), "Syndicated News Feed"),
]

# Patterns detecting clickbait, sensationalism, and emotional manipulation
CLICKBAIT_ALL_CAPS = re.compile(r"\b(?:SHOCKING|UNBELIEVABLE|SCANDAL|BOMBSHELL|MASSIVE|CRAZY|EXPOSED|MUST SEE|WATCH)\b")
CLICKBAIT_PUNCTUATION = re.compile(r"[!?]{2,}")
CLICKBAIT_TITLES = re.compile(
    r"\b(?:you\s+won'?t\s+believe|what\s+happened\s+next|blows\s+your\s+mind|"
    r"doctors\s+hate\s+(?:this|him|her)|this\s+one\s+trick)\b",
    re.IGNORECASE,
)

# Patterns detecting named sources, official attribution, and spokespersons
NAMED_SOURCE_PATTERNS = [
    re.compile(r"\b(?:according to|reported by|quoted|citing)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
    re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:said|stated|told|alleged|confirmed|argued|denied|warned)\b"),
    re.compile(r"\b(?:police commissioner|spokesperson|deputy commissioner|officials?|director general|ministry of [A-Za-z]+|supreme court bench)\b", re.IGNORECASE),
]


@dataclass
class SourceEvaluation:
    """Detailed quality evaluation and provenance metadata for an individual source."""

    evidence_id: str
    category: SourceCategory
    quality_score: int  # 0 to 100
    is_primary: bool
    is_opinion: bool
    is_syndicated_wire: bool
    wire_agency: Optional[str] = None
    is_independent: bool = True
    transparency_notes: List[str] = field(default_factory=list)
    demotion_reason: Optional[str] = None
    independence_status: IndependenceStatus = IndependenceStatus.INDEPENDENT
    relevance_level: RelevanceLevel = RelevanceLevel.DIRECT
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    has_named_sources: bool = False
    has_direct_quotes: bool = False
    has_specific_metrics: bool = False
    ranking_reasons: List[str] = field(default_factory=list)
    temporal_note: Optional[str] = None
    relevance_assessment: Optional[RelevanceAssessment] = None
    is_relevant: bool = True


@dataclass
class EvaluatedEvidenceItem:
    """Evidence item enriched with objective source quality evaluation."""

    evidence: EvidenceItem
    evaluation: SourceEvaluation


@dataclass
class ClaimEvaluationSummary:
    """Aggregated source evaluation metrics for a single claim's evidence collection."""

    claim_id: str
    evaluated_items: List[EvaluatedEvidenceItem] = field(default_factory=list)
    filtered_irrelevant_items: List[EvaluatedEvidenceItem] = field(default_factory=list)
    primary_source_count: int = 0
    independent_source_count: int = 0
    opinion_source_count: int = 0
    duplicate_wire_count: int = 0
    filtered_irrelevant_count: int = 0
    has_sufficient_independent_evidence: bool = False
    quality_assessment_note: str = ""
    unique_domains: Set[str] = field(default_factory=set)
    unique_domain_count: int = 0
    syndicated_source_count: int = 0


def detect_opinion_content(title: str, url: str, snippet: str) -> Tuple[bool, Optional[str]]:
    """Detects whether a source represents opinion, editorial, or columnist commentary."""
    # Check URL path
    try:
        path = urlparse(url).path
        if OPINION_PATH_PATTERNS.search(path):
            return True, "URL path indicates an opinion, blog, or editorial section."
    except Exception:
        pass

    # Check Title prefix or headline structure
    if OPINION_TITLE_PATTERNS.search(title):
        return True, "Title contains standard opinion, editorial, or commentary markers."

    # Check Snippet indicator
    if OPINION_TITLE_PATTERNS.search(snippet):
        return True, "Content snippet indicates an opinion or editorial stance."

    return False, None


def detect_syndicated_wire(title: str, snippet: str, domain: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Detects whether content originates from a syndicated wire agency (PTI, ANI, AP, Reuters, IANS)."""
    text_to_check = f"{title} {snippet}"
    clean_domain = (domain or "").lower().strip()

    for pattern, agency in WIRE_ATTRIBUTION_PATTERNS:
        if pattern.search(text_to_check):
            # If the domain is the agency's primary site (e.g. reuters.com), it is direct reporting
            if clean_domain in ("reuters.com", "apnews.com", "ptinews.com", "aninews.in"):
                return False, agency
            return True, agency
    return False, None


def detect_clickbait_sensationalism(title: str) -> Tuple[bool, Optional[str]]:
    """Detects sensational or clickbait headline characteristics."""
    if CLICKBAIT_PUNCTUATION.search(title):
        return True, "Excessive punctuation (e.g. '???', '!!!') indicates sensational framing."

    caps_matches = CLICKBAIT_ALL_CAPS.findall(title)
    if len(caps_matches) >= 2 or (len(caps_matches) == 1 and len(title.split()) <= 6):
        return True, f"Sensational capitalized keywords detected ({', '.join(caps_matches)})."

    if CLICKBAIT_TITLES.search(title):
        return True, "Formulaic clickbait headline pattern detected."

    return False, None


def detect_primary_provenance(domain: str) -> Tuple[bool, bool, SourceCategory]:
    """Evaluates domain to determine if it is a primary official or academic source.

    Returns:
        Tuple of (is_primary: bool, is_academic: bool, category: SourceCategory)
    """
    clean_domain = domain.lower().strip()

    # Tier 1: Official government / court / regulatory registry
    if any(clean_domain.endswith("." + gd) or clean_domain == gd for gd in GOVERNMENT_DOMAINS):
        return True, False, SourceCategory.PRIMARY_OFFICIAL

    # Tier 2: Institutional / Central Bank / Regulatory
    if any(clean_domain.endswith("." + inst) or clean_domain == inst for inst in INSTITUTIONAL_DOMAINS):
        return True, False, SourceCategory.PRIMARY_OFFICIAL

    # Academic & Research Repositories
    if any(clean_domain.endswith("." + ac) or clean_domain == ac for ac in ACADEMIC_DOMAINS):
        return True, True, SourceCategory.ACADEMIC_RESEARCH

    # Reputable journalism
    if any(clean_domain.endswith("." + news) or clean_domain == news for news in REPUTABLE_NEWS_DOMAINS):
        return False, False, SourceCategory.FACTUAL_REPORTING

    return False, False, SourceCategory.GENERAL_WEB


def detect_named_sources(title: str, snippet: str) -> Tuple[bool, List[str]]:
    """Detects named officials, spokespersons, or institutions cited in text."""
    combined = f"{title}. {snippet}"
    found: List[str] = []
    for pattern in NAMED_SOURCE_PATTERNS:
        for match in pattern.finditer(combined):
            val = match.group(1) if match.groups() else match.group(0)
            cleaned = val.strip()
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return (len(found) > 0, found)


def detect_direct_quotes(text: str) -> bool:
    """Detects whether substantive direct quotations are present in the text."""
    return bool(re.search(r'["“][^"”]{5,}["”]', text) or "according to" in text.lower())


def detect_specific_metrics(text: str) -> bool:
    """Detects numbers, timestamps, legal sections, or specific metrics in text."""
    has_num = bool(re.search(r"(?:\$|€|£|₹)?\b\d+(?:\.\d+)?%?", text))
    has_time = bool(re.search(r"\b\d{1,2}(?::|\.)\d{2}\s*(?:am|pm)?\b", text, re.IGNORECASE))
    has_legal = bool(re.search(r"\b(?:section\s+\d+|bnss|crpc|ipc|act|order)\b", text, re.IGNORECASE))
    return has_num or has_time or has_legal


def detect_temporal_relevance(publication_date: Optional[str], claim_text: str) -> Tuple[Optional[str], Optional[int]]:
    """Evaluates temporal proximity between publication date and claim context."""
    if not publication_date:
        return ("Publication timestamp not provided in source metadata.", None)

    pub_year_match = re.search(r"\b(19\d\d|20\d\d)\b", publication_date)
    claim_year_match = re.search(r"\b(19\d\d|20\d\d)\b", claim_text)

    if pub_year_match and claim_year_match:
        pub_year = int(pub_year_match.group(1))
        claim_year = int(claim_year_match.group(1))
        diff = abs(pub_year - claim_year)
        if diff == 0:
            return (f"Publication date ({publication_date}) is contemporary with claimed event year ({claim_year}).", 0)
        elif diff <= 1:
            return (f"Publication date ({publication_date}) is reasonably close to claimed event year ({claim_year}).", diff)
        else:
            return (f"Publication date ({publication_date}) differs substantially from claimed event year ({claim_year}).", diff)

    return (f"Publication date verified: {publication_date}", None)


def detect_reference_or_linguistic_page(
    url: str,
    domain: str,
    title: str,
    snippet: str,
    claim_text: str = "",
) -> Tuple[bool, Optional[str]]:
    """Generically detects linguistic, dictionary, alphabet, grammar, directory, or reference entries.

    These pages define language, characters, or general terms rather than reporting on news events,
    and must be filtered at the Relevance Gate unless the claim itself is explicitly about linguistics.
    """
    clean_url = url.lower()
    clean_dom = domain.lower()
    clean_title = title.strip()
    clean_snippet = snippet.lower()

    # If the claim is explicitly about linguistics or dictionary definitions, allow
    linguistic_claim_keywords = {"dictionary", "grammar", "alphabet", "word origin", "शब्दावली", "व्याकरण", "वर्णमाला", "सूत"}
    if any(k in claim_text.lower() for k in linguistic_claim_keywords):
        return False, None

    # 1. Linguistic / Dictionary / Lexicon Domains
    DICTIONARY_DOMAINS = {
        "wiktionary.org", "dictionary.cambridge.org", "merriam-webster.com",
        "shabdkosh.com", "lexico.com", "collinsdictionary.com", "thefreedictionary.com",
        "urbandictionary.com", "wordsmith.org", "hin.wiktionary.org", "hi.wiktionary.org",
        "en.wiktionary.org", "hindidictionary.org", "oxfordlearnersdictionaries.com",
        "vocabulary.com", "macmillandictionary.com"
    }
    if any(clean_dom.endswith(d) or clean_dom == d for d in DICTIONARY_DOMAINS):
        return True, "Filtered: Dictionary/lexicon domain does not constitute evidentiary news reporting."

    # 2. URL paths indicative of dictionary, vocabulary, alphabet, or grammar pages
    DICTIONARY_PATH_PATTERNS = [
        "/wiki/wiktionary:", "/dictionary/", "/definition/", "/meaning-in-",
        "/meaning/", "/grammar/", "/alphabet/", "/varnamala/", "/words/"
    ]
    if any(p in clean_url for p in DICTIONARY_PATH_PATTERNS):
        return True, "Filtered: Language reference or dictionary URL path detected."

    # 3. Alphabet / Single Character Entries
    # Matches titles like "र - विक्षनरी", "R - Wikipedia", "Letter R", "र (अक्षर)"
    if re.match(r"^(?:[\u0900-\u097F]|[A-Za-z])\s*[-–—:\(\]]", clean_title):
        return True, "Filtered: Single-character/alphabet entry does not constitute news evidence."

    # 4. Content patterns indicating linguistic definition or grammar tables
    LINGUISTIC_CONTENT_MARKERS = [
        "वर्णमाला का", "व्यंजन वर्ण", "स्वर वर्ण", "is the letter of the alphabet",
        "definition and meaning", "meaning in hindi", "pronunciation and definition",
        "शब्द का अर्थ", "व्याकरण में", "प्रत्याहार सूत्र", "माहेश्वर सूत्र",
        "synonyms and antonyms", "part of speech", "noun, plural"
    ]
    if any(marker in clean_snippet or marker in clean_title.lower() for marker in LINGUISTIC_CONTENT_MARKERS):
        return True, "Filtered: Linguistic/phonetic/grammar definition content detected."

    # 5. Generic Disambiguation Pages
    if "(disambiguation)" in clean_title.lower() or "(बहुविकल्पी शब्द)" in clean_title:
        return True, "Filtered: Encyclopedia disambiguation page lacks specific factual reporting."

    if "may refer to:" in clean_snippet or "के कई अर्थ हो सकते हैं" in clean_snippet:
        return True, "Filtered: Generic disambiguation directory page."

    # 6. Category / Tag / Search Directory Pages (when not an actual article)
    CATEGORY_PATH_PATTERNS = ["/category/", "/tag/", "/topics/", "/archive/", "/search/"]
    if any(cp in clean_url for cp in CATEGORY_PATH_PATTERNS) and (
        "articles in category" in clean_snippet or "list of topics" in clean_snippet or "श्रेणी:" in clean_title
    ):
        return True, "Filtered: Directory/category listing page does not constitute evidentiary reporting."

    # 7. Search Directories / Legal or Business Portals (generic listings without news reporting)
    DIRECTORY_MARKERS = ["directory", "search-portal", "yellowpages", "find-a-lawyer", "-directory-"]
    if any(dm in clean_dom or dm in clean_url for dm in DIRECTORY_MARKERS) or "legal directory" in clean_title.lower() or "search our legal database" in clean_snippet:
        return True, "Filtered: Directory/portal listing page does not constitute evidentiary news reporting."

    return False, None


def evaluate_result_relevance(
    claim_text: str,
    title: str,
    snippet: str,
    url: str = "",
    domain: str = "",
    claim_obj: Optional[Any] = None,
) -> RelevanceAssessment:
    """Robust, domain-agnostic evaluation of factual relevance between a search result and a claim.

    Evaluates:
    0. Generic Reference & Linguistic Page Filtering (drops dictionaries, alphabet pages, grammar guides).
    1. Core Named Entity Compatibility (rejects false relevance from generic concept overlap).
    2. Event & Predicate Alignment (same action vs unrelated event).
    3. Geographic & Location Alignment.
    4. Temporal / Date Alignment.
    5. Quantitative & Metric Context.

    Returns:
        RelevanceAssessment with level in (DIRECT, PARTIAL, WEAK, IRRELEVANT) and audit rationale.
    """
    combined = f"{title}. {snippet}".strip()
    combined_lower = combined.lower()
    claim_lower = claim_text.lower()
    clean_domain = domain.lower().strip() if domain else extract_domain_from_url(url).lower()

    # Step 0: Reference & Linguistic Page Filter (Runs FIRST)
    is_ref_page, ref_reason = detect_reference_or_linguistic_page(url, clean_domain, title, snippet, claim_text)
    if is_ref_page:
        return RelevanceAssessment(
            level=RelevanceLevel.IRRELEVANT,
            score=0.0,
            is_eligible_for_stance=False,
            matched_entities=[],
            missing_entities=[],
            matched_predicates=[],
            matched_numbers=[],
            explanation=ref_reason or "Irrelevant: General reference or linguistic page.",
            rejection_reasons=[ref_reason or "General reference/dictionary page lacks factual evidentiary reporting."],
        )

    # 1. Extract structured claim elements
    decomposed = getattr(claim_obj, "decomposed", None) if claim_obj else None
    if not decomposed:
        from src.claim_extractor import decompose_claim_text
        decomposed = decompose_claim_text(claim_text)

    claim_named_entities = decomposed.named_entities if decomposed else []
    if not claim_named_entities:
        # Fallback to capitalized proper noun tokens excluding first word
        words = claim_text.split()
        claim_named_entities = [
            w.strip(".,;:\"'()?!।॥") for idx, w in enumerate(words)
            if idx > 0 and len(w) >= 3 and w[0].isupper() and w.isalpha()
            and w.lower() not in QUERY_STOPWORDS
        ]

    # Map well-known domain/entity aliases bidirectionally
    entity_aliases: Dict[str, Set[str]] = {
        "rbi": {"rbi", "reserve bank", "reserve bank of india"},
        "reserve bank of india": {"rbi", "reserve bank", "reserve bank of india"},
        "reserve bank": {"rbi", "reserve bank of india"},
        "nasa": {"nasa", "national aeronautics", "national aeronautics and space administration"},
        "national aeronautics and space administration": {"nasa", "national aeronautics"},
        "who": {"who", "world health organization"},
        "world health organization": {"who"},
        "un": {"un", "united nations"},
        "united nations": {"un"},
        "isro": {"isro", "indian space research", "indian space research organisation", "indian space research organization"},
        "indian space research organisation": {"isro", "indian space research"},
        "imf": {"imf", "international monetary fund"},
        "international monetary fund": {"imf"},
        "sebi": {"sebi", "securities and exchange board"},
        "bjp": {"bjp", "bharatiya janata party"},
        "inc": {"congress", "indian national congress"},
        "finmin": {"finmin", "finance ministry", "ministry of finance"},
        "ministry of finance": {"finmin", "finance ministry", "ministry of finance"},
        "finance ministry": {"finmin", "finance ministry", "ministry of finance"},
    }

    # 2. Check Named Entity Overlap
    matched_entities: List[str] = []
    missing_entities: List[str] = []

    for entity in claim_named_entities:
        ent_lower = entity.lower().strip()
        # Strip leading article if present (e.g. "the reserve bank" -> "reserve bank")
        ent_normalized = re.sub(r"^(?:the|a|an)\s+", "", ent_lower).strip()
        ent_words = [w for w in ent_normalized.split() if w not in QUERY_STOPWORDS]

        # Check if full entity or primary words appear in combined text, domain, or url
        is_matched = False
        target_corpus = f"{combined_lower} {clean_domain} {url.lower()}"
        if ent_normalized in target_corpus or ent_lower in target_corpus:
            is_matched = True
        elif ent_words and all(w in target_corpus for w in ent_words):
            is_matched = True
        else:
            # Check aliases
            aliases = entity_aliases.get(ent_normalized, set()) | entity_aliases.get(ent_lower, set())
            if any(a in target_corpus for a in aliases):
                is_matched = True

        if is_matched:
            matched_entities.append(entity)
        else:
            missing_entities.append(entity)

    # 3. Unicode-Aware Substantive Concept Nouns Overlap (strictly rejecting single characters)
    generic_stopwords = QUERY_STOPWORDS | {
        "study", "report", "survey", "nonprofit", "organization", "city", "near",
        "found", "said", "stated", "announced", "reported", "confirmed", "revealed",
        "percent", "year", "years", "day", "days", "official", "officials", "statement",
        "press", "release", "news", "update", "article", "source", "author"
    } | HINDI_STOPWORDS

    def normalize_token(t: str) -> str:
        if not any('\u0900' <= c <= '\u097F' for c in t):
            if t.endswith("ies") and len(t) > 4:
                return t[:-3] + "y"
            elif t.endswith("sses"):
                return t[:-2]
            elif t.endswith("is") or t.endswith("us") or t.endswith("ss"):
                return t
            elif t.endswith("s") and len(t) > 3:
                return t[:-1]
        return t

    def extract_substantive_tokens(text: str) -> Set[str]:
        raw = re.findall(r"[\u0900-\u097F\w]+", text.lower())
        tokens = set()
        for t in raw:
            # Single-character tokens must NEVER be counted as substantive overlap
            if len(t) < 2:
                continue
            is_dev = any('\u0900' <= c <= '\u097F' for c in t)
            if is_dev:
                if t not in HINDI_STOPWORDS and len(t) >= 2:
                    tokens.add(t)
            else:
                if t not in generic_stopwords and len(t) >= 3:
                    tokens.add(t)
                    norm = normalize_token(t)
                    if norm not in generic_stopwords and len(norm) >= 3:
                        tokens.add(norm)
        return tokens

    claim_substantive_tokens = extract_substantive_tokens(claim_text)
    evidence_substantive_tokens = extract_substantive_tokens(combined)
    substantive_overlap = claim_substantive_tokens & evidence_substantive_tokens

    # Generic word overlap for audit notes
    all_claim_tokens = {t for t in re.findall(r"[\u0900-\u097F\w]+", claim_lower) if len(t) >= 2}
    all_ev_tokens = {t for t in re.findall(r"[\u0900-\u097F\w]+", combined_lower) if len(t) >= 2}
    generic_overlap = (all_claim_tokens & all_ev_tokens) & generic_stopwords

    # 4. Check Predicate / Action Overlap (English & Hindi actions)
    claim_actions = set()
    if decomposed and decomposed.action:
        claim_actions.add(decomposed.action.lower())
    for w in claim_substantive_tokens:
        if (w.endswith("ed") or w.endswith("ing") or w.endswith("ised") or w.endswith("ized")) and w not in QUERY_STOPWORDS:
            claim_actions.add(w)

    HINDI_ACTION_VOCAB = {
        "हमला", "घुसपैठ", "गिरफ्तार", "हिरासत", "घोषणा", "दावा", "खारिज", "पुष्टि",
        "प्रतिबंध", "आदेश", "फैसला", "शुरू", "हस्ताक्षर", "दर्ज", "जांच", "मंजूरी",
        "रद्द", "हताहत", "मौत", "बताया", "कहा"
    }
    for ha in HINDI_ACTION_VOCAB:
        if ha in claim_lower:
            claim_actions.add(ha)

    matched_actions = list(claim_actions & evidence_substantive_tokens)
    for group in ACTION_SYNONYM_GROUPS:
        if (claim_actions & group) and (evidence_substantive_tokens & group):
            for act in (evidence_substantive_tokens & group):
                if act not in matched_actions:
                    matched_actions.append(act)

    # 5. Check Numbers & Quantities
    claim_nums = set(re.findall(r"(?:\$|€|£|₹)?\b\d+(?:\.\d+)?%?", claim_lower))
    evidence_nums = set(re.findall(r"(?:\$|€|£|₹)?\b\d+(?:\.\d+)?%?", combined_lower))
    matched_nums = list(claim_nums & evidence_nums)

    # 6. Temporal and Geographic Check
    temporal_match = None
    if decomposed and decomposed.time:
        temporal_match = any(t.lower() in combined_lower for t in decomposed.time.split(","))

    location_match = None
    if decomposed and decomposed.location:
        loc_words = [
            lw for lw in decomposed.location.lower().split()
            if lw not in ("city", "near", "across", "the", "in", "at", "में", "पर")
            and len(lw) >= 2
        ]
        location_match = any(lw in combined_lower for lw in loc_words)

    # 7. Apply Domain-Agnostic Multi-Signal Relevance Rules

    # Rule A: Claim has prominent Named Entities, but Evidence matches NONE of them
    if claim_named_entities and not matched_entities:
        ent_words_set = {
            w for ent in claim_named_entities
            for w in ent.lower().split()
            if w not in QUERY_STOPWORDS and len(w) >= 2
        }
        raw_noise = {
            "bank", "court", "government", "ministry", "board", "commission",
            "department", "authority", "council", "organization", "agency", "percent",
            "basis", "point", "points", "rate", "rates", "year", "years", "crore", "lakh", "dollar"
        }
        generic_subject_noise = raw_noise | {normalize_token(w) for w in raw_noise}
        pure_subject_overlap = substantive_overlap - ent_words_set - claim_actions - generic_subject_noise
        has_substantive_event = len(pure_subject_overlap) >= 2 or (len(pure_subject_overlap) >= 1 and len(matched_actions) >= 1)
        if not has_substantive_event and not (location_match and len(pure_subject_overlap) >= 1):
            rejection_reasons = [
                f"Entity mismatch: Does not mention claim entity '{', '.join(claim_named_entities[:3])}'.",
                f"Shares only generic or peripheral words ({', '.join(list(substantive_overlap)[:3]) or 'none'}) without identifying the specific organization, person, or subject.",
                "Irrelevant to the factual assertion."
            ]
            return RelevanceAssessment(
                level=RelevanceLevel.IRRELEVANT,
                score=0.0,
                is_eligible_for_stance=False,
                matched_entities=[],
                missing_entities=missing_entities,
                matched_predicates=matched_actions,
                matched_numbers=matched_nums,
                location_match=location_match,
                temporal_match=temporal_match,
                explanation=f"Irrelevant: Source discusses an unrelated subject and misses key entity '{', '.join(claim_named_entities[:2])}'.",
                rejection_reasons=rejection_reasons,
            )

    # Rule A_multi: Claim has multiple specific entities, but evidence matches only 1 and misses the primary actor/action
    if len(claim_named_entities) >= 2 and len(matched_entities) == 1 and not matched_actions:
        primary_entity = claim_named_entities[0].lower().strip()
        if not any(primary_entity in m.lower() for m in matched_entities):
            single_matched_tokens = set().union(*[m.lower().split() for m in matched_entities])
            sub_outside = substantive_overlap - single_matched_tokens - generic_stopwords
            if len(sub_outside) <= 1:
                return RelevanceAssessment(
                    level=RelevanceLevel.IRRELEVANT,
                    score=0.0,
                    is_eligible_for_stance=False,
                    matched_entities=matched_entities,
                    missing_entities=missing_entities,
                    matched_predicates=[],
                    matched_numbers=[],
                    location_match=location_match,
                    temporal_match=temporal_match,
                    explanation=f"Irrelevant: Mentions '{', '.join(matched_entities)}' but misses primary actor entity '{claim_named_entities[0]}' and action.",
                    rejection_reasons=[
                        f"Missing primary actor entity: '{claim_named_entities[0]}'.",
                        f"Discusses secondary entity '{matched_entities[0]}' in an unrelated context with no predicate match.",
                    ],
                )

    # Compute substantive concept overlap excluding the matched entity names themselves
    entity_tokens = set().union(*[m.lower().split() for m in matched_entities]) if matched_entities else set()
    subject_concept_overlap = substantive_overlap - entity_tokens

    # Rule A2: Institution/Org match without subject/event match
    BROAD_INSTITUTIONS = {
        "who", "un", "united nations", "world health organization", "world bank",
        "supreme court", "high court", "police", "government", "ministry",
        "federal reserve", "parliament", "congress", "senate", "nasa", "isro"
    }
    has_only_broad_inst = bool(matched_entities and all(m.lower().strip() in BROAD_INSTITUTIONS for m in matched_entities))
    if has_only_broad_inst and len(subject_concept_overlap) == 0 and not matched_actions and not location_match:
        # Check if the title/snippet is merely an educational explainer on the institution
        is_explainer = any(w in combined_lower for w in ("explainer", "guide", "governance", "history", "operates", "structure", "overview"))
        if is_explainer:
            return RelevanceAssessment(
                level=RelevanceLevel.WEAK,
                score=0.25,
                is_eligible_for_stance=False,
                matched_entities=matched_entities,
                missing_entities=missing_entities,
                matched_predicates=[],
                matched_numbers=[],
                location_match=location_match,
                temporal_match=temporal_match,
                explanation=f"Weak relevance: Educational background on '{', '.join(matched_entities)}' lacks verification of the specific event or policy action.",
                rejection_reasons=["Background/educational overview of institution only; lacks core factual predicate."],
            )
        return RelevanceAssessment(
            level=RelevanceLevel.IRRELEVANT,
            score=0.0,
            is_eligible_for_stance=False,
            matched_entities=matched_entities,
            missing_entities=missing_entities,
            matched_predicates=[],
            matched_numbers=[],
            explanation=f"Irrelevant: Mentions institution '{', '.join(matched_entities)}' but addresses an unrelated topic (zero subject matter overlap).",
            rejection_reasons=[
                f"Source mentions institution '{', '.join(matched_entities)}' but contains no substantive overlap with the claimed event or subject.",
                f"Missing critical entities/subject matter: {', '.join(missing_entities) or 'core predicate'}.",
                "Zero subject concept overlap outside the broad institution name."
            ],
        )

    # Rule B: Pure keyword fragment with no substantive overlap
    if len(substantive_overlap) == 0 and not matched_entities:
        return RelevanceAssessment(
            level=RelevanceLevel.IRRELEVANT,
            score=0.0,
            is_eligible_for_stance=False,
            matched_entities=[],
            missing_entities=missing_entities,
            matched_predicates=[],
            matched_numbers=[],
            explanation="Irrelevant: Zero substantive entity or predicate overlap with the claim.",
            rejection_reasons=["Zero substantive overlap with claim context."],
        )

    # Rule C: DIRECT Relevance
    has_strong_entity = bool(matched_entities) or (len(substantive_overlap) >= 3 and not claim_named_entities)
    has_action = bool(matched_actions) or (len(matched_entities) >= 2 and len(subject_concept_overlap) >= 1) or (not claim_actions and len(substantive_overlap) >= 3)
    has_num_or_date = bool(matched_nums) or (temporal_match is True) or (not claim_nums and not (decomposed and decomposed.time))

    # For DIRECT relevance: require strong entity + action/location + at least 1-2 substantive concept words
    if has_strong_entity and has_action and (has_num_or_date or len(substantive_overlap) >= 2) and (len(subject_concept_overlap) >= 1 or location_match or len(substantive_overlap) >= 2):
        matched_details = []
        if matched_entities:
            matched_details.append(f"Entity: {', '.join(matched_entities)}")
        if matched_actions:
            matched_details.append(f"Predicate: {', '.join(matched_actions)}")
        if matched_nums:
            matched_details.append(f"Quantity: {', '.join(matched_nums)}")
        if location_match:
            matched_details.append("Location matched")

        return RelevanceAssessment(
            level=RelevanceLevel.DIRECT,
            score=0.90,
            is_eligible_for_stance=True,
            matched_entities=matched_entities,
            missing_entities=missing_entities,
            matched_predicates=matched_actions,
            matched_numbers=matched_nums,
            location_match=location_match,
            temporal_match=temporal_match,
            explanation=f"Directly addresses claim assertions ({'; '.join(matched_details)}).",
            rejection_reasons=[],
        )

    # Rule D: PARTIAL Relevance (Corroborating entity / context)
    has_substantive_for_partial = len(substantive_overlap) >= 3 if claim_named_entities else len(substantive_overlap) >= 2
    if (matched_entities and (len(subject_concept_overlap) >= 1 or location_match or has_action)) or has_substantive_for_partial:
        return RelevanceAssessment(
            level=RelevanceLevel.PARTIAL,
            score=0.60,
            is_eligible_for_stance=True,
            matched_entities=matched_entities,
            missing_entities=missing_entities,
            matched_predicates=matched_actions,
            matched_numbers=matched_nums,
            location_match=location_match,
            temporal_match=temporal_match,
            explanation=f"Partially relevant: Corroborates key entity ({', '.join(matched_entities) if matched_entities else 'topic'}) and context, but secondary predicate detail is omitted.",
            rejection_reasons=[],
        )

    # Rule E: WEAK Relevance (Peripheral mention only)
    if len(substantive_overlap) >= 1 or len(matched_entities) >= 1:
        return RelevanceAssessment(
            level=RelevanceLevel.WEAK,
            score=0.25,
            is_eligible_for_stance=False,  # Weak sources do not participate in decisive stance
            matched_entities=matched_entities,
            missing_entities=missing_entities,
            matched_predicates=matched_actions,
            matched_numbers=matched_nums,
            location_match=location_match,
            temporal_match=temporal_match,
            explanation="Weak relevance: Mentions related terms or subject in passing, but lacks substantive verification of the specific event.",
            rejection_reasons=["Peripheral topical mention only; lacks core factual predicate."],
        )

    # Default fallback: IRRELEVANT
    return RelevanceAssessment(
        level=RelevanceLevel.IRRELEVANT,
        score=0.0,
        is_eligible_for_stance=False,
        matched_entities=[],
        missing_entities=missing_entities,
        matched_predicates=[],
        matched_numbers=[],
        explanation="Irrelevant: Fails to substantiate or reference the core claim assertions.",
        rejection_reasons=["Insufficient topical and entity relevance."],
    )


def determine_evidence_relevance(claim_text: str, title: str, snippet: str) -> RelevanceLevel:
    """Determines degree of direct vs partial relevance to the claim."""
    assessment = evaluate_result_relevance(claim_text, title, snippet)
    return assessment.level


def determine_evidence_strength(
    category: SourceCategory,
    quality_score: int,
    is_independent: bool,
    is_opinion: bool,
    is_clickbait: bool,
    is_primary: bool,
    relevance_level: RelevanceLevel,
) -> EvidenceStrength:
    """Classifies source evidentiary strength into STRONG, MODERATE, WEAK, or INSUFFICIENT."""
    if relevance_level in (RelevanceLevel.IRRELEVANT, RelevanceLevel.WEAK):
        return EvidenceStrength.INSUFFICIENT if relevance_level == RelevanceLevel.IRRELEVANT else EvidenceStrength.WEAK
    if is_opinion or is_clickbait or quality_score < 40:
        return EvidenceStrength.WEAK
    if not is_independent:
        return EvidenceStrength.MODERATE if quality_score >= 60 else EvidenceStrength.WEAK
    if is_primary and relevance_level == RelevanceLevel.DIRECT:
        return EvidenceStrength.STRONG
    if quality_score >= 70 and relevance_level == RelevanceLevel.DIRECT:
        return EvidenceStrength.STRONG
    if quality_score >= 50:
        return EvidenceStrength.MODERATE
    return EvidenceStrength.WEAK


def generate_ranking_reasons(
    category: SourceCategory,
    has_date: bool,
    has_quotes: bool,
    has_named_sources: bool,
    has_metrics: bool,
    is_clickbait: bool,
    is_duplicate_wire: bool,
    is_primary: bool,
    is_independent: bool,
    relevance_level: RelevanceLevel,
    wire_agency: Optional[str],
) -> List[str]:
    """Generates transparent, human-readable reasons for source ranking and weight."""
    reasons: List[str] = []
    if is_primary:
        reasons.append("Primary official documentation (government/court/institutional registry).")
    elif category == SourceCategory.FACTUAL_REPORTING:
        reasons.append("Established factual news reporting with standard editorial oversight.")
    elif category == SourceCategory.OPINION_EDITORIAL:
        reasons.append("Opinion/commentary: lower evidentiary weight for factual assertions.")
    elif category == SourceCategory.SENSATIONAL_CLICKBAIT:
        reasons.append("Sensational/clickbait framing: penalized for emotional exaggeration.")

    if has_quotes:
        reasons.append("Contains direct substantive quotations.")
    if has_named_sources:
        reasons.append("Identifies named sources or official spokespersons.")
    if has_metrics:
        reasons.append("Includes specific metrics, timestamps, or legal references.")
    if has_date:
        reasons.append("Verified publication timestamp present.")
    else:
        reasons.append("Missing publication date in metadata (slight score reduction).")

    if is_duplicate_wire:
        reasons.append(f"Duplicate wire syndication: republishes existing {wire_agency or 'wire'} feed (not counted as independent).")
    elif not is_independent:
        reasons.append("Non-independent reproduction of existing source.")
    else:
        reasons.append("Independent source.")

    if relevance_level == RelevanceLevel.DIRECT:
        reasons.append("Directly addresses the specific claim and core action.")
    elif relevance_level == RelevanceLevel.PARTIAL:
        reasons.append("Partially relevant: touches general topic but does not corroborate the specific action.")
    else:
        reasons.append("Weak relevance: peripheral topical overlap only.")

    return reasons


def calculate_source_quality(
    category: SourceCategory,
    has_date: bool,
    has_citations: bool,
    is_clickbait: bool,
    is_duplicate_wire: bool,
    has_named_sources: bool = False,
    has_metrics: bool = False,
) -> int:
    """Calculates an objective 0-100 quality score based on transparent structural criteria."""
    base_scores = {
        SourceCategory.PRIMARY_OFFICIAL: 90,
        SourceCategory.ACADEMIC_RESEARCH: 88,
        SourceCategory.FACTUAL_REPORTING: 75,
        SourceCategory.SYNDICATED_WIRE: 70,
        SourceCategory.GENERAL_WEB: 50,
        SourceCategory.OPINION_EDITORIAL: 35,
        SourceCategory.SENSATIONAL_CLICKBAIT: 20,
    }
    score = base_scores.get(category, 50)

    # Date bonus / penalty
    if has_date:
        score += 5
    else:
        score -= 5

    # Direct quotation or citation bonus
    if has_citations and category not in (SourceCategory.OPINION_EDITORIAL, SourceCategory.SENSATIONAL_CLICKBAIT):
        score += 5

    # Named sources bonus
    if has_named_sources and category not in (SourceCategory.OPINION_EDITORIAL, SourceCategory.SENSATIONAL_CLICKBAIT):
        score += 5

    # Specific metrics bonus
    if has_metrics:
        score += 3

    # Deductions
    if is_clickbait:
        score -= 25
    if is_duplicate_wire:
        score -= 20

    return max(5, min(100, score))


def evaluate_claim_evidence(claim_result: ClaimEvidenceResult) -> ClaimEvaluationSummary:
    """Evaluates all retrieved sources for a claim, enforcing independence, relevance, and transparency.

    Execution order:
    1. Relevance Evaluation: Rejects irrelevant sources before stance analysis.
    2. Provenance & Quality Scoring: Evaluates official documents, reporting, opinion, and clickbait.
    3. Independence & Wire Syndication: Enforces agency and same-domain deduplication.
    4. Cautious Synthesis: Flags UNCERTAIN if sufficient relevant independent sources are absent.
    """
    evaluated_items: List[EvaluatedEvidenceItem] = []
    filtered_irrelevant_items: List[EvaluatedEvidenceItem] = []
    seen_wire_agencies: Dict[str, str] = {}  # agency -> first evidence_id
    seen_domains: Dict[str, str] = {}        # domain -> first evidence_id
    unique_domains: Set[str] = set()

    claim_text = claim_result.claim.text if claim_result.claim else ""

    for item in claim_result.evidence_items:
        notes: List[str] = []
        demotion_reason: Optional[str] = None
        is_independent = True
        if item.domain:
            unique_domains.add(item.domain)

        # 1. RELEVANCE GATE EVALUATION (Runs FIRST)
        rel_assessment = evaluate_result_relevance(
            claim_text=claim_text,
            title=item.title,
            snippet=item.snippet,
            url=item.url,
            domain=item.domain,
            claim_obj=claim_result.claim,
        )
        relevance_level = rel_assessment.level

        # 2. Provenance evaluation
        is_primary, is_academic, provenance_category = detect_primary_provenance(item.domain)
        category = provenance_category

        if is_primary:
            notes.append("Primary official source (Government / Regulatory / Institutional domain)")
        elif is_academic:
            notes.append("Academic peer-reviewed / research repository")
        elif category == SourceCategory.FACTUAL_REPORTING:
            notes.append("Established news publication with editorial oversight")

        # 3. Opinion / Editorial detection
        is_opinion, opinion_note = detect_opinion_content(item.title, item.url, item.snippet)
        if is_opinion:
            category = SourceCategory.OPINION_EDITORIAL
            demotion_reason = "Opinion / Editorial content: deprioritized for factual claim verification."
            notes.append(opinion_note or "Identified as opinion or commentary.")

        # 4. Clickbait / Sensationalism detection
        is_clickbait, clickbait_note = detect_clickbait_sensationalism(item.title)
        if is_clickbait:
            category = SourceCategory.SENSATIONAL_CLICKBAIT
            if not demotion_reason:
                demotion_reason = "Sensational / Clickbait framing: lower factual reliability."
            notes.append(clickbait_note or "Sensational headline formatting detected.")

        # 5. Syndicated Wire Attribution & Duplicate Corroboration Detection
        is_wire, wire_agency = detect_syndicated_wire(item.title, item.snippet, item.domain)
        is_duplicate_wire = False

        if is_wire and wire_agency:
            if not is_opinion and not is_clickbait:
                category = SourceCategory.SYNDICATED_WIRE

            notes.append(f"Syndicated wire report attributed to {wire_agency}.")

            if wire_agency in seen_wire_agencies:
                is_independent = False
                is_duplicate_wire = True
                first_id = seen_wire_agencies[wire_agency]
                if not demotion_reason:
                    demotion_reason = f"Duplicate wire reporting: shares same {wire_agency} wire feed as {first_id}."
                notes.append(f"Not counted as independent corroboration (republishes same {wire_agency} report as {first_id}).")
            else:
                seen_wire_agencies[wire_agency] = item.evidence_id

        # 6. Same-Domain Duplicate Detection (Requirement 12)
        clean_dom = item.domain.lower().strip()
        if clean_dom:
            if clean_dom in seen_domains and not is_duplicate_wire:
                is_independent = False
                first_id = seen_domains[clean_dom]
                notes.append(f"Same-domain secondary source (shares domain {clean_dom} with {first_id}); not counted as independent.")
                if not demotion_reason:
                    demotion_reason = f"Duplicate publisher: multiple articles from {clean_dom} do not multiply corroboration."
            elif clean_dom not in seen_domains:
                seen_domains[clean_dom] = item.evidence_id

        # Determine independence status enum
        if is_duplicate_wire:
            independence_status = IndependenceStatus.DUPLICATE_WIRE
        elif is_wire:
            independence_status = IndependenceStatus.SYNDICATED
        elif is_independent:
            independence_status = IndependenceStatus.INDEPENDENT
        else:
            independence_status = IndependenceStatus.UNKNOWN

        # 7. Transparency metadata
        has_date = bool(item.publication_date)
        if has_date:
            notes.append(f"Publication date verified: {item.publication_date}")
        else:
            notes.append("Publication date not specified in search metadata")

        has_quotes = detect_direct_quotes(f"{item.title} {item.snippet}")
        if has_quotes:
            notes.append("Contains direct quotations or attributed statements")

        has_named, named_sources = detect_named_sources(item.title, item.snippet)
        if has_named:
            notes.append(f"Identified named sources/attributions: {', '.join(named_sources[:2])}")

        has_metrics = detect_specific_metrics(f"{item.title} {item.snippet}")
        if has_metrics:
            notes.append("Contains specific metrics, timestamps, or legal references")

        temporal_note, _ = detect_temporal_relevance(item.publication_date, claim_text)
        if temporal_note and temporal_note not in notes:
            notes.append(temporal_note)

        # 8. Quality Score & Evidentiary Strength
        quality_score = calculate_source_quality(
            category=category,
            has_date=has_date,
            has_citations=has_quotes,
            is_clickbait=is_clickbait,
            is_duplicate_wire=is_duplicate_wire,
            has_named_sources=has_named,
            has_metrics=has_metrics,
        )

        evidence_strength = determine_evidence_strength(
            category=category,
            quality_score=quality_score,
            is_independent=is_independent,
            is_opinion=is_opinion,
            is_clickbait=is_clickbait,
            is_primary=is_primary,
            relevance_level=relevance_level,
        )

        ranking_reasons = generate_ranking_reasons(
            category=category,
            has_date=has_date,
            has_quotes=has_quotes,
            has_named_sources=has_named,
            has_metrics=has_metrics,
            is_clickbait=is_clickbait,
            is_duplicate_wire=is_duplicate_wire,
            is_primary=is_primary,
            is_independent=is_independent,
            relevance_level=relevance_level,
            wire_agency=wire_agency,
        )

        # Check if source is IRRELEVANT (Filtered at Relevance Gate)
        is_relevant = (relevance_level != RelevanceLevel.IRRELEVANT)

        if not is_relevant:
            evidence_strength = EvidenceStrength.INSUFFICIENT
            is_independent = False
            demotion_reason = f"Filtered by Relevance Gate: {rel_assessment.explanation}"
            ranking_reasons = list(rel_assessment.rejection_reasons) + ranking_reasons

        evaluation = SourceEvaluation(
            evidence_id=item.evidence_id,
            category=category,
            quality_score=quality_score if is_relevant else max(5, min(quality_score, 20)),
            is_primary=is_primary if is_relevant else False,
            is_opinion=is_opinion,
            is_syndicated_wire=is_wire,
            wire_agency=wire_agency,
            is_independent=is_independent if is_relevant else False,
            transparency_notes=notes,
            demotion_reason=demotion_reason,
            independence_status=independence_status,
            relevance_level=relevance_level,
            evidence_strength=evidence_strength,
            has_named_sources=has_named,
            has_direct_quotes=has_quotes,
            has_specific_metrics=has_metrics,
            ranking_reasons=ranking_reasons,
            temporal_note=temporal_note,
            relevance_assessment=rel_assessment,
            is_relevant=is_relevant,
        )

        evaluated_item = EvaluatedEvidenceItem(evidence=item, evaluation=evaluation)

        if is_relevant:
            evaluated_items.append(evaluated_item)
        else:
            filtered_irrelevant_items.append(evaluated_item)

    # Sort evaluated relevant evidence: highest quality score first
    evaluated_items.sort(key=lambda x: -x.evaluation.quality_score)
    filtered_irrelevant_items.sort(key=lambda x: -x.evaluation.quality_score)

    # Calculate aggregate summary metrics strictly on relevant items
    primary_count = sum(1 for e in evaluated_items if e.evaluation.is_primary)
    independent_count = sum(1 for e in evaluated_items if e.evaluation.is_independent and not e.evaluation.is_opinion)
    opinion_count = sum(1 for e in evaluated_items if e.evaluation.is_opinion)
    duplicate_wire_count = sum(1 for e in evaluated_items if not e.evaluation.is_independent)
    syndicated_count = sum(1 for e in evaluated_items if e.evaluation.is_syndicated_wire)
    filtered_irrelevant_count = len(filtered_irrelevant_items)

    has_sufficient = (primary_count >= 1) or (independent_count >= 2)

    if has_sufficient:
        assessment_note = (
            f"Sufficient evidence quality: {primary_count} primary and {independent_count} "
            "independent factual source(s) available for evaluation."
        )
    elif len(evaluated_items) == 0 and filtered_irrelevant_count > 0:
        assessment_note = (
            f"Search results did not contain sufficiently relevant evidence for this claim ({filtered_irrelevant_count} irrelevant result(s) filtered). (Result = UNCERTAIN Candidate / INSUFFICIENT_EVIDENCE)"
        )
    elif len(evaluated_items) == 0:
        assessment_note = (
            "No evidence retrieved: Insufficient information to verify claim. (Result = UNCERTAIN Candidate)"
        )
    else:
        assessment_note = (
            "Insufficient independent evidence: Sources are limited, duplicates of the same wire feed, "
            "or predominantly opinion pieces. (Result = UNCERTAIN Candidate)"
        )

    return ClaimEvaluationSummary(
        claim_id=claim_result.claim.claim_id,
        evaluated_items=evaluated_items,
        filtered_irrelevant_items=filtered_irrelevant_items,
        primary_source_count=primary_count,
        independent_source_count=independent_count,
        opinion_source_count=opinion_count,
        duplicate_wire_count=duplicate_wire_count,
        filtered_irrelevant_count=filtered_irrelevant_count,
        has_sufficient_independent_evidence=has_sufficient,
        quality_assessment_note=assessment_note,
        unique_domains=unique_domains,
        unique_domain_count=len(unique_domains),
        syndicated_source_count=syndicated_count,
    )


def evaluate_all_claims(claims_evidence: List[ClaimEvidenceResult]) -> List[ClaimEvaluationSummary]:
    """Evaluates source quality across all claims in the session."""
    return [evaluate_claim_evidence(cr) for cr in claims_evidence]
