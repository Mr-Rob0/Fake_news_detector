"""Evidence retrieval module for the AI News Verification System (Phase 5).

Generates focused search queries from extracted factual claims, queries external
search providers, prioritizes authoritative and independent sources, enforces
source diversity, and organizes structured evidence metadata with claim traceability.

Does NOT evaluate claims, score veracity, or produce verdicts in this phase.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from urllib.parse import urlparse
import re

from src.claim_extractor import Claim, HINDI_STOPWORDS
from src.search_provider import (
    SearchProvider,
    SearchResult,
    SearchResponse,
    SearchStatus,
    extract_domain_from_url,
)

# Stopwords and generic assertion verbs to remove from claims during query generation
QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "as", "at", "by",
    "for", "from", "in", "into", "of", "off", "on", "onto", "out", "over",
    "to", "up", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "there", "which",
    "who", "whom", "whose", "what", "where", "when", "how", "why", "said",
    "says", "according", "stated", "told", "announced", "also", "increased",
    "decreased", "held", "keeps", "reported", "confirmed", "revealed",
    "समाचार", "खबर", "रिपोर्ट", "अनुसार", "बारे"
} | HINDI_STOPWORDS

# Acronym / synonym normalization map to produce concise, standard search terms
ENTITY_SYNONYMS = {
    "reserve bank of india": "RBI",
    "federal reserve": "Fed",
    "world health organization": "WHO",
    "united nations": "UN",
    "supreme court of india": "Supreme Court",
    "national aeronautics and space administration": "NASA",
    "indian space research organisation": "ISRO",
    "indian space research organization": "ISRO",
    "securities and exchange board of india": "SEBI",
}

# Known authoritative domain registries
GOVERNMENT_DOMAINS = {
    "gov", "gov.in", "nic.in", "pib.gov.in", "gov.uk", "whitehouse.gov",
    "state.gov", "treasury.gov", "parliament.uk", "sansad.in", "india.gov.in"
}

INSTITUTIONAL_DOMAINS = {
    "rbi.org.in", "who.int", "un.org", "nasa.gov", "isro.gov.in",
    "worldbank.org", "imf.org", "sec.gov", "cdc.gov", "nih.gov",
    "wto.org", "wmo.int", "iso.org", "edu", "ac.in", "ac.uk"
}

REPUTABLE_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "indianexpress.com", "bloomberg.com", "wsj.com", "nytimes.com",
    "ft.com", "economist.com", "afp.com", "theguardian.com",
    "hindustantimes.com", "timesofindia.indiatimes.com", "nature.com",
    "sciencemag.org", "aljazeera.com", "dw.com", "france24.com"
}


@dataclass
class SearchQuery:
    """Represents a focused search query derived from a factual claim."""

    claim_id: str
    query_text: str
    preserved_keywords: List[str] = field(default_factory=list)
    additional_queries: List[str] = field(default_factory=list)
    query_types: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """Structured external evidence item associated with a specific claim."""

    evidence_id: str
    claim_id: str
    title: str
    url: str
    domain: str
    snippet: str
    publication_date: Optional[str]
    search_query: str
    source_rank: int
    retrieval_status: SearchStatus
    is_priority_source: bool = False
    priority_tier: str = "General"


@dataclass
class ClaimEvidenceResult:
    """Contains all evidence retrieved and metadata for an individual claim."""

    claim: Claim
    search_query: SearchQuery
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    status: SearchStatus = SearchStatus.SUCCESS
    error_message: Optional[str] = None
    provider_name: str = "Unknown"

    @property
    def source_count(self) -> int:
        return len(self.evidence_items)


def generate_search_query(claim: Claim) -> SearchQuery:
    """Generates concise, high-signal web search queries from a factual claim.

    Preserves important entities, numbers, percentages, dates, locations, and
    action predicates while avoiding naive token truncation. Generates complementary
    query variants (entity+event, exact phrasing, primary source, verification).

    Args:
        claim: The Claim object to generate queries for.

    Returns:
        SearchQuery object containing focused primary query string and complementary variants.
    """
    text = claim.text.strip()

    # Step 1: Normalize common multi-word entity names to compact standard acronyms
    normalized_text = text
    for full_name, acronym in ENTITY_SYNONYMS.items():
        pattern = re.compile(re.escape(full_name), re.IGNORECASE)
        normalized_text = pattern.sub(acronym, normalized_text)

    # Step 2: Extract structured components if available or via quick decomposition
    decomposed = getattr(claim, "decomposed", None)
    if not decomposed:
        from src.claim_extractor import decompose_claim_text
        decomposed = decompose_claim_text(normalized_text)

    # Step 3: Extract and categorize high-priority tokens
    # Group A: Named entities (proper nouns, capitalized sequences, acronyms)
    named_entities: List[str] = []
    if decomposed.named_entities:
        named_entities = list(decomposed.named_entities)
    else:
        # Fallback to uppercase tokens
        caps = re.findall(r"\b[A-Z][a-zA-Z0-9\.\-]+\b", normalized_text)
        for c in caps:
            if c not in named_entities and c.lower() not in QUERY_STOPWORDS:
                named_entities.append(c)

    # Group B: Specific Numbers & Percentages
    numeric_tokens = re.findall(r"(?:[\$€£₹]\s*)?\b\d+(?:\.\d+)?%?", normalized_text)
    # Check for word percentages
    if "percent" in normalized_text.lower():
        num_pct = re.findall(r"\b\d+\s+percent\b", normalized_text, re.IGNORECASE)
        numeric_tokens.extend(num_pct)

    # Group C: Dates & Years
    date_tokens = re.findall(r"\b(?:19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", normalized_text, re.IGNORECASE)

    # Group D: Location
    location_tokens: List[str] = []
    if decomposed.location:
        loc_words = [w for w in decomposed.location.split() if w.lower() not in QUERY_STOPWORDS and w.lower() not in ("city", "near", "across")]
        location_tokens.extend(loc_words)

    # Group E: Key Action Verbs / Predicates
    action_words: List[str] = []
    if decomposed.action:
        action_words.append(decomposed.action)
    action_matches = re.findall(
        r"\b(?:detained|arrested|launched|increased|decreased|approved|rejected|"
        r"reported|confirmed|found|announced|ordered|banned|ruled|beaten|punished|"
        r"killed|signed|passed|held|kept|used|repo|satellite)\b",
        normalized_text,
        re.IGNORECASE,
    )
    for aw in action_matches:
        if aw.lower() not in [a.lower() for a in action_words]:
            action_words.append(aw)

    # Group F: Core concept words
    concept_words: List[str] = []
    if decomposed.core_concepts:
        concept_words = [
            cw for cw in decomposed.core_concepts
            if (len(cw) >= 2 if any('\u0900' <= c <= '\u097F' for c in cw) else len(cw) >= 3)
            and cw.lower() not in QUERY_STOPWORDS
        ]

    # Combine in priority order: Named Entities -> Numbers/Percentages -> Dates -> Actions -> Locations -> Concepts
    primary_terms: List[str] = []
    seen_words: Set[str] = set()

    for term_list in [named_entities, numeric_tokens, date_tokens, action_words, location_tokens, concept_words]:
        for item in term_list:
            clean_item = item.strip(".,;:\"'()?![]{}।॥")
            # Single-character tokens must NEVER become search terms
            if not clean_item or len(clean_item) < 2:
                continue
            item_words = [
                w.lower() for w in clean_item.split()
                if w.lower() not in QUERY_STOPWORDS and len(w) >= 2
            ]
            if not item_words:
                continue
            # Avoid adding if all constituent words are already in seen_words
            if all(w in seen_words for w in item_words):
                continue
            for w in item_words:
                seen_words.add(w)
            primary_terms.append(clean_item)

    # Cap primary query to top 9 high-signal terms
    if primary_terms:
        selected_primary = primary_terms[:9]
    else:
        # Fallback to standard tokens (Unicode-aware, excluding single characters and stopwords)
        raw_tokens = re.findall(r"[\u0900-\u097F\w\.\%\$€£₹]+", normalized_text)
        selected_primary = [
            t for t in raw_tokens
            if len(t) >= 2 and t.lower() not in QUERY_STOPWORDS
        ][:7]

    primary_query_str = " ".join(selected_primary)

    # Step 4: Generate Complementary Queries
    # Query 2: Exact/Core phrasing query (Entities + Action + Numbers + Location)
    core_items = [t for t in selected_primary if t in named_entities or t in action_words or t in numeric_tokens or t in location_tokens]
    if len(core_items) < 2 and len(selected_primary) >= 2:
        core_items = selected_primary[:4]
    query_exact = " ".join(core_items) if core_items else primary_query_str

    # Query 3: Primary source / official documentation query
    source_terms = [t for t in (named_entities + date_tokens + action_words + location_tokens)[:5]]
    if not source_terms and selected_primary:
        source_terms = selected_primary[:3]
    query_primary_source = " ".join(source_terms + ["official", "report"])

    # Query 4: Fact check / verification query
    query_verification = " ".join(source_terms + ["fact check", "verification"])

    additional = [query_exact, query_primary_source, query_verification]
    # Deduplicate while preserving order
    clean_additional = []
    for q in additional:
        if q and q != primary_query_str and q not in clean_additional:
            clean_additional.append(q)

    query_types = {
        "entity_event_date": primary_query_str,
        "core_factual": query_exact,
        "primary_source": query_primary_source,
        "fact_check": query_verification,
    }

    return SearchQuery(
        claim_id=claim.claim_id,
        query_text=primary_query_str,
        preserved_keywords=selected_primary,
        additional_queries=clean_additional,
        query_types=query_types,
    )


def classify_source_priority(domain: str) -> Tuple[bool, str]:
    """Classifies domain into priority hierarchy for news verification.

    Returns:
        Tuple of (is_priority: bool, tier_name: str)
    """
    clean_domain = domain.lower().strip()

    # Tier 1: Official government sources
    if any(clean_domain.endswith("." + gd) or clean_domain == gd for gd in GOVERNMENT_DOMAINS):
        return True, "Government / Official"

    # Tier 2: Institutional, regulatory, and established multilateral bodies
    if any(clean_domain.endswith("." + inst) or clean_domain == inst for inst in INSTITUTIONAL_DOMAINS):
        return True, "Institutional / Regulatory"

    # Tier 3: Established reputable news publications
    if any(clean_domain.endswith("." + news) or clean_domain == news for news in REPUTABLE_NEWS_DOMAINS):
        return True, "Reputable News"

    return False, "General"


def retrieve_evidence_for_claim(
    claim: Claim,
    provider: SearchProvider,
    max_sources: int = 3,
    max_per_domain: int = 2,
) -> ClaimEvidenceResult:
    """Retrieves external evidence for an individual claim using the given SearchProvider.

    Enforces source limits, domain diversity, traceability, and priority classification.
    """
    search_query = generate_search_query(claim)

    # Query the provider
    try:
        response: SearchResponse = provider.search(
            query=search_query.query_text,
            max_results=max_sources * 2,  # Request extra to allow domain diversity filtering
        )
    except Exception as e:
        return ClaimEvidenceResult(
            claim=claim,
            search_query=search_query,
            evidence_items=[],
            status=SearchStatus.NETWORK_FAILURE,
            error_message=f"Search provider unexpected error: {str(e)}",
            provider_name=provider.name,
        )

    # Handle error or empty response from provider
    if response.status != SearchStatus.SUCCESS:
        return ClaimEvidenceResult(
            claim=claim,
            search_query=search_query,
            evidence_items=[],
            status=response.status,
            error_message=response.error_message,
            provider_name=provider.name,
        )

    # Process and filter results with diversity and priority
    evidence_items: List[EvidenceItem] = []
    seen_urls: Set[str] = set()
    domain_counts: Dict[str, int] = {}

    for idx, res in enumerate(response.results, start=1):
        if len(evidence_items) >= max_sources:
            break

        clean_url = res.url.strip()
        if not clean_url or clean_url in seen_urls:
            continue

        domain = res.domain or extract_domain_from_url(clean_url)

        # Enforce domain diversity (avoid flooding with same domain)
        current_domain_count = domain_counts.get(domain, 0)
        if current_domain_count >= max_per_domain:
            continue

        seen_urls.add(clean_url)
        domain_counts[domain] = current_domain_count + 1

        is_priority, tier = classify_source_priority(domain)
        evidence_id = f"E{len(evidence_items) + 1:03d}"

        evidence_items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                claim_id=claim.claim_id,
                title=res.title,
                url=clean_url,
                domain=domain,
                snippet=res.snippet,
                publication_date=res.publication_date,
                search_query=search_query.query_text,
                source_rank=res.rank or idx,
                retrieval_status=SearchStatus.SUCCESS,
                is_priority_source=is_priority,
                priority_tier=tier,
            )
        )

    # Check if results survived deduplication
    final_status = SearchStatus.SUCCESS if evidence_items else SearchStatus.NO_RESULTS
    error_msg = None if evidence_items else "No valid sources remained after diversity filtering."

    return ClaimEvidenceResult(
        claim=claim,
        search_query=search_query,
        evidence_items=evidence_items,
        status=final_status,
        error_message=error_msg,
        provider_name=provider.name,
    )


def retrieve_evidence_for_claims(
    claims: List[Claim],
    provider: SearchProvider,
    max_sources_per_claim: int = 3,
) -> List[ClaimEvidenceResult]:
    """Retrieves external evidence for a batch of extracted claims.

    Args:
        claims: List of Claim objects.
        provider: An active SearchProvider instance.
        max_sources_per_claim: Maximum evidence sources to retain per claim.

    Returns:
        List of ClaimEvidenceResult objects, maintaining 1-to-1 correspondence with claims.
    """
    results: List[ClaimEvidenceResult] = []
    for claim in claims:
        claim_result = retrieve_evidence_for_claim(
            claim=claim,
            provider=provider,
            max_sources=max_sources_per_claim,
        )
        results.append(claim_result)
    return results
