"""Search provider abstraction and implementations for Evidence Retrieval.

Defines a unified interface for web search providers, standardized search
result models, a deterministic MockSearchProvider for testing and offline demo,
and a real TavilySearchProvider for live web evidence retrieval.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import re
import os
import requests


class SearchStatus(str, Enum):
    """Execution status for a search provider query."""

    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    PROVIDER_UNCONFIGURED = "PROVIDER_UNCONFIGURED"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    API_FAILURE = "API_FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_RESPONSE = "INVALID_RESPONSE"


@dataclass
class SearchResult:
    """Represents an individual search result item returned by a provider."""

    title: str
    url: str
    domain: str
    snippet: str
    publication_date: Optional[str] = None
    rank: int = 1

    def __post_init__(self):
        # Ensure domain is properly populated and lowercase
        if not self.domain and self.url:
            self.domain = extract_domain_from_url(self.url)
        else:
            self.domain = self.domain.lower()


@dataclass
class SearchResponse:
    """Standardized response from any SearchProvider."""

    status: SearchStatus
    results: List[SearchResult] = field(default_factory=list)
    error_message: Optional[str] = None
    provider_name: str = "Unknown"


def extract_domain_from_url(url: str) -> str:
    """Extracts the clean, normalized domain name from a URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(":")[0]
        # Remove common www. prefix for consistent grouping
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def load_env_file(filepath: str = ".env") -> None:
    """Lightweight loader for .env key-value pairs into os.environ.

    Avoids external dependencies while allowing local configuration.
    Does not overwrite existing environment variables.
    """
    if not os.path.isfile(filepath):
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    key, val = stripped.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        # Silently pass if unable to read .env
        pass


class SearchProvider(ABC):
    """Abstract base class for all search provider implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the search provider."""
        pass

    @property
    def is_real(self) -> bool:
        """Whether this provider executes genuine external web search."""
        return False

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Executes a search query and returns a standardized SearchResponse.

        Args:
            query: The search query string.
            max_results: Maximum number of sources to return.

        Returns:
            SearchResponse containing status, results, and provider metadata.
        """
        pass


class TavilySearchProvider(SearchProvider):
    """Real web search provider integrated with the Tavily Search API.

    Requires an API key set via parameter or TAVILY_API_KEY environment variable.
    Never logs or exposes API keys.
    """

    DEFAULT_ENDPOINT = "https://api.tavily.com/search"
    DEFAULT_TIMEOUT = 10

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            load_env_file()
            self._api_key = os.environ.get("TAVILY_API_KEY", "").strip()
            if not self._api_key or self._api_key == "your_api_key_here":
                try:
                    import streamlit as st
                    if hasattr(st, "secrets") and "TAVILY_API_KEY" in st.secrets:
                        self._api_key = str(st.secrets["TAVILY_API_KEY"]).strip()
                except Exception:
                    pass
        self.endpoint = endpoint
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "TavilySearchProvider"

    @property
    def is_real(self) -> bool:
        return True

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Executes search against Tavily API."""
        q_clean = query.strip()
        if not q_clean:
            return SearchResponse(
                status=SearchStatus.NO_RESULTS,
                results=[],
                error_message="Query is empty.",
                provider_name=self.name,
            )

        # 1. Missing or unconfigured API key check
        if not self._api_key:
            return SearchResponse(
                status=SearchStatus.PROVIDER_UNCONFIGURED,
                results=[],
                error_message=(
                    "Tavily API key is missing. Set the TAVILY_API_KEY environment variable "
                    "or add it to your local .env file to enable real web search."
                ),
                provider_name=self.name,
            )

        payload = {
            "api_key": self._api_key,
            "query": q_clean,
            "search_depth": "basic",
            "max_results": max(1, min(max_results, 10)),
            "include_answer": False,
        }

        # 2. Network execution with granular error mapping
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            return SearchResponse(
                status=SearchStatus.NETWORK_FAILURE,
                results=[],
                error_message=f"Tavily search API request timed out after {self.timeout} seconds.",
                provider_name=self.name,
            )
        except requests.exceptions.ConnectionError:
            return SearchResponse(
                status=SearchStatus.NETWORK_FAILURE,
                results=[],
                error_message="Failed to connect to Tavily search API. Check your internet connection.",
                provider_name=self.name,
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                status=SearchStatus.NETWORK_FAILURE,
                results=[],
                error_message=f"Network request to search provider failed: {type(e).__name__}",
                provider_name=self.name,
            )

        # 3. HTTP status code evaluation
        if resp.status_code == 401 or resp.status_code == 403:
            return SearchResponse(
                status=SearchStatus.API_FAILURE,
                results=[],
                error_message="Invalid or unauthorized Tavily API key (HTTP 401/403).",
                provider_name=self.name,
            )

        if resp.status_code == 429:
            return SearchResponse(
                status=SearchStatus.RATE_LIMITED,
                results=[],
                error_message="Tavily search API rate limit exceeded (HTTP 429). Please try again later.",
                provider_name=self.name,
            )

        if resp.status_code >= 500:
            return SearchResponse(
                status=SearchStatus.API_FAILURE,
                results=[],
                error_message=f"Tavily search API server error (HTTP {resp.status_code}).",
                provider_name=self.name,
            )

        if resp.status_code != 200:
            return SearchResponse(
                status=SearchStatus.API_FAILURE,
                results=[],
                error_message=f"Tavily search API request failed with status HTTP {resp.status_code}.",
                provider_name=self.name,
            )

        # 4. JSON parsing and structure validation
        try:
            data = resp.json()
        except Exception:
            return SearchResponse(
                status=SearchStatus.INVALID_RESPONSE,
                results=[],
                error_message="Tavily search API returned a malformed non-JSON response.",
                provider_name=self.name,
            )

        if not isinstance(data, dict):
            return SearchResponse(
                status=SearchStatus.INVALID_RESPONSE,
                results=[],
                error_message="Unexpected payload format from Tavily search API (expected JSON object).",
                provider_name=self.name,
            )

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            return SearchResponse(
                status=SearchStatus.INVALID_RESPONSE,
                results=[],
                error_message="Tavily search API payload missing 'results' list.",
                provider_name=self.name,
            )

        if not raw_results:
            return SearchResponse(
                status=SearchStatus.NO_RESULTS,
                results=[],
                error_message="No external search results found for the query.",
                provider_name=self.name,
            )

        # 5. Result mapping into SearchResult models
        mapped_results: List[SearchResult] = []
        for idx, item in enumerate(raw_results[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled Source").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or "").strip()
            pub_date = item.get("published_date")
            if pub_date:
                # Retain clean date substring if formatted with ISO timestamp
                pub_date_str = str(pub_date).split("T")[0]
            else:
                pub_date_str = None

            mapped_results.append(
                SearchResult(
                    title=title,
                    url=url,
                    domain=extract_domain_from_url(url),
                    snippet=snippet,
                    publication_date=pub_date_str,
                    rank=idx,
                )
            )

        if not mapped_results:
            return SearchResponse(
                status=SearchStatus.NO_RESULTS,
                results=[],
                error_message="No valid results extracted from Tavily response.",
                provider_name=self.name,
            )

        return SearchResponse(
            status=SearchStatus.SUCCESS,
            results=mapped_results,
            provider_name=self.name,
        )


class MockSearchProvider(SearchProvider):
    """Deterministic mock search provider for testing and offline demonstration.

    Supports pre-registered query mappings, simulated error statuses, and
    realistic default fallback sources for news/economic topics.
    """

    def __init__(
        self,
        name: str = "MockSearchProvider",
        responses: Optional[Dict[str, List[SearchResult]]] = None,
        force_status: Optional[SearchStatus] = None,
        error_message: Optional[str] = None,
    ):
        self._name = name
        self.responses: Dict[str, List[SearchResult]] = responses or {}
        self.force_status = force_status
        self.error_message = error_message
        self.recorded_queries: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_real(self) -> bool:
        return False

    def set_mock_results(self, query: str, results: List[SearchResult]) -> None:
        """Registers specific results for an exact or substring query match."""
        self.responses[query.lower().strip()] = results

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Executes search against mock data or forced behaviors."""
        q_clean = query.strip()
        self.recorded_queries.append(q_clean)

        # Handle forced error simulations
        if self.force_status:
            if self.force_status == SearchStatus.SUCCESS:
                pass
            elif self.force_status == SearchStatus.NO_RESULTS:
                return SearchResponse(
                    status=SearchStatus.NO_RESULTS,
                    results=[],
                    error_message=None,
                    provider_name=self.name,
                )
            else:
                return SearchResponse(
                    status=self.force_status,
                    results=[],
                    error_message=self.error_message or f"Simulated {self.force_status.value} error.",
                    provider_name=self.name,
                )

        q_lower = q_clean.lower()

        # 1. Check exact, substring, or token-set match in registered responses
        q_tokens = set(re.findall(r"\w+", q_lower))
        for key, mock_items in self.responses.items():
            key_lower = key.lower().strip()
            if key_lower == "*" or key_lower in q_lower or q_lower in key_lower:
                trimmed = mock_items[:max_results]
                return SearchResponse(
                    status=SearchStatus.SUCCESS if trimmed else SearchStatus.NO_RESULTS,
                    results=trimmed,
                    provider_name=self.name,
                )
            key_tokens = set(re.findall(r"\w+", key_lower))
            if key_tokens and (key_tokens.issubset(q_tokens) or q_tokens.issubset(key_tokens)):
                trimmed = mock_items[:max_results]
                return SearchResponse(
                    status=SearchStatus.SUCCESS if trimmed else SearchStatus.NO_RESULTS,
                    results=trimmed,
                    provider_name=self.name,
                )

        # 2. Realistic contextual fallback generator based on keywords
        canned_results = self._generate_contextual_canned_results(q_lower)
        if canned_results:
            trimmed = canned_results[:max_results]
            return SearchResponse(
                status=SearchStatus.SUCCESS,
                results=trimmed,
                provider_name=self.name,
            )

        # 3. Default generic no results if no match
        return SearchResponse(
            status=SearchStatus.NO_RESULTS,
            results=[],
            error_message="No external search results found for query.",
            provider_name=self.name,
        )

    def _generate_contextual_canned_results(self, q_lower: str) -> List[SearchResult]:
        """Generates realistic external search results based on common news keywords."""
        results: List[SearchResult] = []

        # Monetary / Banking / RBI queries
        if any(term in q_lower for term in ["repo", "rbi", "rate", "reserve bank", "interest", "monetary"]):
            results.append(
                SearchResult(
                    title="Monetary Policy Statement: Policy Repo Rate Remains at 6.50%",
                    url="https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=57241",
                    domain="rbi.org.in",
                    snippet=(
                        "The Monetary Policy Committee (MPC) at its meeting today decided to keep "
                        "the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent."
                    ),
                    publication_date="2024-02-08",
                    rank=1,
                )
            )
            results.append(
                SearchResult(
                    title="RBI keeps repo rate unchanged at 6.5%: What MPC decisions mean for home loans",
                    url="https://www.thehindu.com/business/Economy/rbi-monetary-policy-repo-rate-decision/article67824151.ece",
                    domain="thehindu.com",
                    snippet=(
                        "The Reserve Bank of India kept the benchmark repo rate unchanged at 6.5% for the sixth "
                        "consecutive time, maintaining its focus on bringing inflation down towards 4%."
                    ),
                    publication_date="2024-02-08",
                    rank=2,
                )
            )
            results.append(
                SearchResult(
                    title="RBI MPC Meeting Highlights: Key announcements and repo rate status",
                    url="https://timesofindia.indiatimes.com/business/india-business/rbi-monetary-policy-meeting-highlights/articleshow/107513204.cms",
                    domain="timesofindia.indiatimes.com",
                    snippet=(
                        "RBI Governor Shaktikanta Das announced that the monetary policy committee voted "
                        "with a 5:1 majority to hold the policy repo rate steady at 6.5%."
                    ),
                    publication_date="2024-02-08",
                    rank=3,
                )
            )
            results.append(
                SearchResult(
                    title="India Central Bank Leaves Benchmark Rate Unchanged at 6.5%",
                    url="https://www.bloomberg.com/news/articles/2024-02-08/india-central-bank-leaves-benchmark-rate-unchanged-at-6-5",
                    domain="bloomberg.com",
                    snippet=(
                        "India's central bank kept its key interest rate at 6.5% on Thursday, signaling "
                        "borrowing costs will stay high until inflation is durable at target."
                    ),
                    publication_date="2024-02-08",
                    rank=4,
                )
            )
            return results

        # Space / NASA / ISRO queries
        if any(term in q_lower for term in ["nasa", "space", "moon", "lunar", "isro", "orbit", "satellite"]):
            results.append(
                SearchResult(
                    title="NASA Updates Artemis Lunar Exploration Program and Timeline",
                    url="https://www.nasa.gov/news-release/nasa-shares-progress-toward-early-artemis-moon-missions-with-crew/",
                    domain="nasa.gov",
                    snippet=(
                        "NASA provided updates on its Artemis campaign, noting the planned crewed lunar landing mission "
                        "and international scientific collaborations under the multi-billion dollar program."
                    ),
                    publication_date="2024-01-09",
                    rank=1,
                )
            )
            results.append(
                SearchResult(
                    title="Space Exploration and Lunar Landings: Reuters Special Report",
                    url="https://www.reuters.com/technology/space/nasa-artemis-moon-landing-schedule-2024-01-09/",
                    domain="reuters.com",
                    snippet=(
                        "NASA officials announced schedule revisions for lunar missions under the Artemis program, "
                        "prioritizing astronaut safety and vehicle testing."
                    ),
                    publication_date="2024-01-09",
                    rank=2,
                )
            )
            results.append(
                SearchResult(
                    title="Lunar Missions and Spaceflight Updates",
                    url="https://www.bbc.com/news/science-environment-67926941",
                    domain="bbc.com",
                    snippet=(
                        "International space agencies continue preparations for deep space exploration and lunar orbit payloads."
                    ),
                    publication_date="2024-01-10",
                    rank=3,
                )
            )
            return results

        # Default multi-source news generator for any general claim
        sanitized = re.sub(r"[^\w\s]", "", q_clean).strip()
        results.append(
            SearchResult(
                title=f"Official Press Release: {sanitized.title()}",
                url=f"https://pib.gov.in/PressReleasePage.aspx?PRID={abs(hash(q_clean)) % 1000000}",
                domain="pib.gov.in",
                snippet=(
                    f"Official government briefing regarding {sanitized}. "
                    "Relevant departments released factual details and administrative guidance."
                ),
                publication_date="2024-03-01",
                rank=1,
            )
        )
        results.append(
            SearchResult(
                title=f"News Coverage & Fact Sheet: {sanitized.title()}",
                url=f"https://www.reuters.com/world/news-overview-{abs(hash(q_clean)) % 100000}",
                domain="reuters.com",
                snippet=(
                    f"Independent reporting regarding {sanitized}. "
                    "Correspondents verified statements with official spokespersons."
                ),
                publication_date="2024-03-02",
                rank=2,
            )
        )
        results.append(
            SearchResult(
                title=f"Detailed Report: {sanitized.title()}",
                url=f"https://www.bbc.com/news/world-report-{abs(hash(q_clean)) % 100000}",
                domain="bbc.com",
                snippet=(
                    f"BBC verification coverage: An examination of reports concerning {sanitized}."
                ),
                publication_date="2024-03-02",
                rank=3,
            )
        )
        return results


def is_real_search_configured() -> bool:
    """Checks whether real web search (Tavily API key) is configured."""
    load_env_file()
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key or key == "your_api_key_here":
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "TAVILY_API_KEY" in st.secrets:
                key = str(st.secrets["TAVILY_API_KEY"]).strip()
        except Exception:
            pass
    return bool(key and key != "your_api_key_here")


def get_default_search_provider(prefer_real: bool = True) -> SearchProvider:
    """Returns the active search provider.

    If prefer_real is True and TAVILY_API_KEY is configured in the environment,
    returns TavilySearchProvider. Otherwise returns MockSearchProvider for safe,
    deterministic offline operation.
    """
    if prefer_real and is_real_search_configured():
        return TavilySearchProvider()
    return MockSearchProvider()
