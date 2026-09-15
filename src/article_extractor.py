"""Article extraction component for the AI News Verification System.

Fetches web pages safely and extracts the main readable article content,
title, domain, and metadata without executing JavaScript or large browser engines.
Features granular status classification, defensive DOM traversal, related-content
filtering, and text normalization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse
import re
import requests
from bs4 import BeautifulSoup, Tag

# Minimum substantive length for extracted article text to be considered viable
MIN_ARTICLE_WORDS = 30
MIN_ARTICLE_CHARS = 100

# Default request timeout in seconds
DEFAULT_TIMEOUT = 10

# Standard headers to identify user agent politely and avoid basic blocks
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 "
        "(AI News Verification System; Academic Research)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags to completely strip from HTML before parsing readable text
BOILERPLATE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside",
    "form", "noscript", "svg", "iframe", "button", "dialog"
]

# Patterns in class or id names that typically indicate non-article boilerplate or related content
BOILERPLATE_CLASS_ID_PATTERNS = re.compile(
    r"(?:cookie|advertisement|ad-container|ad-box|ad-wrapper|ad-slot|sponsored|banner|newsletter|social-share|"
    r"share-tools|sidebar|comment|disclaimer|promo|widget|signup|taboola|outbrain|navigation|menu[-_]?container|"
    r"related[-_]?(?:articles?|stories|posts?|links?|content|box|news|item)|"
    r"recommend(?:ed|ation)?[-_]?(?:articles?|stories|posts?|box)|"
    r"trending[-_]?(?:articles?|stories|posts?|box)?|"
    r"read[-_]?also|also[-_]?read|more[-_]?(?:stories|articles)|"
    r"tags?[-_]?(?:list|container|cloud)?|author[-_]?(?:bio|profile|box))",
    re.IGNORECASE
)

# Patterns identifying in-paragraph related story links, photo credits, and promotional calls
UNWANTED_CONTENT_PATTERNS = re.compile(
    r"^\s*(?:"
    # Also Read / Related Articles prefixes (English)
    r"(?:also\s+read|read\s+also|read\s+more|must\s+read|see\s+also|related\s+(?:news|stories|articles?|posts?)|more\s+on\s+this|suggested\s+reading)\s*[:\-\u2013\u2014|]|"
    # Also Read prefixes (Hindi & multilingual)
    r"(?:यह\s*भी\s*प(?:ढ़ें|ढ़े)|ये\s*भी\s*प(?:ढ़ें|ढ़े)|संबंधित\s*(?:खबरें|समाचार)|आगे\s*प(?:ढ़ें|ढ़े)|यह\s*भी\s*देखें)\s*[:\-\u2013\u2014|]|"
    # Standalone photo credits
    r"(?:photo\s*[:\-]|image\s*credit\s*[:\-]|source\s*[:\-]|getty\s*images|pti\s*photo|reuters\s*photo|afp\s*photo|file\s*photo|representative\s*(?:image|photo)|फोटो\s*[:\-]|फोटो\s*साभार|फाइल\s*फोटो)|"
    # Social / newsletter subscription calls / sponsored
    r"(?:click\s+here\s+to\s+join|follow\s+us\s+on|download\s+(?:the\s+)?app|join\s+(?:our\s+)?whatsapp\s+channel|हमारे\s+व्हाट्सएप\s+चैनल\s+से\s+जुड़ें|sponsored\s+content|promoted\s+story)|"
    # Disclaimers, developing story notices, and publisher boilerplates
    r"(?:disclaimer\s*[:\-]|editor'?s?\s*note\s*[:\-]|author'?s?\s*note\s*[:\-]|"
    r"यह\s*(?:समाचार|खबर)\s*(?:प्रारम्भिक|प्रारंभिक|पुष्टि|एजेंसी)|"
    r"पाठकों\s*से\s*अनुरोध|ताजा\s*जानकारी\s*के\s*लिए)"
    r")",
    re.IGNORECASE
)


class ExtractionStatus(str, Enum):
    """Categorized status codes for article fetching and extraction."""
    SUCCESS = "SUCCESS"
    ACCESS_DENIED = "ACCESS_DENIED"              # HTTP 401, 403 (anti-bot / paywalls)
    NOT_FOUND = "NOT_FOUND"                      # HTTP 404
    RATE_LIMITED = "RATE_LIMITED"                # HTTP 429
    SERVER_ERROR = "SERVER_ERROR"                # Actual HTTP 5xx responses from target server
    CLIENT_ERROR = "CLIENT_ERROR"                # Other HTTP 4xx
    NETWORK_ERROR = "NETWORK_ERROR"              # Connection, DNS, or SSL failure
    TIMEOUT = "TIMEOUT"                          # Request timeout
    NON_HTML = "NON_HTML"                        # Non-HTML content type (e.g. PDF)
    EMPTY_PAGE = "EMPTY_PAGE"                    # Empty response body
    NO_ARTICLE_CONTENT = "NO_ARTICLE_CONTENT"    # Reachable page, but no article body found
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"      # Text found, but below substantive threshold
    INVALID_URL = "INVALID_URL"                  # Unsupported scheme or malformed URL
    INTERNAL_ERROR = "INTERNAL_ERROR"            # Application-side parser or processing exception


@dataclass
class ExtractedArticle:
    """Represents successfully extracted article content and metadata."""
    title: str
    text: str
    domain: str
    final_url: str
    canonical_url: Optional[str] = None
    character_count: int = 0
    word_count: int = 0


@dataclass
class ExtractionResult:
    """Result of article extraction containing status, data, or actionable error info."""
    success: bool
    status: ExtractionStatus
    article: Optional[ExtractedArticle] = None
    error_message: Optional[str] = None
    suggested_action: Optional[str] = None


def clean_text(text: str) -> str:
    """Normalizes whitespace and unicode characters while preserving paragraphs and sentence structure."""
    if not text:
        return ""
    # Normalize unicode non-breaking spaces and zero-width spaces
    cleaned = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    # Strip horizontal spaces adjacent to newlines
    cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
    # Normalize multiple spaces/tabs within lines into a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Clean space before commas, periods, colons, semicolons if preceded by a letter
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    # Collapse 3 or more newlines into double newlines (paragraph separator)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_unwanted_content(element: Tag, text: str) -> bool:
    """Checks if a paragraph or caption represents related links, credits, or promo text.

    Args:
        element: The BeautifulSoup Tag (p, figcaption, etc.)
        text: The cleaned text extracted from the element.

    Returns:
        True if the element should be excluded from the final article body.
    """
    if not text or len(text.strip()) == 0:
        return True

    # 1. Matches prefix patterns for related articles, photo credits, or subscription prompts
    if UNWANTED_CONTENT_PATTERNS.search(text):
        return True

    # 2. Check if element is a figcaption containing only photo attribution / credits
    if element.name == "figcaption":
        # Retain only substantive descriptions; exclude short photo citations
        if len(text.split()) < 6 or re.search(r"photo|credit|getty|reuters|pti|साभार|फोटो", text, re.I):
            return True

    # 3. Check if element's class explicitly indicates related or promotional content
    attrs = getattr(element, "attrs", None)
    if isinstance(attrs, dict):
        classes = attrs.get("class", [])
        class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
        elem_id = str(attrs.get("id", ""))
        combined = f"{class_str} {elem_id}".lower()
        if any(keyword in combined for keyword in ("related", "recommend", "also-read", "read-also", "read-more", "caption-credit")):
            return True

    return False


def extract_title(soup: BeautifulSoup) -> str:
    """Extracts article title with fallback hierarchy: OpenGraph -> Twitter -> H1 -> Title tag."""
    # 1. OpenGraph title
    og_title = soup.find("meta", property="og:title")
    if og_title and isinstance(og_title, Tag):
        attrs = getattr(og_title, "attrs", None)
        if isinstance(attrs, dict) and attrs.get("content"):
            content = str(attrs["content"]).strip()
            if content:
                return content

    # 2. Twitter card title
    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
    if tw_title and isinstance(tw_title, Tag):
        attrs = getattr(tw_title, "attrs", None)
        if isinstance(attrs, dict) and attrs.get("content"):
            content = str(attrs["content"]).strip()
            if content:
                return content

    # 3. Main <h1> tag
    h1 = soup.find("h1")
    if h1 and isinstance(h1, Tag) and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # 4. Fallback to <title>
    title_tag = soup.find("title")
    if title_tag and isinstance(title_tag, Tag) and title_tag.get_text(strip=True):
        title_text = title_tag.get_text(strip=True)
        # Clean common site name suffixes (e.g. "Title - BBC News" -> "Title")
        for separator in [" - ", " | ", " — ", " :: "]:
            if separator in title_text:
                parts = title_text.split(separator)
                if len(parts[0].strip()) > 10:
                    return parts[0].strip()
        return title_text

    return "Untitled Article"


def extract_canonical_url(soup: BeautifulSoup) -> Optional[str]:
    """Extracts canonical URL if declared in link tag or OpenGraph metadata."""
    canonical = soup.find("link", rel="canonical")
    if canonical and isinstance(canonical, Tag):
        attrs = getattr(canonical, "attrs", None)
        if isinstance(attrs, dict) and attrs.get("href"):
            href = str(attrs["href"]).strip()
            if href.startswith("http"):
                return href

    og_url = soup.find("meta", property="og:url")
    if og_url and isinstance(og_url, Tag):
        attrs = getattr(og_url, "attrs", None)
        if isinstance(attrs, dict) and attrs.get("content"):
            content = str(attrs["content"]).strip()
            if content.startswith("http"):
                return content

    return None


def extract_article_from_html(html: str, source_url: str) -> ExtractionResult:
    """Parses raw HTML and extracts readable article text and metadata.

    This function does not perform any network calls and can be tested deterministically.
    Implements defensive attribute access to avoid AttributeError during DOM mutation.
    Ensures inline spacing between tags and filters out related story recommendations.

    Args:
        html: The HTML string of the fetched page.
        source_url: The URL where this HTML was fetched from.

    Returns:
        ExtractionResult containing extracted article or granular error status.
    """
    if not html or not html.strip():
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.EMPTY_PAGE,
            error_message="The webpage returned an empty response.",
            suggested_action="Please check if the article URL is valid, or copy and paste the article text directly."
        )

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.INTERNAL_ERROR,
            error_message=f"HTML parsing failed: {type(e).__name__}",
            suggested_action="Please copy and paste the article text directly."
        )

    # Extract title and canonical URL before stripping tags
    title = extract_title(soup)
    canonical_url = extract_canonical_url(soup)
    domain = urlparse(source_url).netloc.lower()

    # 1. Remove non-content boilerplate tags safely
    for tag_name in BOILERPLATE_TAGS:
        for element in soup.find_all(tag_name):
            if not getattr(element, "decomposed", False):
                element.decompose()

    # 2. Remove elements matching common boilerplate classes/ids defensively.
    # Note: Decomposing a parent tag in bs4 decomposes its child tags and sets
    # their attrs to None. We must defensively check decomposed status and attrs dict.
    for element in soup.find_all(True):
        if getattr(element, "decomposed", False):
            continue
        attrs = getattr(element, "attrs", None)
        if not attrs or not isinstance(attrs, dict):
            continue

        classes_val = attrs.get("class", [])
        classes = " ".join(classes_val) if isinstance(classes_val, list) else str(classes_val)
        elem_id = str(attrs.get("id", ""))
        combined_attr = f"{classes} {elem_id}"

        if BOILERPLATE_CLASS_ID_PATTERNS.search(combined_attr):
            element.decompose()

    # 3. Find primary content container if present
    content_container = None
    article_tag = soup.find("article") or soup.find(attrs={"itemprop": "articleBody"})
    if article_tag and not getattr(article_tag, "decomposed", False):
        content_container = article_tag
    else:
        # Check standard article content class identifiers
        content_container = (
            soup.find(class_=re.compile(r"article[-_]?(?:body|content|text)", re.I))
            or soup.find(class_=re.compile(r"story[-_]?(?:body|content|text)", re.I))
            or soup.find(class_=re.compile(r"post[-_]?(?:body|content|text)", re.I))
            or soup.find("main")
        )

    # If no specific container found, use soup body
    search_root = content_container if content_container else (soup.body if soup.body else soup)

    # Extract all paragraph texts from the container
    paragraphs = []
    seen_paragraphs = set()
    # Search for <p> tags and meaningful <figcaption> elements
    for p in search_root.find_all(["p", "figcaption"]):
        if getattr(p, "decomposed", False):
            continue

        # Use separator=" " to avoid fused words between adjacent inline elements
        p_text = clean_text(p.get_text(separator=" "))

        # Filter out related article recommendations, photo credits, or promo calls
        if is_unwanted_content(p, p_text):
            continue

        # Filter out trivial fragments (e.g. single words, trivial snippets)
        if len(p_text) >= 20 and len(p_text.split()) >= 4:
            # Deduplicate repeated paragraphs (e.g. mobile/desktop duplicate wrappers, pull quotes)
            norm_p = " ".join(p_text.lower().split())
            if norm_p not in seen_paragraphs:
                seen_paragraphs.add(norm_p)
                paragraphs.append(p_text)

    # Fallback: if no <p> tags found, look for text in divs
    if not paragraphs and content_container:
        raw_content = clean_text(content_container.get_text(separator="\n\n"))
        if len(raw_content) >= MIN_ARTICLE_CHARS:
            paragraphs = [raw_content]

    extracted_text = "\n\n".join(paragraphs).strip()
    words = extracted_text.split()

    if not extracted_text:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.NO_ARTICLE_CONTENT,
            error_message=(
                "The page was reached, but no readable article content could be identified. "
                "The content may be rendered dynamically via JavaScript, behind a paywall/login, "
                "or contain non-article media."
            ),
            suggested_action="Please copy the article text from your browser and paste it directly into the text box above."
        )

    if len(extracted_text) < MIN_ARTICLE_CHARS or len(words) < MIN_ARTICLE_WORDS:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.CONTENT_TOO_SHORT,
            error_message=(
                f"The extracted text is too short ({len(words)} words, {len(extracted_text)} characters) "
                "to represent a complete news article."
            ),
            suggested_action="Please copy the full article text from your browser and paste it directly into the text box above."
        )

    article = ExtractedArticle(
        title=title,
        text=extracted_text,
        domain=domain,
        final_url=source_url,
        canonical_url=canonical_url,
        character_count=len(extracted_text),
        word_count=len(words)
    )

    return ExtractionResult(
        success=True,
        status=ExtractionStatus.SUCCESS,
        article=article
    )


def extract_article_from_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> ExtractionResult:
    """Fetches a webpage safely and extracts its readable article content.

    Distinguishes strictly between actual remote HTTP response codes (5xx, 4xx)
    and internal application-level exceptions (INTERNAL_ERROR).

    Args:
        url: The HTTP or HTTPS URL to fetch.
        timeout: Maximum seconds to wait for network response.

    Returns:
        ExtractionResult with article data or structured failure status.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.INVALID_URL,
            error_message=f"Unsupported URL protocol '{parsed.scheme}'. Only HTTP and HTTPS URLs are supported.",
            suggested_action="Please ensure the URL begins with http:// or https://"
        )

    try:
        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True
        )

        # 1. HTTP 401 or 403: Access Denied / Anti-bot protection
        if response.status_code in (401, 403):
            return ExtractionResult(
                success=False,
                status=ExtractionStatus.ACCESS_DENIED,
                error_message=(
                    f"This website did not allow automated access to the article (HTTP {response.status_code}). "
                    "Many major news outlets (such as Reuters, AP News, or subscription publications) "
                    "use automated bot protection or require interactive browser sessions."
                ),
                suggested_action="Please copy the article text directly from your browser and paste it into the 'Paste News / Article Text' box above."
            )

        # 2. HTTP 429: Rate Limited
        if response.status_code == 429:
            return ExtractionResult(
                success=False,
                status=ExtractionStatus.RATE_LIMITED,
                error_message="The website is currently rate limiting automated requests (HTTP 429 Too Many Requests).",
                suggested_action="Please try again later or copy and paste the article text directly."
            )

        # 3. HTTP 404: Not Found
        if response.status_code == 404:
            return ExtractionResult(
                success=False,
                status=ExtractionStatus.NOT_FOUND,
                error_message="The requested article page was not found on the website (HTTP 404 Not Found).",
                suggested_action="Please verify that the article URL is correct and has not been moved or deleted."
            )

        # 4. Other HTTP 4xx Client Errors
        if 400 <= response.status_code < 500:
            return ExtractionResult(
                success=False,
                status=ExtractionStatus.CLIENT_ERROR,
                error_message=f"The website rejected the request with client error (HTTP {response.status_code}).",
                suggested_action="Please check the URL or paste the article text directly."
            )

        # 5. HTTP 5xx Server Errors (Real remote server failures ONLY)
        if response.status_code >= 500:
            return ExtractionResult(
                success=False,
                status=ExtractionStatus.SERVER_ERROR,
                error_message=f"The source website server appears temporarily unavailable or returned an error (HTTP {response.status_code}).",
                suggested_action="Please try again later or copy and paste the article text directly."
            )

        # 6. Validate Content-Type is HTML
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return ExtractionResult(
                success=False,
                status=ExtractionStatus.NON_HTML,
                error_message=f"The URL points to a non-HTML file (Content-Type: '{content_type}'). Only readable HTML news articles are supported.",
                suggested_action="Please provide a link to an HTML article rather than a binary file or document."
            )

        final_url = response.url if response.url else url
        return extract_article_from_html(response.text, source_url=final_url)

    except requests.exceptions.Timeout:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.TIMEOUT,
            error_message=f"Connection timed out after {timeout} seconds while attempting to reach the website.",
            suggested_action="The server may be overloaded or unresponsive. Please try again or paste the article text directly."
        )
    except requests.exceptions.SSLError:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.NETWORK_ERROR,
            error_message="A secure SSL/TLS connection could not be established with the website.",
            suggested_action="The website's SSL certificate may be invalid. Please verify the URL or paste the text directly."
        )
    except requests.exceptions.ConnectionError:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.NETWORK_ERROR,
            error_message="Failed to establish a network connection to the server (DNS failure or connection refused).",
            suggested_action="Please check your internet connection and confirm the domain is currently reachable, or paste the article text directly."
        )
    except requests.exceptions.RequestException as e:
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.NETWORK_ERROR,
            error_message=f"Network request failed: {type(e).__name__}",
            suggested_action="Please try again or paste the article text directly."
        )
    except Exception as e:
        # Separate internal application exceptions from remote HTTP server errors
        return ExtractionResult(
            success=False,
            status=ExtractionStatus.INTERNAL_ERROR,
            error_message=f"An internal error occurred during article extraction: {type(e).__name__}",
            suggested_action="Please paste the article text directly into the text area."
        )
