"""Input handling and validation module for the AI News Verification System.

This module validates and normalizes user input (raw text or article URL)
before downstream processing (e.g. extraction, claim verification).
No external network calls, scraping, or model predictions are performed here.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from urllib.parse import urlparse
import re

# Minimum thresholds for text to be considered a viable news claim or statement
MIN_TEXT_LENGTH = 20
MIN_WORD_COUNT = 3
MAX_TEXT_LENGTH = 50000  # Guard against extremely long inputs causing performance issues

# Regex to check basic hostname validity (e.g. example.com, sub.domain.org)
HOSTNAME_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$|^localhost$"
)


@dataclass
class NormalizedInput:
    """Represents a clean, normalized user input ready for downstream processing."""
    input_type: Literal["text", "url"]
    raw_content: str
    cleaned_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of input validation, containing status, error message, and normalized data."""
    is_valid: bool
    error_message: Optional[str] = None
    normalized_input: Optional[NormalizedInput] = None


def validate_text(raw_text: str) -> ValidationResult:
    """Validates and normalizes user-provided news text.

    Args:
        raw_text: The uncleaned text entered by the user.

    Returns:
        ValidationResult indicating validity, error details, or normalized text.
    """
    cleaned_text = raw_text.strip()

    if not cleaned_text:
        return ValidationResult(
            is_valid=False,
            error_message="The news text input is empty. Please enter a valid news statement or article text."
        )

    if len(cleaned_text) < MIN_TEXT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"The provided text is too short ({len(cleaned_text)} characters). "
                f"A minimum of {MIN_TEXT_LENGTH} characters is required for a meaningful news statement."
            )
        )

    if len(cleaned_text) > MAX_TEXT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"The provided text is too long ({len(cleaned_text):,} characters). "
                f"Please limit the input to {MAX_TEXT_LENGTH:,} characters for reliable processing."
            )
        )

    words = cleaned_text.split()
    if len(words) < MIN_WORD_COUNT:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"The text must contain at least {MIN_WORD_COUNT} words to represent a coherent statement "
                f"(found {len(words)} word{'s' if len(words) != 1 else ''})."
            )
        )

    normalized = NormalizedInput(
        input_type="text",
        raw_content=raw_text,
        cleaned_content=cleaned_text,
        metadata={
            "character_count": len(cleaned_text),
            "word_count": len(words)
        }
    )
    return ValidationResult(is_valid=True, normalized_input=normalized)


def validate_url(raw_url: str) -> ValidationResult:
    """Validates and normalizes a user-provided article URL.

    Checks HTTP/HTTPS scheme and valid host structure without making network requests.

    Args:
        raw_url: The uncleaned URL entered by the user.

    Returns:
        ValidationResult indicating validity, error details, or normalized URL.
    """
    cleaned_url = raw_url.strip()

    if not cleaned_url:
        return ValidationResult(
            is_valid=False,
            error_message="The URL field is empty. Please provide a valid web article URL."
        )

    # Reject URLs containing spaces
    if " " in cleaned_url:
        return ValidationResult(
            is_valid=False,
            error_message="The URL contains invalid spaces. Please enter a properly formatted URL."
        )

    try:
        parsed = urlparse(cleaned_url)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Failed to parse URL: {e}"
        )

    if parsed.scheme.lower() not in ("http", "https"):
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Invalid URL protocol '{parsed.scheme or 'none'}'. "
                "Only standard HTTP or HTTPS URLs (e.g., https://example.com/article) are supported."
            )
        )

    netloc = parsed.netloc.strip().lower()
    # Strip optional port if present for hostname validation
    hostname = netloc.split(":")[0] if ":" in netloc else netloc

    if not hostname or not HOSTNAME_REGEX.match(hostname):
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Invalid domain name in URL '{cleaned_url}'. "
                "Please make sure it includes a valid domain (e.g., https://www.bbc.com/news/...)."
            )
        )

    normalized = NormalizedInput(
        input_type="url",
        raw_content=raw_url,
        cleaned_content=cleaned_url,
        metadata={
            "scheme": parsed.scheme.lower(),
            "domain": hostname,
            "path": parsed.path
        }
    )
    return ValidationResult(is_valid=True, normalized_input=normalized)


def validate_and_normalize_input(news_text: Optional[str], article_url: Optional[str]) -> ValidationResult:
    """Top-level validator enforcing mutually exclusive input selection and rules.

    Args:
        news_text: Optional text input from the user.
        article_url: Optional URL input from the user.

    Returns:
        ValidationResult with status, actionable message, and normalized representation.
    """
    has_text = bool(news_text and news_text.strip())
    has_url = bool(article_url and article_url.strip())

    if not has_text and not has_url:
        return ValidationResult(
            is_valid=False,
            error_message="Please provide either news text or an article URL to proceed."
        )

    if has_text and has_url:
        return ValidationResult(
            is_valid=False,
            error_message=(
                "Both news text and an article URL were provided. "
                "Please provide either news text OR an article URL, not both. Clear one field to proceed."
            )
        )

    if has_text:
        return validate_text(news_text)  # type: ignore[arg-type]

    return validate_url(article_url)  # type: ignore[arg-type]
