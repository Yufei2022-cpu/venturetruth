"""
Core functionality for claim extraction.
"""

import sys
import json
from typing import Dict, Any

from openai import OpenAI

from prompts import build_claims_prompt
from schemas import ClaimsResponse
from utils import extract_json_from_markdown


def extract_claims(
        text: str,
        model: str = "gpt-4o",
        max_claims: int = 30,
        api_key: str = None
) -> ClaimsResponse:
    """
    Extract factual claims from text using OpenAI API.

    Args:
        text: Input text to analyze
        model: OpenAI model to use
        max_claims: Maximum number of claims to extract
        api_key: OpenAI API key (uses environment variable if not provided)

    Returns:
        ClaimsResponse object with extracted claims

    Raises:
        SystemExit: If API call fails or response is invalid
    """
    # Use provided API key or environment variable
    if not api_key:
        api_key = "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"

    if not api_key:
        sys.stderr.write(
            "Error: OpenAI API key is required. "
            "Set OPENAI_API_KEY environment variable or pass api_key parameter.\n"
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    prompt = build_claims_prompt(text, max_claims=max_claims)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for more deterministic output
        )
    except Exception as e:
        sys.stderr.write(f"OpenAI API error: {e}\n")
        sys.exit(1)

    # Extract the response content
    raw = response.choices[0].message.content
    if not raw:
        sys.stderr.write("Warning: Empty response from API.\n")
        sys.exit(1)

    # Clean the response - extract JSON from markdown if needed
    cleaned_raw = extract_json_from_markdown(raw)

    # Parse and validate the response
    try:
        data = json.loads(cleaned_raw)
        claims_response = ClaimsResponse(**data)
    except (json.JSONDecodeError, ValueError) as e:
        sys.stderr.write("Error: model output was not valid JSON or doesn't match schema.\n")
        sys.stderr.write(f"Raw output:\n{raw}\n")
        sys.stderr.write(f"Cleaned output:\n{cleaned_raw}\n")
        sys.stderr.write(f"Details: {e}\n")
        sys.exit(1)

    return claims_response