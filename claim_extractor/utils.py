"""
Utility functions for claim extraction.
"""

import os
import sys
import re
import json
from typing import Optional


def load_text_from_source(path: Optional[str]) -> str:
    """
    Load text from file or stdin.

    Args:
        path: Path to input file, or None for stdin

    Returns:
        Loaded text content

    Raises:
        SystemExit: If text cannot be loaded
    """
    if path:
        if not os.path.exists(path):
            sys.stderr.write(f"Error: file not found: {path}\n")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        # Read from stdin
        if sys.stdin.isatty():
            sys.stderr.write("No input given and no stdin data. Aborting.\n")
            sys.exit(1)
        text = sys.stdin.read().strip()

    if not text:
        sys.stderr.write("Error: input text is empty.\n")
        sys.exit(1)
    return text


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON from markdown code blocks if present.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        Extracted JSON string or original text if no code blocks found
    """
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # Use the first match (should be the JSON content)
        return matches[0].strip()

    # If no code blocks found, return the original text
    return text.strip()


def save_output(data: dict, output_path: Optional[str] = None,
                input_path: Optional[str] = None) -> str:
    """
    Save output to JSON file.

    Args:
        data: Data to save
        output_path: Explicit output path, or None for auto-generation
        input_path: Input file path for auto-generating output filename

    Returns:
        Path to the saved file
    """
    if output_path:
        final_output_path = output_path
    elif input_path:
        base_name = os.path.splitext(input_path)[0]
        final_output_path = f"{base_name}_claims.json"
    else:
        final_output_path = "extracted_claims.json"

    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return final_output_path