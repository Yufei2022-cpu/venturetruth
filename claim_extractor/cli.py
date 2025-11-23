"""
Command-line interface for claim extraction.
"""

import os
import sys
import argparse
from typing import Optional

from core import extract_claims
from utils import load_text_from_source, save_output


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract factual claims from text using the OpenAI API."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to input text file. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to output JSON file. If omitted, generate automatically.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o",
        help="Model name (default: gpt-4o). You can use e.g. gpt-3.5-turbo.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=30,
        help="Maximum number of claims to extract (default: 30).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (default: uses OPENAI_API_KEY environment variable)",
    )
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Load text
    text_input = load_text_from_source(args.input)

    # Extract claims
    result = extract_claims(
        text=text_input,
        model=args.model,
        max_claims=args.max_claims,
        api_key=args.api_key
    )

    # Save output
    output_file = save_output(
        data=result.to_dict(),
        output_path=args.output,
        input_path=args.input
    )

    # Print results
    print(result.to_json(indent=2, ensure_ascii=False))
    print(f"\nResults have been saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()