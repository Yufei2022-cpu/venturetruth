#!/usr/bin/env python3
"""
extract_claims.py

Summarize an input text into a set of factual claims using the OpenAI API.
"""

import os
import sys
import json
import argparse
import re
from typing import Any, Dict

from openai import OpenAI  # pip install openai


def parse_args() -> argparse.Namespace:
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
        default="gpt-5",
        help="Model name (default: gpt-4o). You can use e.g. gpt-3.5-turbo.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=30,
        help="Maximum number of claims to extract (default: 30).",
    )
    return parser.parse_args()


def load_text_from_source(path: str | None) -> str:
    if path:
        if not os.path.exists(path):
            sys.stderr.write(f"Error: file not found: {path}\n")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        # Read from stdin
        if sys.stdin.isatty():
            sys.stderr.write("No --input given and no stdin data. Aborting.\n")
            sys.exit(1)
        text = sys.stdin.read().strip()

    if not text:
        sys.stderr.write("Error: input text is empty.\n")
        sys.exit(1)
    return text


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON from markdown code blocks if present.
    Handles cases where the response is wrapped in ```json ... ``` or ``` ... ```
    """
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # Use the first match (should be the JSON content)
        return matches[0].strip()

    # If no code blocks found, return the original text
    return text.strip()


def build_prompt(text: str, max_claims: int) -> str:
    """
    We ask the model for STRICT JSON with a clear schema.
    """
    return f"""
You are an information extraction system.

TASK:
Read the text below and extract concise, verifiable claims.

DEFINITION OF A CLAIM:
- An atomic statement that can be checked as true/false.
- No vague stylistic comments unless they express a factual assertion.
- Merge duplicates. Remove contradictions or mark them as "uncertain".
- Ignore instructions, formatting hints, and unrelated boilerplate.

EXAMPLES OF GOOD CLAIMS:
- "Eco-Friendly Glitter Market size is estimated to reach $450 Million by 2030, growing at a CAGR of 11.4 percent during 2024-2030."
- "€10B Market of Carbon Tracking Solutions and AI Supply Chain Optimization."
- "Iscent's technology generates 10 times less carbon footprint than traditional industries."
- "The company was founded in 2015 and has 200 employees."
- "Product X reduces energy consumption by 30% compared to conventional solutions."

OUTPUT FORMAT:
Return ONLY valid JSON, no extra text. Do not wrap the JSON in markdown code blocks.

Schema:
{{
  "claims": [
    {{
      "id": "C1",
      "claim": "Plain English description of the claim.",
      "confidence": 0.0 to 1.0,
      "evidence": "Short snippet from the original text (max 200 chars)"
    }},
    ...
  ]
}}

Rules:
- Max {max_claims} claims.
- Use stable ids: C1, C2, C3, ...
- "confidence":
    - 1.0 for very clear and explicit statements.
    - Around 0.7–0.9 for strong but slightly indirect statements.
    - Below 0.7 if the language is speculative / conditional.
- "evidence" must be copied verbatim from the original text (up to 200 characters).
- Focus on factual claims about market size, technology performance, company information, product features, etc.
- Extract numerical data, dates, comparisons, and specific achievements.
- Escape quotes properly so JSON is valid.

TEXT:
{text}
""".strip()



def extract_claims(
        text: str,
        model: str = "gpt-4o",
) -> Dict[str, Any]:
    # Use the provided API key directly
    api_key = "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"

    if not api_key:
        sys.stderr.write(
            "Error: API key is not available.\n"
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    prompt = build_prompt(text, max_claims=args.max_claims)

    try:
        # Using the chat completions API which is more standard
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,  # Low temperature for more deterministic output
        )
    except Exception as e:
        sys.stderr.write(f"OpenAI API error: {e}\n")
        sys.exit(1)

    # Extract the response content
    raw = response.choices[0].message.content
    if not raw:
        sys.stderr.write(
            "Warning: Empty response from API.\n"
        )
        sys.exit(1)

    # Clean the response - extract JSON from markdown if needed
    cleaned_raw = extract_json_from_markdown(raw)

    # Ensure it's valid JSON
    try:
        data = json.loads(cleaned_raw)
    except json.JSONDecodeError as e:
        sys.stderr.write("Error: model output was not valid JSON.\n")
        sys.stderr.write(f"Raw output:\n{raw}\n")
        sys.stderr.write(f"Cleaned output:\n{cleaned_raw}\n")
        sys.stderr.write(f"Details: {e}\n")
        sys.exit(1)

    # Light sanity check
    if "claims" not in data or not isinstance(data["claims"], list):
        sys.stderr.write(
            "Error: JSON does not contain expected 'claims' list.\n"
        )
        sys.stderr.write(f"Got:\n{json.dumps(data, indent=2, ensure_ascii=False)}\n")
        sys.exit(1)

    return data


if __name__ == "__main__":
    args = parse_args()
    text_input = load_text_from_source(args.input)
    result = extract_claims(text_input, model=args.model)

    if args.output:
        output_file = args.output
    elif args.input:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_claims.json"
    else:
        output_file = "extracted_claims.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nResults have been saved to: {output_file}", file=sys.stderr)