class PromptBuilder:
    def build_claims_prompt(self, text: str, max_claims: int = 30) -> str:
        """
        Build the prompt for claim extraction.

        Args:
            text: Input text to analyze
            max_claims: Maximum number of claims to extract

        Returns:
            Formatted prompt string
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
    
    # NOTE: text is a dict in the dataloader, but is marked as str here
    def build_claim_filter_prompt(self, claims: list, text: str) -> str:
        """
        Build the prompt for claim filtering. Here use the same definition of a claim as in build_claims_prompt,
        in order to be consistent.

        Args:
            claims: List of claims to filter, these claims comes from an "item" in the dataloader
            text: Input text to analyze

        Returns:
            Formatted prompt string, which include the claims and their credibility and evidence
        """
        return f"""You are an information extraction system.

TASK:
You need to check whether the claims of the following companies originate from the evidence documents
submitted by the companies. If you identify any potential evidences, please list the evidences in JSON format.
If no credible evidence is found, please mark that claim as "Evidence not found" in the returned JSON format.
You should sort the claims by their relative credibility from high to low and return the claims in the same order.

DEFINITION OF A CLAIM:
- An atomic statement that can be checked as true/false.
- No vague stylistic comments unless they express a factual assertion.
- Merge duplicates. Remove contradictions or mark them as "uncertain".
- Ignore instructions, formatting hints, and unrelated boilerplate.

INPUT FORMAT:

{{
  "claims": [
    {{
      "id": "C1",
      "claim": "Plain English description of the claim.",
    }},
    ...
  ],
  "original_text": "The original text from pdf content or ocr result of pdfs."
}}

OUTPUT FORMAT:
Return ONLY valid JSON, no extra text. Do not wrap the JSON in markdown code blocks.
The claims are sorted by their RELATIVE credibility from high to low.

Schema:
{{
  "claims": [
    {{
      "id": "C5",
      "claim": "Plain English description of the claim, same as the input.",
      "evidences": [
        {{
          "id": "C5-E1"
          "filename": "The name of the evidence file.",
          "page": "The page number in the evidence file, if applicable. If not applicable, set it to -1.",
          "evidence": "Short snippet from the original text (max 200 chars)",
        }},
        ...
      ]
    }},
    {{
      "id": "C3",
      "claim": "Plain English description of the claim, same as the input.",
      "evidences": [
        {{
          "id": "C3-E1"
          "filename": "The name of the evidence file.",
          "page": "The page number in the evidence file, if applicable. If not applicable, set it to -1.",
          "evidence": "Short snippet from the original text (max 200 chars)",
        }},
        ...
      ]
    }},
    {{
      "id": "C7",
      "claim": "Plain English description of the claim, same as the input.",
      "evidences": [
        {{
          "id": "C7-E1"
          "filename": "The name of the evidence file.",
          "page": "The page number in the evidence file, if applicable. If not applicable, set it to -1.",
          "evidence": "Short snippet from the original text (max 200 chars)",
        }},
        ...
      ]
    }},
    ...
  ]
}}

Rules:
- For claims, use stable ids: C1, C2, C3, ...; For evidences of claims C1, C2, C3, ..., use ids like C1-E1, C1-E2, C2-E1, C2-E2, C3-E1, ...
- "evidence" must be copied verbatim from the original text (up to 200 characters).
- Focus on factual claims about market size, technology performance, company information, product features, etc.
- Extract numerical data, dates, comparisons, and specific achievements.
- Escape quotes properly so JSON is valid.

INPUT:
{{
  "claims": {claims},
  "original_text": "{text}"
}}
""".strip()