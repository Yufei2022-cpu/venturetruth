class PromptBuilder:
    def build_claims_prompt(self, text: str, max_claims: int = 30, company_name: str = "Company") -> str:
        """
        Build the prompt for claim extraction.

        Args:
            text: Input text to analyze
            max_claims: Maximum number of claims to extract
            company_name: Name of the company for prefixing claim IDs

        Returns:
            Formatted prompt string
        """
        # Sanitize company name for use in IDs (remove special chars, spaces to underscores)
        safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name).strip("_")
        
        return f"""
You are an information extraction system specialized in identifying marketing claims from startup pitch materials.

TASK:
Read the text below and extract ONLY marketing claims — promotional statements that assert facts about the company, its products, technology, market, or achievements that can be externally verified.

DEFINITION OF A MARKETING CLAIM:
- A factual assertion made to promote or validate the company/product.
- An atomic statement that can be checked as true/false using external sources.
- Typically includes: market size claims, technology performance metrics, customer/revenue numbers, product capabilities, competitive advantages, partnerships, awards, or published research.

WHAT TO EXCLUDE (NOT marketing claims):
- Goals, intentions, or future plans (e.g., "We aim to...", "We plan to raise...", "Seeking investment...")
- Funding requests or investment asks (e.g., "Looking for €500K seed round")
- Internal company metadata (e.g., founding date, number of employees, company location)
- Pricing information without a comparative claim
- Team bios and credentials (unless claiming specific external achievements)
- Subjective opinions or aspirational statements
- Generic filler (e.g., "We are a passionate team...")

EXAMPLES OF GOOD CLAIMS (INCLUDE):
- "Eco-Friendly Glitter Market size is estimated to reach $450 Million by 2030, growing at a CAGR of 11.4%."
- "Our technology generates 10x less carbon footprint than traditional industries."
- "Product X reduces energy consumption by 30% compared to conventional solutions."
- "We have 150+ enterprise customers across 12 countries."
- "Our platform processes over 1 million transactions per day."
- "Named a Gartner Cool Vendor in 2024."

EXAMPLES OF BAD CLAIMS (EXCLUDE):
- "We are seeking €500K investment." (funding request, not a verifiable marketing claim)
- "Founded in 2015 with 50 employees." (metadata, not promotional)
- "We plan to expand to 10 new markets." (future intention, not verifiable fact)
- "Our team is passionate about sustainability." (subjective, not verifiable)
- "Monthly subscription costs €99." (pricing without comparative claim)

OUTPUT FORMAT:
Return ONLY valid JSON, no extra text. Do not wrap the JSON in markdown code blocks.

Schema:
{{
  "claims": [
    {{
      "id": "{safe_company_name}-C1",
      "claim": "Plain English description of the marketing claim.",
      "category": "market_size",
      "evidence": "Short snippet from the original text (max 200 chars)"
    }},
    ...
  ]
}}

Rules:
- Max {max_claims} claims.
- Use stable ids with company prefix: {safe_company_name}-C1, {safe_company_name}-C2, {safe_company_name}-C3, ...
- "category" must be one of: market_size, technology, customer, partnership, award, financial, other.
- "evidence" must be copied verbatim from the original text (up to 200 characters).
- Focus ONLY on externally verifiable marketing claims: market size, technology performance, customer metrics, product capabilities, achievements, partnerships.
- Exclude all goals, requests, metadata, and internally-focused statements.
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