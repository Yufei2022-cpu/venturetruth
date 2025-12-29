import sys
import json

from openai import OpenAI

from src.claim_extractor.prompt_builder import PromptBuilder
from utils.utils import extract_json_from_markdown
from common.schemes import ClaimsResponse, FilteredClaims

class ClaimExtractor:
    
    def __init__(self, api_key, model="gpt-4o", temperature=0, max_claims=30):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_claims = max_claims
        self.prompt_builder = PromptBuilder()
        self.is_setup = False
    
    def setup(self):
        self.client = OpenAI(api_key=self.api_key)
        self.is_setup = True
    
    def extract_claims(self, text):
        if not self.is_setup:
            print(f"The ClaimExtractor is not setup!")
            return
        
        prompt = self.prompt_builder.build_claims_prompt(text, self.max_claims)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,  # Low temperature for more deterministic output
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
    
    def filter_claims(self, claims, text):
        """
        Basically the same as extract_claims, but with a different prompt, which is more focused on sort the claims by
        their relativ credibility from high to low, because the LLM can't estimate the exact number of credibilty(like 0.67, 0.78, etc.)
        reliabily.
        """
        if not self.is_setup:
            print(f"The Claim Extractor is not setup!")
            return
        
        prompt = self.prompt_builder.build_claim_filter_prompt(claims, text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,  # Low temperature for more deterministic output
            )
        except Exception as e:
            sys.stderr.write(f"OpenAI API error: {e}\n")
            sys.exit(1)
        
        raw = response.choices[0].message.content
        if not raw:
            sys.stderr.write("Warning: Empty response from API.\n")
            sys.exit(1)
        
        # Clean the response - extract JSON from markdown if needed
        cleaned_raw = extract_json_from_markdown(raw)

        try:
            data = json.loads(cleaned_raw)
            claims_filtered = FilteredClaims(**data)
        except (json.JSONDecodeError, ValueError) as e:
            sys.stderr.write("Error: model output was not valid JSON or doesn't match schema.\n")
            sys.stderr.write(f"Raw output:\n{raw}\n")
            sys.stderr.write(f"Cleaned output:\n{cleaned_raw}\n")
            sys.stderr.write(f"Details: {e}\n")
            sys.exit(1)

        return claims_filtered