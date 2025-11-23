# Claim Extractor User Guide

## Overview

Claim Extractor is a Python module designed to extract factual claims from text using the OpenAI API. It analyzes input text, identifies factual statements, and structures them into a standardized JSON format.

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation Steps
1. Clone or download the `claim_extractor` folder to your project directory
2. Install dependencies:
```bash
pip install openai pydantic
```

## Quick Start

### Using as a Command Line Tool

#### Basic Usage
```bash
# Extract claims from a file
python -m claim_extractor.cli --input article.txt

# Extract claims from stdin
cat article.txt | python -m claim_extractor.cli

# Specify output file
python -m claim_extractor.cli --input article.txt --output claims.json
```

#### Full Parameter List
```bash
python -m claim_extractor.cli \
  --input article.txt \
  --output claims.json \
  --model gpt-4o \
  --max-claims 20 \
  --api-key your_openai_api_key_here
```

**Parameter Description:**
- `--input, -i`: Path to input text file (optional, defaults to stdin)
- `--output, -o`: Path to output JSON file (optional, auto-generated)
- `--model, -m`: OpenAI model name (default: gpt-4o)
- `--max-claims`: Maximum number of claims to extract (default: 30)
- `--api-key`: OpenAI API key (optional, can use environment variable)

### Using as a Python Library

#### Basic Example
```python
from claim_extractor import extract_claims

# Extract claims from string
text = """
Tesla delivered 1.8 million vehicles in 2023, a 38% increase from 2022.
The company expects production to reach 2 million vehicles in 2024.
"""

result = extract_claims(text)

# Iterate through all claims
for claim in result.claims:
    print(f"{claim.id}: {claim.claim}")
    print(f"  Confidence: {claim.confidence}")
    print(f"  Evidence: {claim.evidence}")
    print()
```

#### Advanced Usage
```python
from claim_extractor import extract_claims

# Custom parameters
result = extract_claims(
    text="Your text content...",
    model="gpt-4o",
    max_claims=15,
    api_key="your_api_key_here"  # Optional, uses env var by default
)

# Convert to dictionary
data = result.to_dict()

# Convert to JSON string
json_str = result.to_json(indent=2, ensure_ascii=False)

# Access specific claim
if result.claims:
    first_claim = result.claims[0]
    print(f"First claim: {first_claim.claim}")
```

## Output Format

### JSON Structure
```json
{
  "claims": [
    {
      "id": "C1",
      "claim": "Tesla delivered 1.8 million vehicles in 2023",
      "confidence": 0.95,
      "evidence": "Tesla delivered 1.8 million vehicles in 2023"
    },
    {
      "id": "C2", 
      "claim": "The company expects production to reach 2 million vehicles in 2024",
      "confidence": 0.85,
      "evidence": "The company expects production to reach 2 million vehicles in 2024"
    }
  ]
}
```

### Field Description
- `id`: Unique claim identifier (C1, C2, C3...)
- `claim`: Concise description of the claim content
- `confidence`: Confidence score (0.0-1.0)
- `evidence`: Evidence snippet from original text

## Configuring API Key

### Method 1: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Method 2: Command Line Parameter
```bash
python -m claim_extractor.cli --input text.txt --api-key "your_key"
```

### Method 3: Specify in Python Code
```python
result = extract_claims(text, api_key="your_key")
```

## Usage Examples

### Example 1: Analyzing News Articles
```python
from claim_extractor import extract_claims

news_article = """
Apple released its earnings today, showing Q3 revenue reached $81.7 billion, a 36% year-over-year increase.
iPhone sales accounted for $39.5 billion, representing 48% of total revenue. The company expects next quarter revenue to be between $85-90 billion.
"""

claims = extract_claims(news_article)

print(f"Extracted {len(claims.claims)} claims:")
for claim in claims.claims:
    print(f"- [{claim.confidence:.2f}] {claim.claim}")
```

### Example 2: Processing Long Documents
```python
def process_large_document(text, chunk_size=2000):
    """Strategy for processing long documents"""
    claims = []
    
    # Process in chunks
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunk_claims = extract_claims(chunk, max_claims=10)
        claims.extend(chunk_claims.claims)
    
    return claims
```

### Example 3: Batch Processing Files
```python
import glob
import os
from claim_extractor import extract_claims

def batch_process(input_pattern, output_dir):
    """Batch process multiple files"""
    for filepath in glob.glob(input_pattern):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        claims = extract_claims(text)
        
        # Save results
        output_file = f"{output_dir}/{os.path.basename(filepath)}_claims.json"
        with open(output_file, 'w') as f:
            f.write(claims.to_json(indent=2))
```

## Error Handling

### Common Errors and Solutions

1. **API Key Errors**
   ```python
   try:
       result = extract_claims(text)
   except SystemExit as e:
       print("Please check API key configuration")
   ```

2. **Network Connection Issues**
   ```python
   import time
   
   def extract_with_retry(text, retries=3):
       for attempt in range(retries):
           try:
               return extract_claims(text)
           except SystemExit:
               if attempt < retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff
                   continue
               else:
                   raise
   ```

3. **Input Text Too Long**
   ```python
   # Split long text
   def split_text(text, max_length=4000):
       return [text[i:i + max_length] for i in range(0, len(text), max_length)]
   ```

## Best Practices

1. **Preprocessing Text**
   - Clean irrelevant formatting markers
   - Remove HTML tags
   - Normalize encoding format

2. **Post-processing Results**
   ```python
   def post_process_claims(claims):
       """Post-processing: filter low-confidence claims, deduplicate, etc."""
       # Filter low confidence
       high_confidence = [c for c in claims if c.confidence > 0.7]
       
       # Simple deduplication (based on claim content)
       seen = set()
       unique_claims = []
       for claim in high_confidence:
           if claim.claim not in seen:
               seen.add(claim.claim)
               unique_claims.append(claim)
       
       return unique_claims
   ```

3. **Performance Optimization**
   - Batch process multiple documents
   - Set appropriate `max_claims` parameter
   - Use suitable models (gpt-3.5-turbo is faster)

## Module Structure

```
claim_extractor/
├── __init__.py      # Package initialization
├── core.py          # Core extraction logic
├── prompts.py       # Prompt templates
├── schemas.py       # Data models
├── utils.py         # Utility functions
└── cli.py           # Command line interface
```

## Support and Feedback

If you encounter issues or need new features, please check:
1. API key is properly configured
2. Network connection is stable
3. Input text format is appropriate
4. Model is available and accessible

## License

This module is open source under the MIT License.