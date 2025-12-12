# Complete Guide for the Claim Verification Pipeline

**Prerequisites**:
* Python >= 3.10
* Installed dependencies (`python -m pip install -e .`) 
* `poppler` installed on your machine
* `OPEN_API_KEY` and `PERPLEXITY_API_KEY` in the `.env` file

The entire logic can be separated in three parts:
1. [File Content Extraction](#file-content-extraction)
2. [Claim Extraction](#claim-extraction)
3. [Claim Verification](#claim-verification-guide)

## File Content Extraction
The key functionality is following:
* Extract the metadata and file content of each company that can be found in the list
* Store the metadata in the JSON file
* Custom data loader for the extracted information 

### Configuration
The configuration for the **File Content Extraction** can be found in the `config/config.json` in the `file_content_extractor` section.

Currently, following configarations are available:
* `csv_path`: path to the CSV file that stores infomation about the companies and the documents associated with them
* `pdf_folder`: path to the folder where the documents accosicated with companies are stored
* `output_path`: path where to store the extracted information
* `limit`: maximal amount of companies which information is extracted

### Code Structure:
The code can be found under `src/file_content_extraction` 
* `base_extractor.py`: base class for the different extraction strategies
* `standard_extractor.py`: class for standard text extractor
* `ocr_extractor.py`: class for text extraction from the images
* `data_schemes.py`: dataclasses that are used to store the extracted information
* `pdf_processor.py`: class that handles processing of the single PDF file
* `ingestion_pipeline.py`: class that orchestrates the loading of the CSV metadata and processing of documents

### Output Format
```json
[
    {
        "metadata": {
            "Attachment File Names": "[\"document_1\", \"document_2\"]",
            "UVC Investment Opportunity: ID": "id",
            "Account Name": "name",
            "Website": "website",
            "Startup Country": "country",
            "Startup Industry Sector": "sector",
            "UVC Investment Opportunity: Created Date": "creation date",
            "UVC Investment Status": "status",
            "Rejection Email Sent": "rejection date",
            "Internal Rejection Reason": "reason",
            "Advisor": "advisor",
            "Last Modified Date": "modification date",
            "Initial Impression": "impression",
            "UVC Investment Opportunity: Number": "number",
            "IO Comment": "comment",
            "UVC Startup Description": "description",
            "Startup Source Category": "category",
            "00 \u00e2\u20ac\u201c Outbound lead added": "",
            "Created Date vs. First Discussion": "1000000",
            "Network Deal": "1",
            "Female Founder": "0",
            "r - rejected/out": "rejection_date",
            "h - on hold": "",
            "Follow-Up": "",
            "Former Names": "",
            "Initial Opportunity Contact": "contact",
            "Description": "description",
            "0 - Documentation received": "date of receival",
            "? - First Discussion": ""
        },
        "files": [
            {
                "filename": "document_1.pdf",
                "extension": ".pdf",
                "type": "document",
                "content": "text content",
                "content_ocr": "ocr content"
            },
            {
                "filename": "document_2.pdf",
                "extension": ".pdf",
                "type": "document",
                "content": "text content",
                "content_ocr": "ocr content"
            }
        ]
    }
]
```

---

## Claim Extraction

Claim Extractor is a Python module designed to extract factual claims from text using the OpenAI API. It analyzes input text, identifies factual statements, and structures them into a standardized JSON format.

### Configuration

The configuration for the **Claim Extraction** can be found in the `config/config.json` in the `claim_extractor` section.

Currently, following configurations are available:
* `model`: name of the Open AI model that should be used to extract claims
* `max_claims`: manimum number of claims to extract 
* `temperature`: temperature values that the model should use
* `output_path`: path where to store the claims

*Note: by default the claims are not stored in the JSON file. The claim extractor has a function to store claims. It can be used for the debug purposes.*

### Code Structure
The code can be found under `src/claim_extractor` 
* `claim_extractor.py`: class that performs claim extraction
* `prompt_builder.py`: class that builds the prompt for the claim extraction

### Output Format
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

#### Field Description
* `id`: Unique claim identifier (C1, C2, C3...)
* `claim`: Concise description of the claim content
* `confidence`: Confidence score (0.0-1.0)
* `evidence`: Evidence snippet from original text

---

## Claim Verification
Claim Verification is a Python module designed to verify factual claims from text using the OpenAI and Perplexity. It performs internet search, combines the result with provided information, performs verification, and structures the result into a standardized JSON format.

### Configuration

The configuration for the **Claim Verification** can be found in the `config/config.json` in the `claim_verifier` section.

Currently, following configurations are available:
* `model`: name of the Open AI model that should be used to extract claims
* `temperature`: temperature values that the model should use
* `output_path`: path where to store the claims

### Code Structure
The code can be found under `src/claim_verification` 
* `claim_verifier.py`: class that performs claim verification
* `search_manager.py`: class that performs the internet search for the provided claims

### Output Format
```json
{
    "verification_results": [
        {
            "claim_id": 1,
            "verdict": "verdict",
            "confidence": 0.95,
            "reasoning": "reasoning",
            "sources": [
                "source_1",
                "source_2",
                "source_3"
            ]
        },
        {
            "claim_id": 2,
            "verdict": "verdict",
            "confidence": 0.9,
            "reasoning": "reasoning",
            "sources": [
                "source_1",
                "source_2"
            ]
        }
    ]
}
```

#### Field Description
* `claim_id`: id of the claim that the verification result refers to
* `verdict`: Specifies whether the claim is supported, refuted, or there is insufficient evidence
* `confidence`: numerical value from 0 to 1 that indicates how certain is the model about the verdict
* `reasoning`: explanation
* `sources`: list of the URL sources used to support the reasoning

---

## How to run
To start the program, execute:

```bash
python -m main
```

Depending on the use case you can adjust program entry point.

## Support and Feedback

If you encounter issues or need new features, please check:
1. API key is properly configured
2. Network connection is stable
3. Input text format is appropriate
4. Model is available and accessible

## License

This module is open source under the MIT License.