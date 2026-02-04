# VentureTruth

**An AI-powered fact-checking system for venture capital due diligence**

VentureTruth is an automated pipeline that extracts, analyzes, and verifies factual claims from startup pitch documents and investment materials. Using advanced LLM technology and web search integration, it helps investors identify potential misinformation and assess the credibility of startup claims.

---

## 🎯 Overview

The system processes startup documents through a multi-stage pipeline:

1. **Document Ingestion** - Extracts text from PDFs and company metadata
2. **Claim Extraction** - Identifies factual claims using GPT models
3. **Claim Verification** - Validates claims against web sources using Perplexity AI
4. **Quality Assessment** - Evaluates verification quality with improvement suggestions
5. **Robustness Testing** - Tests claim stability across multiple prompt variations
6. **Evidence Analysis** - Analyzes source quality and reliability

The pipeline features an iterative refinement loop that automatically improves verification quality over multiple rounds based on AI-generated feedback.

---

## 🚀 Quick Start

### Prerequisites

- Python >= 3.10
- [Poppler](https://poppler.freedesktop.org/) (for PDF processing)
- OpenAI API key
- Perplexity API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd venturetruth
```

2. Install dependencies:
```bash
pip install -e .
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# PERPLEXITY_API_KEY=your_key_here
```

4. Run the pipeline:
```bash
python -m src.main_pipeline
```

---

## 📋 Configuration

All pipeline settings are configured in `config/config.yaml`:

### File Content Extraction
```yaml
file_content_extractor:
  csv_path: path/to/companies.csv
  pdf_folder: path/to/documents
  output_path: res/output.json
  limit: 10  # Maximum number of companies to process
```

### Claim Extraction
```yaml
claim_extractor:
  model: gpt-4o  # OpenAI model name
  max_claims: 20  # Maximum claims per document
  temperature: 0.0
```

### Claim Verification
```yaml
claim_verifier:
  model: gpt-4o
  temperature: 0.0
  max_rounds: 3  # Number of iterative refinement rounds
```

### Quality Assessment
```yaml
quality_checker:
  model: gpt-4o
  temperature: 0.0
  output_path: res/quality_report.json
```

---

## 🏗️ Architecture

### Pipeline Stages

#### 1. Document Ingestion (`src/file_content_extraction/`)

Processes company documents and extracts structured information:
- **`ingestion_pipeline.py`** - Orchestrates CSV loading and document processing
- **`pdf_processor.py`** - Handles individual PDF files
- **`standard_extractor.py`** - Standard text extraction
- **`ocr_extractor.py`** - OCR for image-based text
- **`data_loader.py`** - Custom data loader for processed results

**Output**: `res/output.json`
```json
[
  {
    "metadata": {
      "Account Name": "Company XYZ",
      "Website": "https://...",
      "Startup Industry Sector": "FinTech",
      ...
    },
    "files": [
      {
        "filename": "pitch_deck.pdf",
        "content": "extracted text...",
        "content_ocr": "ocr text..."
      }
    ]
  }
]
```

#### 2. Claim Extraction (`src/claim_extractor/`)

Uses GPT to identify factual claims from documents:
- **`claim_extractor.py`** - Main extraction logic
- **`prompt_builder.py`** - Optimized prompt construction

**Output**: `res/claims_*.json` (one per company)
```json
{
  "claims": [
    {
      "id": "C1",
      "claim": "Company achieved 300% YoY revenue growth",
      "confidence": 0.95,
      "evidence": "We grew our revenue by 300% year-over-year..."
    }
  ]
}
```

#### 3. Claim Verification (`src/claim_verification/`)

Validates claims using web search and AI reasoning:
- **`claim_verifier.py`** - Verification logic with quality-aware improvements
- **`search_manager.py`** - Perplexity AI integration for web search

**Output**: `res/verification_*.json`
```json
{
  "verification_results": [
    {
      "claim_id": "C1",
      "verdict": "SUPPORTED",
      "confidence": 0.85,
      "reasoning": "Multiple credible sources confirm...",
      "sources": [
        "https://techcrunch.com/...",
        "https://crunchbase.com/..."
      ]
    }
  ]
}
```

Verdict categories:
- `SUPPORTED` - Claim backed by credible sources
- `CONTRADICTED` - Claim refuted by evidence
- `INSUFFICIENT_EVIDENCE` - Not enough data to verify

#### 4. Quality Assessment (`src/quality_checker/`)

AI-powered quality evaluator that:
- Analyzes verification reasoning quality
- Assesses source credibility and relevance
- Identifies verification weaknesses
- Generates specific improvement suggestions
- Evaluates search query effectiveness

**Output**: `res/quality_report.json`
```json
{
  "overall_quality_score": 0.82,
  "overall_rating": "GOOD",
  "critical_issues_count": 2,
  "top_issues": [
    "Consider checking company's official financial disclosures",
    "Seek independent verification beyond press releases"
  ],
  "suggested_improvements": [...]
}
```

#### 5. Iterative Refinement Loop

The pipeline automatically runs multiple verification rounds:
1. Initial verification with web search
2. Quality assessment identifies issues
3. Re-verification with quality-aware improvements
4. Repeat until max rounds reached or quality threshold met

This ensures progressively higher verification quality.

#### 6. Evaluation Modules (`src/evaluation/`)

Comprehensive evaluation framework:
- **`robustness_checker.py`** - Tests claim stability across 3 prompt variations
- **`claims_evidence_analyzer.py`** - Analyzes source quality per claim
- **`metadata_evaluator.py`** - Compares extracted metadata with ground truth
- **`prompt_variations.py`** - Different verification prompt strategies

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `res/output.json` | Extracted company data and document content |
| `res/claims_*.json` | Extracted claims per company |
| `res/claims_by_company.json` | All claims organized by company |
| `res/verification_*.json` | Verification results per company |
| `res/final_report.json` | Comprehensive multi-company report |
| `res/quality_report.json` | Quality assessment with improvement suggestions |
| `res/robustness_report.json` | Robustness testing results |
| `res/evidence_analysis.json` | Source quality analysis |
| `res/metadata_eval.json` | Metadata evaluation report |

---

## 🔍 Example Workflow

```python
from src.main_pipeline import main_pipeline

# Run complete pipeline with all stages
main_pipeline()
```

The pipeline will:
1. ✅ Extract content from company documents
2. ✅ Extract factual claims using GPT-4
3. ✅ Verify claims through web search (3 rounds with quality feedback)
4. ✅ Test verification robustness
5. ✅ Analyze evidence quality
6. ✅ Generate comprehensive reports

Example output:
```
🚀 Starting Round 1 of 3...
--- Processing company TechStartup Inc., ID 1 ---
📝 Extracting claims...
   Found 15 claims
🔍 Verifying claims...
   ✅ Supported: 10
   ❌ Contradicted: 2
   ⚠️ Insufficient Evidence: 3
📊 Analyzing verification quality...
   Overall Score: 0.78
   Critical Issues: 3
🚀 Starting Round 2 of 3...
   [Incorporating quality improvements...]
```

---

## 🛠️ Advanced Usage

### Run Individual Stages

```python
from src.main_pipeline import (
    ingestion_pipeline,
    claim_extraction,
    claim_verification,
    quality_assessment
)

# Run only document extraction
ingestion_pipeline()

# Run only claim extraction
claim_extraction()

# Run only verification
all_search_results, all_company_reports = claim_verification()
```

### Customize Configuration

Edit `config/config.yaml` to:
- Change AI models (e.g., `gpt-4o-mini` for faster/cheaper runs)
- Adjust temperature for creativity vs. consistency
- Set maximum claims per document
- Configure number of refinement rounds
- Enable/disable evaluation stages

### Access Verification Results

```python
import json

# Load final report
with open('res/final_report.json', 'r') as f:
    report = json.load(f)

# Get overall statistics
print(f"Total Claims: {report['overall_summary']['total_claims']}")
print(f"Supported: {report['overall_summary']['supported']}")
print(f"High Risk: {report['overall_summary']['high_risk_count']}")

# Iterate through company reports
for company in report['companies']:
    print(f"\n{company['company_name']}:")
    for detail in company['claim_details']:
        if detail['verdict'] == 'CONTRADICTED':
            print(f"  ⚠️ {detail['claim']}")
```

---

## 📈 Key Features

- **Multi-stage Pipeline**: Automated end-to-end processing from documents to verified claims
- **AI-Powered Extraction**: GPT-4 identifies factual claims with confidence scores
- **Web Search Integration**: Perplexity AI searches the web for evidence
- **Iterative Refinement**: Quality checker provides feedback to improve verification
- **Robustness Testing**: Tests claim stability across multiple prompt variations
- **Evidence Analysis**: Evaluates source credibility and diversity
- **Comprehensive Reporting**: Detailed JSON reports with statistics and insights
- **Configurable**: Easy YAML configuration for all pipeline parameters

---

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

---

## 📝 Project Structure

```
venturetruth/
├── config/
│   └── config.yaml                 # Pipeline configuration
├── src/
│   ├── main_pipeline.py            # Main pipeline orchestration
│   ├── file_content_extraction/    # Stage 1: Document processing
│   ├── claim_extractor/            # Stage 2: Claim extraction
│   ├── claim_verification/         # Stage 3: Claim verification
│   ├── quality_checker/            # Stage 4: Quality assessment
│   ├── evaluation/                 # Evaluation modules
│   │   ├── robustness_checker.py
│   │   ├── claims_evidence_analyzer.py
│   │   └── metadata_evaluator.py
│   ├── common/                     # Shared utilities
│   └── utils/                      # Helper functions
├── res/                            # Output directory
├── tests/                          # Test suite
└── README.md
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Extraction results not found`
- **Solution**: Run `ingestion_pipeline()` first to extract documents

**Issue**: `API rate limit exceeded`
- **Solution**: Add delays in config or use different API keys for parallel processing

**Issue**: PDF extraction fails
- **Solution**: Ensure Poppler is installed and in PATH

**Issue**: Poor verification quality
- **Solution**: Increase `max_rounds` in config to allow more refinement iterations

### Verify Setup

```python
# Check API keys
import os
from dotenv import load_dotenv
load_dotenv()
print(f"OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
print(f"Perplexity: {'✓' if os.getenv('PERPLEXITY_API_KEY') else '✗'}")

# Check Poppler installation
import subprocess
try:
    subprocess.run(['pdfinfo', '-v'], capture_output=True)
    print("Poppler: ✓")
except FileNotFoundError:
    print("Poppler: ✗ (not installed)")
```

---

## 📚 Citation

If you use VentureTruth in your research, please cite:

```bibtex
@software{venturetruth2024,
  title={VentureTruth: AI-Powered Fact-Checking for Venture Capital},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/venturetruth}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 💡 Future Enhancements

- [ ] Support for more document formats (Word, Excel, web pages)
- [ ] Interactive web dashboard for results visualization
- [ ] Real-time claim verification API
- [ ] Multi-language support
- [ ] Custom fact-checking rule engine
- [ ] Integration with CRM systems (Salesforce, HubSpot)
- [ ] Blockchain-based audit trail for verifications

---

## 📧 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing documentation in `/docs`
- Review configuration in `config/config.yaml`

---

**Built with ❤️ for better due diligence in venture capital**