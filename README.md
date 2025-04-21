# CV Extraction and Analysis Tool

A web application for extracting and analyzing skills and experience from CV/resume documents. The system uses NLP and machine learning techniques to identify relevant skills and experience from uploaded CVs and match them against job requirements.

## Features

- PDF document upload and text extraction
- Skills extraction using hybrid approach (rule-based and ML-based)
- Experience detection and duration calculation
- Candidate evaluation against job requirements
- Detailed feedback on skill matching and experience relevance

## Technical Stack

- Python 3.7+
- Flask web framework
- PyTorch for machine learning components
- Transformers library (BERT, BART)
- Apache Tika for PDF parsing
- spaCy for NLP processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cv-extraction.git
cd cv-extraction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the application:
```bash
python app.py
```

The application will be available at http://localhost:5000

## Usage

1. Access the web interface
2. Upload a CV/resume in PDF format
3. Enter job requirements (skills and years of experience)
4. Click "Analyze CV" to get the evaluation results

## License

MIT 