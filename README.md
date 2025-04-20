# CV Skills Extractor

A web application that analyzes CVs/resumes to extract skills and assess candidate fit for job requirements. Built with Python Flask and spaCy NLP.

## Features

- Extract skills from CV text or PDF files
- Analyze candidate fit based on required skills and experience
- Calculate match scores and provide detailed feedback
- Support for both text input and PDF upload
- Modern, responsive UI with real-time analysis

## Requirements

- Python 3.8+
- Flask
- spaCy
- Apache Tika
- PyTorch
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cv-skills-extractor.git
cd cv-skills-extractor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Either:
   - Paste CV text directly into the text input
   - Upload a PDF file
   - Specify job requirements (skills and experience)
   - Click "Analyze CV" to get results

## Project Structure

```
cv-skills-extractor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Frontend template
├── static/               # Static files (if any)
├── uploads/             # Temporary folder for uploads
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How It Works

1. **Skills Extraction**:
   - Uses spaCy NLP for text processing
   - Matches against known technical skills
   - Identifies potential skills using NLP patterns
   - Handles multi-word skills and variations

2. **Experience Analysis**:
   - Extracts work experience details
   - Calculates total experience duration
   - Identifies roles and companies
   - Provides context for each position

3. **Candidate Assessment**:
   - Matches required skills against extracted skills
   - Evaluates experience against requirements
   - Calculates overall match score
   - Provides detailed feedback

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 