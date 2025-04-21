from flask import Flask, render_template, request, jsonify
from tika import parser
import os
import json
from model.hybrid_extractor import HybridSkillExtractor
from model.experience_extractor import ExperienceExtractor
from model.candidate_evaluator import CandidateEvaluator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize evaluator
evaluator = CandidateEvaluator()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using Tika"""
    temp_path = None
    try:
        # Ensure uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.pdf')
        print(f"Saving PDF to temporary path: {temp_path}")
        pdf_file.save(temp_path)
        
        # Initialize Tika server (if not already running)
        print("Initializing Tika server...")
        from tika import initVM
        initVM()
        
        # Parse PDF using Tika
        print("Parsing PDF with Tika...")
        parsed = parser.from_file(temp_path)
        print("Tika parsing completed")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temporary file removed")
        
        if parsed and "content" in parsed and parsed["content"]:
            content = parsed["content"].strip()
            print(f"Successfully extracted {len(content)} characters")
            return content
        print("No content extracted from PDF")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Cleaned up temporary file after error")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_cv():
    """Analyze CV text and extract skills and experience."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        cv_text = data['text']
        requirements = data.get('requirements', {})
        
        # Get requirements
        required_skills = [s.strip() for s in requirements.get('required_skills', '').split(',') if s.strip()]
        required_years = int(requirements.get('years_experience', 0))
        
        # Analyze candidate
        analysis = evaluator.evaluate_candidate(
            cv_text=cv_text,
            required_skills=required_skills,
            required_years=required_years
        )
        
        return jsonify(analysis)
    except Exception as e:
        print(f"Error in analyze_cv: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and analysis."""
    try:
        print("Received upload request")
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        print(f"Received file: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            print("Not a PDF file")
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Get requirements from form data
        requirements = {}
        print("Form data:", request.form)
        if 'requirements' in request.form:
            try:
                requirements = json.loads(request.form['requirements'])
                print("Parsed requirements:", requirements)
            except json.JSONDecodeError as e:
                print(f"Error parsing requirements JSON: {e}")
                return jsonify({'error': 'Invalid requirements format'}), 400
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        content = extract_text_from_pdf(file)
        if content is None:
            print("Failed to extract text from PDF")
            return jsonify({'error': 'Failed to extract text from PDF'}), 400
        print(f"Extracted text length: {len(content)} characters")
        
        # Get requirements
        required_skills = [s.strip() for s in requirements.get('required_skills', '').split(',') if s.strip()]
        required_years = int(requirements.get('years_experience', 0))
        print(f"Processing with requirements - Skills: {required_skills}, Years: {required_years}")
        
        # Analyze candidate
        analysis = evaluator.evaluate_candidate(
            cv_text=content,
            required_skills=required_skills,
            required_years=required_years
        )
        print("Analysis completed successfully")
        
        return jsonify({
            'analysis': analysis,
            'text': content
        })
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Initialize Tika server
    from tika import initVM
    initVM()
    
    app.run(debug=True) 