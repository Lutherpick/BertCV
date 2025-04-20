from flask import Flask, render_template, request, jsonify
import torch
from preprocess import CVPreprocessor
from model import SkillExtractionBERT, predict_skills
import os
from tika import parser
import io
import spacy
import re
from datetime import datetime
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model and preprocessor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocessor = CVPreprocessor()
model = SkillExtractionBERT()
model.to(device)

# Load spaCy for better text processing
nlp = spacy.load('en_core_web_sm')

# Common words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'about', 'into', 'over', 'after', 'process', 'using', 'have', 'has', 'had',
    'this', 'that', 'these', 'those', 'auto', 'extract', 'analyzing'
}

# Common technical skills and tools
COMMON_SKILLS = {
    'python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 'vue', 'node', 'nodejs',
    'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git',
    'machine learning', 'deep learning', 'ai', 'artificial intelligence', 'data science',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'opencv', 'nlp', 'kotlin',
    'natural language processing', 'computer vision', 'devops', 'ci/cd', 'agile', 'scrum'
}

def is_likely_skill(token):
    """Check if a token is likely to be a skill"""
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_token = re.sub(r'[^a-zA-Z0-9\s]', '', token.lower())
    
    # Skip empty strings and common words
    if not cleaned_token or cleaned_token in STOP_WORDS:
        return False
    
    # Check if it's in our known skills list
    if cleaned_token in COMMON_SKILLS:
        return True
    
    # Check if it's a proper noun or a technical term
    doc = nlp(token)
    if len(doc) == 1:
        token = doc[0]
        if token.pos_ in ['PROPN', 'NOUN'] and len(token.text) > 2:
            return True
    
    return False

def extract_skills_from_text(text):
    """Extract skills from text using rules and known skills"""
    # Ensure text is not empty
    if not text or len(text.strip()) == 0:
        return []
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract potential skills using rules and known skills
    skills = set()
    
    # Look for skills section
    text_lower = text.lower()
    skills_section = None
    
    # Common section headers
    section_headers = {
        'skills': ['technical skills', 'skills', 'technologies', 'technical expertise'],
        'experience': ['experience', 'work experience', 'employment history'],
        'education': ['education', 'academic background'],
        'projects': ['projects', 'project experience']
    }
    
    # Find skills section
    for header in section_headers['skills']:
        if header in text_lower:
            start = text_lower.find(header)
            # Find the next section after skills
            next_section_start = len(text)
            for section_type in section_headers.values():
                for section_header in section_type:
                    pos = text_lower.find(section_header, start + len(header))
                    if pos != -1 and pos < next_section_start:
                        next_section_start = pos
            skills_section = text[start:next_section_start]
            break
    
    # If skills section found, prioritize skills from there
    if skills_section:
        doc_skills = nlp(skills_section)
    else:
        doc_skills = doc
    
    # Look for known skills (including multi-word skills)
    for skill in COMMON_SKILLS:
        if skill.lower() in text_lower:
            skills.add(skill)
    
    # Look for technical terms and proper nouns in bullet points or lists
    for line in doc_skills.text.split('\n'):
        line = line.strip('•⚫⚪●○⦿⦾-–—•∙◦≫→⇒ \t')
        if line:
            line_doc = nlp(line)
            for token in line_doc:
                if token.pos_ in ['PROPN', 'NOUN'] and len(token.text) > 1:
                    # Check if it looks like a technical term
                    if (token.text.lower() in COMMON_SKILLS or 
                        any(char.isupper() for char in token.text[1:]) or  # Camel case
                        token.text.isupper() or  # Acronym
                        '.' in token.text or  # Likely a technology (e.g., Node.js)
                        token.text.lower().endswith(('js', 'db', 'ml', 'ai', 'api'))):
                        skills.add(token.text)
    
    # Look for programming languages and technologies
    code_pattern = re.compile(r'([A-Z][A-Za-z0-9]*[+.#]*[+]*|[A-Z]+)')
    for match in code_pattern.finditer(doc_skills.text):
        potential_skill = match.group(0)
        if potential_skill in COMMON_SKILLS or potential_skill.lower() in COMMON_SKILLS:
            skills.add(potential_skill)
    
    # Format skills consistently
    formatted_skills = []
    for skill in skills:
        # Keep acronyms uppercase
        if skill.isupper():
            formatted_skills.append(skill)
        # Keep known skills in their known format
        elif skill in COMMON_SKILLS:
            formatted_skills.append(skill)
        else:
            # Capitalize first letter of each word for other skills
            formatted_skills.append(skill.title())
    
    return sorted(formatted_skills)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using Tika"""
    try:
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.pdf')
        pdf_file.save(temp_path)
        
        # Parse PDF using Tika
        parsed = parser.from_file(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        if parsed["content"]:
            return parsed["content"].strip()
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def extract_experience(text):
    """Extract work experience details from text"""
    experience = []
    
    # Find the experience section
    text_lower = text.lower()
    experience_start = -1
    experience_end = len(text)
    
    # Common section headers
    experience_headers = ['experience', 'work experience', 'employment history', 'professional experience']
    next_section_headers = ['education', 'skills', 'projects', 'certifications', 'achievements']
    
    # Find start of experience section
    for header in experience_headers:
        pos = text_lower.find(header)
        if pos != -1 and (experience_start == -1 or pos < experience_start):
            experience_start = pos
    
    # Find end of experience section
    if experience_start != -1:
        for header in next_section_headers:
            pos = text_lower.find(header, experience_start)
            if pos != -1 and pos < experience_end:
                experience_end = pos
        
        # Extract experience section
        experience_text = text[experience_start:experience_end]
        
        # Split into entries (usually separated by blank lines or dates)
        entries = re.split(r'\n\s*\n', experience_text)
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = entry.strip().split('\n')
            current_exp = {
                'role': None,
                'company': None,
                'dates': None,
                'description': []
            }
            
            # Process each line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for dates
                date_match = re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*[-–]\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s*\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*[-–]\s*Present', line)
                
                if date_match:
                    current_exp['dates'] = date_match.group(0)
                    # Look for role and company in the same line
                    role_company = line.replace(date_match.group(0), '').strip()
                    if '|' in role_company:
                        parts = role_company.split('|')
                        current_exp['role'] = parts[0].strip()
                        current_exp['company'] = parts[1].strip()
                    elif 'at' in role_company.lower():
                        parts = role_company.lower().split('at')
                        current_exp['role'] = parts[0].strip()
                        current_exp['company'] = parts[1].strip()
                
                # If line starts with a bullet point or is indented, it's likely a description
                elif line.startswith(('•', '-', '∙', '◦', '  ', '\t')) or (current_exp['role'] and current_exp['company']):
                    current_exp['description'].append(line.strip('•⚫⚪●○⦿⦾-–—•∙◦≫→⇒ \t'))
                
                # If no role/company set yet, this might be the title line
                elif not current_exp['role']:
                    if '|' in line:
                        parts = line.split('|')
                        current_exp['role'] = parts[0].strip()
                        current_exp['company'] = parts[1].strip()
                    elif 'at' in line.lower():
                        parts = line.lower().split('at')
                        current_exp['role'] = parts[0].strip()
                        current_exp['company'] = parts[1].strip()
            
            # Clean up and add if we found both role and company
            if current_exp['role'] and current_exp['company']:
                # Clean up company name
                company = current_exp['company']
                company = re.sub(r'\s*\|.*$', '', company)  # Remove everything after |
                company = re.sub(r'\s*\(.*\)', '', company)  # Remove parentheses and contents
                
                experience.append({
                    'company': company.strip(),
                    'role': current_exp['role'].strip(),
                    'dates': current_exp['dates'],
                    'context': ' '.join(current_exp['description']) if current_exp['description'] else None
                })
    
    return experience

def analyze_candidate_fit(text, requirements):
    """Analyze how well a candidate fits specific job requirements"""
    analysis = {
        'skills_match': [],
        'experience_match': {},
        'overall_score': 0,
        'detailed_feedback': []
    }
    
    # Extract skills and experience
    candidate_skills = extract_skills_from_text(text)
    experience = extract_experience(text)
    
    # Analyze skills match
    required_skills = requirements.get('required_skills', [])
    for skill in required_skills:
        skill_lower = skill.lower()
        # Check for exact matches and related skills
        found = any(s.lower() == skill_lower for s in candidate_skills)
        # Check for partial matches (e.g., "Python" matches "Python Programming")
        if not found:
            found = any(skill_lower in s.lower() or s.lower() in skill_lower for s in candidate_skills)
        
        analysis['skills_match'].append({
            'skill': skill,
            'found': found
        })
    
    # Calculate skills match percentage
    if required_skills:
        skills_score = sum(1 for skill in analysis['skills_match'] if skill['found']) / len(required_skills)
    else:
        skills_score = 1.0  # If no skills required, give full score
    
    # Analyze experience
    required_years = requirements.get('years_of_experience', 0)
    total_experience_months = 0
    
    analysis['experience_match'] = {
        'companies_found': len(experience),
        'experience_details': experience
    }
    
    # Calculate total experience
    for exp in experience:
        if exp['dates']:
            # Parse dates
            dates = exp['dates'].split('-')
            if len(dates) == 2:
                start_date = parse_date(dates[0].strip())
                end_date = parse_date(dates[1].strip()) if 'present' not in dates[1].lower() else datetime.now()
                if start_date and end_date:
                    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    total_experience_months += months
    
    total_experience_years = total_experience_months / 12
    experience_score = min(1.0, total_experience_years / required_years) if required_years > 0 else 1.0
    
    # Generate detailed feedback
    feedback = []
    
    # Skills feedback
    matched_skills = [skill['skill'] for skill in analysis['skills_match'] if skill['found']]
    missing_skills = [skill['skill'] for skill in analysis['skills_match'] if not skill['found']]
    
    if matched_skills:
        feedback.append(f"✓ Has {len(matched_skills)} of {len(required_skills)} required skills: {', '.join(matched_skills)}")
    if missing_skills:
        feedback.append(f"⚠ Missing required skills: {', '.join(missing_skills)}")
    
    # Experience feedback
    if experience:
        feedback.append(f"✓ Has {len(experience)} relevant positions:")
        for exp in experience:
            feedback.append(f"  • {exp['role']} at {exp['company']} ({exp['dates']})")
        
        if total_experience_years >= required_years:
            feedback.append(f"✓ Has {total_experience_years:.1f} years of experience (requirement: {required_years} years)")
        else:
            feedback.append(f"⚠ Has {total_experience_years:.1f} years of experience (requirement: {required_years} years)")
    else:
        feedback.append("⚠ No clear work experience found")
    
    # Calculate overall score (weighted average: 60% skills, 40% experience)
    analysis['overall_score'] = (skills_score * 0.6 + experience_score * 0.4) * 100
    
    # Add score interpretation
    if analysis['overall_score'] >= 80:
        feedback.append("🌟 Strong match for the position")
    elif analysis['overall_score'] >= 60:
        feedback.append("👍 Good potential, but some gaps exist")
    else:
        feedback.append("⚠ May need additional experience or skills")
    
    analysis['detailed_feedback'] = feedback
    return analysis

def parse_date(date_str):
    """Parse date string into datetime object"""
    try:
        return datetime.strptime(date_str.strip(), '%B %Y')
    except ValueError:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            content = None
            if file.filename.lower().endswith('.pdf'):
                # Handle PDF file
                content = extract_text_from_pdf(file)
                if content is None:
                    return jsonify({'error': 'Failed to extract text from PDF'}), 400
            else:
                # Handle text file
                content = file.read().decode('utf-8')
            
            # Get job requirements from request
            requirements = {}
            if 'requirements' in request.form:
                requirements = json.loads(request.form['requirements'])
            
            # Extract skills and analyze candidate
            skills = extract_skills_from_text(content)
            analysis = analyze_candidate_fit(content, requirements)
            
            return jsonify({
                'skills': skills,
                'analysis': analysis,
                'text': content
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        skills = extract_skills_from_text(text)
        
        # Get job requirements from request or use defaults
        requirements = data.get('requirements', {
            'required_skills': ['Python', 'Machine Learning'],
            'years_of_experience': 2
        })
        
        # Perform candidate analysis
        analysis = analyze_candidate_fit(text, requirements)
        
        return jsonify({
            'skills': skills,
            'analysis': analysis,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True) 