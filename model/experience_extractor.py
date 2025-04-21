import spacy
from typing import List, Dict, Tuple
import re
from datetime import datetime
import dateparser

class ExperienceExtractor:
    def __init__(self):
        """Initialize the experience extractor with NLP model."""
        self.nlp = spacy.load("en_core_web_sm")
        
        # Common date formats and patterns
        self.date_patterns = [
            r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
            r'Dec(?:ember)?)[,\s]+\d{4}',
            r'\d{4}[-/]\d{2}[-/]\d{2}',
            r'\d{2}[-/]\d{2}[-/]\d{4}',
            r'\d{4}'
        ]
        
    def _extract_dates(self, text: str) -> List[datetime]:
        """Extract dates from text using multiple approaches."""
        dates = []
        
        # Try pattern matching
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                parsed_date = dateparser.parse(match.group())
                if parsed_date:
                    dates.append(parsed_date)
        
        # Use spaCy's entity recognition for dates
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                parsed_date = dateparser.parse(ent.text)
                if parsed_date:
                    dates.append(parsed_date)
        
        return sorted(list(set(dates)))
    
    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organization names from text."""
        doc = self.nlp(text)
        organizations = []
        
        # Use spaCy's entity recognition
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                organizations.append(ent.text)
        
        return list(set(organizations))
    
    def _extract_experience_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract blocks of text that likely represent work experience."""
        experience_blocks = []
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            # Check if paragraph likely contains work experience
            doc = self.nlp(para)
            org_count = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
            date_count = len([ent for ent in doc.ents if ent.label_ in ['DATE', 'TIME']])
            
            if org_count > 0 and date_count > 0:
                experience_blocks.append({
                    'text': para,
                    'organizations': self._extract_organizations(para),
                    'dates': self._extract_dates(para)
                })
        
        return experience_blocks
    
    def extract_experience(self, text: str) -> Dict[str, any]:
        """Extract and analyze work experience from CV text."""
        experience_blocks = self._extract_experience_blocks(text)
        
        # Calculate total experience and number of companies
        all_organizations = set()
        date_ranges = []
        
        for block in experience_blocks:
            all_organizations.update(block['organizations'])
            if len(block['dates']) >= 2:
                date_ranges.append((min(block['dates']), max(block['dates'])))
        
        # Calculate total experience in years
        total_experience = 0
        if date_ranges:
            earliest = min(start for start, _ in date_ranges)
            latest = max(end for _, end in date_ranges)
            total_experience = (latest - earliest).days / 365.25
        
        return {
            'total_years': round(total_experience, 1),
            'num_companies': len(all_organizations),
            'companies': list(all_organizations),
            'experience_blocks': experience_blocks
        }
    
    def analyze_experience_match(self, experience_data: Dict[str, any], 
                               required_years: float = 0,
                               min_companies: int = 0) -> Dict[str, any]:
        """Analyze how well the experience matches requirements."""
        total_years = experience_data['total_years']
        num_companies = experience_data['num_companies']
        
        years_match = total_years >= required_years if required_years > 0 else True
        companies_match = num_companies >= min_companies if min_companies > 0 else True
        
        score = 0.0
        if required_years > 0:
            score += min(total_years / required_years, 1.0) * 0.7
        if min_companies > 0:
            score += min(num_companies / min_companies, 1.0) * 0.3
        
        return {
            'matches_requirements': years_match and companies_match,
            'score': round(score, 2),
            'feedback': {
                'years': f"Has {total_years} years of experience" + 
                        (f" (required: {required_years})" if required_years > 0 else ""),
                'companies': f"Worked at {num_companies} companies" +
                           (f" (required: {min_companies})" if min_companies > 0 else "")
            }
        } 