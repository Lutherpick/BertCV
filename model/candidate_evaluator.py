import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from .hybrid_extractor import HybridSkillExtractor
from .experience_extractor import ExperienceExtractor

class CandidateEvaluator:
    def __init__(self, device: str = None):
        """Initialize the candidate evaluator with skill and experience extractors."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.skill_extractor = HybridSkillExtractor(device)
        self.experience_extractor = ExperienceExtractor()
    
    def evaluate_candidate(self, 
                         cv_text: str, 
                         required_skills: List[str],
                         required_years: int,
                         job_description: str = None) -> Dict:
        """Evaluate candidate against job requirements."""
        # Extract skills and experience
        skills = self.skill_extractor.extract_skills(cv_text)
        experiences = self.experience_extractor.extract_experience(cv_text)
        
        # Calculate skill match
        skill_matches = []
        total_skill_score = 0
        
        for required_skill in required_skills:
            confidence = self.skill_extractor.get_skill_confidence(required_skill, cv_text)
            matches = confidence > 0.5
            skill_matches.append({
                'skill': required_skill,
                'found': matches,
                'confidence': round(confidence, 2)
            })
            total_skill_score += confidence
        
        skill_score = total_skill_score / len(required_skills) if required_skills else 1.0
        
        # Calculate experience match
        experience_match = self.experience_extractor.analyze_experience_match(
            experiences,
            required_years=required_years
        )
        
        # Generate feedback
        feedback = []
        
        # Skills feedback
        matched_skills = [m['skill'] for m in skill_matches if m['found']]
        missing_skills = [m['skill'] for m in skill_matches if not m['found']]
        
        if matched_skills:
            feedback.append(f"✓ Has {len(matched_skills)} of {len(required_skills)} required skills: {', '.join(matched_skills)}")
        if missing_skills:
            feedback.append(f"⚠ Missing required skills: {', '.join(missing_skills)}")
        
        # Experience feedback
        if experiences:
            feedback.append(f"✓ Has {experiences['num_companies']} relevant positions:")
            for company in experiences['companies']:
                feedback.append(f"  • {company}")
            
            if experiences['total_years'] >= required_years:
                feedback.append(f"✓ Has {experiences['total_years']:.1f} years of experience (requirement: {required_years} years)")
            else:
                feedback.append(f"⚠ Has {experiences['total_years']:.1f} years of experience (requirement: {required_years} years)")
        else:
            feedback.append("⚠ No clear work experience found")
        
        # Calculate overall score (weighted: 60% skills, 40% experience)
        overall_score = (skill_score * 0.6 + experience_match['score'] * 0.4) * 100
        
        # Add score interpretation
        if overall_score >= 80:
            feedback.append("🌟 Strong match for the position")
        elif overall_score >= 60:
            feedback.append("👍 Good potential, but some gaps exist")
        else:
            feedback.append("⚠ May need additional experience or skills")
        
        return {
            'skills_match': {
                'matches': skill_matches,
                'score': round(skill_score * 100, 1)
            },
            'experience_match': {
                'details': experiences,
                'score': round(experience_match['score'] * 100, 1),
                'feedback': experience_match['feedback']
            },
            'overall_score': round(overall_score, 1),
            'detailed_feedback': feedback
        } 