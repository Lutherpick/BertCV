import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.nn.functional as F
import spacy
import numpy as np
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

@dataclass
class Skill:
    name: str
    confidence: float
    source: str  # 'bert', 'rules', or 'both'
    context: str = None
    
    def __hash__(self):
        return hash((self.name, self.source))
    
    def __eq__(self, other):
        if not isinstance(other, Skill):
            return False
        return self.name == other.name and self.source == other.source

@dataclass
class Experience:
    company: str
    role: str
    dates: str
    duration_months: int
    description: str
    confidence: float
    relevance_score: float = None

class SkillExtractionBERT(nn.Module):
    def __init__(self, num_labels=2, dropout=0.1):
        super(SkillExtractionBERT, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence output instead of just [CLS]
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get logits for each token
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=3, learning_rate=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs['loss'].item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)

def predict_skills(model, text, preprocessor, device):
    """Predict skills from a given CV text"""
    # Preprocess and tokenize the text
    processed_data = preprocessor.prepare_dataset([text], None)
    
    # Move to device
    input_ids = processed_data['input_ids'].to(device)
    attention_mask = processed_data['attention_mask'].to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get predictions for each token
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1)
    
    return predictions 

class HybridSkillExtractor:
    def __init__(self, device: str = None):
        # Initialize device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load zero-shot classification pipeline
        self.zero_shot = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=0 if self.device == 'cuda' else -1)
        
        # Initialize skill embeddings cache
        self.skill_embeddings = {}
        self._precompute_skill_embeddings()
    
    def _precompute_skill_embeddings(self):
        """Precompute embeddings for known skills"""
        with torch.no_grad():
            for skill in COMMON_SKILLS:
                inputs = self.tokenizer(skill, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                self.skill_embeddings[skill] = outputs.last_hidden_state.mean(dim=1)
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for text"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
    
    def _extract_rule_based(self, text: str) -> Set[Skill]:
        """Extract skills using rule-based approach"""
        skills = set()
        doc = self.nlp(text)
        
        # Extract known skills
        for skill in COMMON_SKILLS:
            if skill.lower() in text.lower():
                skills.add(Skill(
                    name=skill,
                    confidence=1.0,
                    source='rules'
                ))
        
        # Extract technical terms and patterns
        code_pattern = re.compile(r'([A-Z][A-Za-z0-9]*[+.#]*[+]*|[A-Z]+)')
        for match in code_pattern.finditer(doc.text):
            potential_skill = match.group(0)
            if potential_skill in COMMON_SKILLS or potential_skill.lower() in COMMON_SKILLS:
                skills.add(Skill(
                    name=potential_skill,
                    confidence=0.9,
                    source='rules'
                ))
        
        return skills
    
    def _extract_bert_based(self, text: str) -> Set[Skill]:
        """Extract skills using BERT-based approach"""
        skills = set()
        text_embedding = self._get_text_embedding(text)
        
        # Split text into sentences for context
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Use zero-shot classification for skill identification
        for sentence in sentences:
            # Skip short or irrelevant sentences
            if len(sentence.split()) < 3:
                continue
                
            # Classify if sentence contains skills
            result = self.zero_shot(
                sentence,
                candidate_labels=["technical skill", "programming language", "technology", "tool", "not a skill"],
                multi_label=True
            )
            
            # If sentence likely contains skills, extract them
            if any(score > 0.7 for label, score in zip(result['labels'], result['scores']) 
                  if label != "not a skill"):
                # Get embedding for this sentence
                sent_embedding = self._get_text_embedding(sentence)
                
                # Compare with known skill embeddings
                for skill, skill_emb in self.skill_embeddings.items():
                    similarity = F.cosine_similarity(sent_embedding, skill_emb)
                    if similarity > 0.7:  # Threshold can be adjusted
                        skills.add(Skill(
                            name=skill,
                            confidence=float(similarity),
                            source='bert',
                            context=sentence
                        ))
        
        return skills
    
    def extract_skills(self, text: str) -> List[Skill]:
        """Extract skills using both approaches"""
        # Get skills from both methods
        rule_skills = self._extract_rule_based(text)
        bert_skills = self._extract_bert_based(text)
        
        # Combine results
        combined_skills = {}
        
        # Add rule-based skills
        for skill in rule_skills:
            combined_skills[skill.name.lower()] = skill
        
        # Add or update with BERT-based skills
        for skill in bert_skills:
            skill_key = skill.name.lower()
            if skill_key in combined_skills:
                # If found by both methods, update confidence and source
                existing_skill = combined_skills[skill_key]
                combined_skills[skill_key] = Skill(
                    name=skill.name,
                    confidence=max(existing_skill.confidence, skill.confidence),
                    source='both',
                    context=skill.context
                )
            else:
                combined_skills[skill_key] = skill
        
        return sorted(combined_skills.values(), key=lambda x: (-x.confidence, x.name))

class ExperienceExtractor:
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load zero-shot classification for role/company detection
        self.zero_shot = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=0 if self.device == 'cuda' else -1)
    
    def _extract_dates(self, text: str) -> List[Tuple[datetime, datetime]]:
        """Extract date ranges from text"""
        date_pattern = re.compile(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*[-–]\s*'
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s*\d{4}|'
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*[-–]\s*Present',
            re.IGNORECASE
        )
        
        dates = []
        for match in date_pattern.finditer(text):
            date_range = match.group(0)
            parts = date_range.split('–')
            if len(parts) != 2:
                parts = date_range.split('-')
            
            if len(parts) == 2:
                start_date = datetime.strptime(parts[0].strip(), '%B %Y')
                if 'present' in parts[1].lower():
                    end_date = datetime.now()
                else:
                    end_date = datetime.strptime(parts[1].strip(), '%B %Y')
                dates.append((start_date, end_date))
        
        return dates
    
    def _calculate_duration(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate duration in months"""
        return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    
    def _is_company_name(self, text: str) -> float:
        """Determine if text is likely a company name"""
        result = self.zero_shot(
            text,
            candidate_labels=["company name", "not a company"],
            multi_label=False
        )
        return result['scores'][0] if result['labels'][0] == "company name" else 0.0
    
    def _is_job_role(self, text: str) -> float:
        """Determine if text is likely a job role"""
        result = self.zero_shot(
            text,
            candidate_labels=["job title", "not a job title"],
            multi_label=False
        )
        return result['scores'][0] if result['labels'][0] == "job title" else 0.0
    
    def extract_experience(self, text: str) -> List[Experience]:
        experiences = []
        doc = self.nlp(text)
        
        # Find experience section
        text_lower = text.lower()
        experience_headers = ['experience', 'work history', 'employment']
        experience_start = min((text_lower.find(header) for header in experience_headers 
                              if text_lower.find(header) != -1), default=-1)
        
        if experience_start == -1:
            return experiences
        
        # Split into potential experience entries
        entries = re.split(r'\n\s*\n', text[experience_start:])
        
        for entry in entries:
            # Skip short entries
            if len(entry.split()) < 5:
                continue
            
            # Extract dates
            dates = self._extract_dates(entry)
            if not dates:
                continue
            
            # Get sentences and analyze each
            sentences = [sent.text.strip() for sent in self.nlp(entry).sents]
            if not sentences:
                continue
            
            # Find company and role
            company = None
            role = None
            company_confidence = 0.0
            role_confidence = 0.0
            
            for sent in sentences:
                # Look for company names
                doc_sent = self.nlp(sent)
                for ent in doc_sent.ents:
                    if ent.label_ == 'ORG':
                        conf = self._is_company_name(ent.text)
                        if conf > company_confidence:
                            company = ent.text
                            company_confidence = conf
                
                # Look for job roles
                for chunk in doc_sent.noun_chunks:
                    conf = self._is_job_role(chunk.text)
                    if conf > role_confidence:
                        role = chunk.text
                        role_confidence = conf
            
            if company and role:
                start_date, end_date = dates[0]  # Use first date range found
                experiences.append(Experience(
                    company=company,
                    role=role,
                    dates=f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}",
                    duration_months=self._calculate_duration(start_date, end_date),
                    description=' '.join(sentences),
                    confidence=min(company_confidence, role_confidence)
                ))
        
        return sorted(experiences, key=lambda x: (-x.duration_months, -x.confidence))

class CandidateEvaluator:
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.skill_extractor = HybridSkillExtractor(device)
        self.experience_extractor = ExperienceExtractor(device)
        
        # Load sentence transformer for semantic matching
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
    
    def _calculate_skill_match(self, 
                             candidate_skills: List[Skill], 
                             required_skills: List[str]) -> Tuple[float, List[Dict]]:
        """Calculate skill match score and details"""
        if not required_skills:
            return 1.0, []
        
        matches = []
        total_score = 0.0
        
        for req_skill in required_skills:
            best_match = None
            best_score = 0.0
            
            # Get embedding for required skill
            req_emb = self._get_text_embedding(req_skill)
            
            for cand_skill in candidate_skills:
                # Get embedding for candidate skill
                cand_emb = self._get_text_embedding(cand_skill.name)
                
                # Calculate similarity
                similarity = float(F.cosine_similarity(req_emb, cand_emb))
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = cand_skill
            
            matches.append({
                'required_skill': req_skill,
                'matched_skill': best_match.name if best_match else None,
                'confidence': best_score,
                'found': best_score > 0.7
            })
            
            total_score += best_score
        
        return total_score / len(required_skills), matches
    
    def _calculate_experience_match(self,
                                  experiences: List[Experience],
                                  required_years: int,
                                  job_description: str = None) -> Tuple[float, List[Dict]]:
        """Calculate experience match score and details"""
        if not experiences:
            return 0.0, []
        
        # Calculate total relevant experience
        total_months = sum(exp.duration_months for exp in experiences)
        total_years = total_months / 12
        
        # Calculate experience score
        experience_score = min(1.0, total_years / required_years) if required_years > 0 else 1.0
        
        # Calculate relevance scores if job description provided
        if job_description:
            job_emb = self._get_text_embedding(job_description)
            
            for exp in experiences:
                exp_emb = self._get_text_embedding(exp.description)
                exp.relevance_score = float(F.cosine_similarity(job_emb, exp_emb))
        
        return experience_score, [{
            'company': exp.company,
            'role': exp.role,
            'duration_months': exp.duration_months,
            'relevance_score': exp.relevance_score if exp.relevance_score is not None else 1.0
        } for exp in experiences]
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get text embedding using sentence transformer"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
    
    def evaluate_candidate(self, 
                         cv_text: str, 
                         required_skills: List[str],
                         required_years: int,
                         job_description: str = None) -> Dict:
        """Evaluate candidate against job requirements"""
        # Extract skills and experience
        skills = self.skill_extractor.extract_skills(cv_text)
        experiences = self.experience_extractor.extract_experience(cv_text)
        
        # Calculate skill match
        skill_score, skill_matches = self._calculate_skill_match(skills, required_skills)
        
        # Calculate experience match
        experience_score, experience_details = self._calculate_experience_match(
            experiences, required_years, job_description
        )
        
        # Generate feedback
        feedback = []
        
        # Skills feedback
        matched_skills = [m['required_skill'] for m in skill_matches if m['found']]
        missing_skills = [m['required_skill'] for m in skill_matches if not m['found']]
        
        if matched_skills:
            feedback.append(f"✓ Has {len(matched_skills)} of {len(required_skills)} required skills: {', '.join(matched_skills)}")
        if missing_skills:
            feedback.append(f"⚠ Missing required skills: {', '.join(missing_skills)}")
        
        # Experience feedback
        total_months = sum(exp['duration_months'] for exp in experience_details)
        total_years = total_months / 12
        
        if experiences:
            feedback.append(f"✓ Has {len(experiences)} relevant positions:")
            for exp in experiences:
                feedback.append(f"  • {exp.role} at {exp.company} ({exp.dates})")
            
            if total_years >= required_years:
                feedback.append(f"✓ Has {total_years:.1f} years of experience (requirement: {required_years} years)")
            else:
                feedback.append(f"⚠ Has {total_years:.1f} years of experience (requirement: {required_years} years)")
        else:
            feedback.append("⚠ No clear work experience found")
        
        # Calculate overall score (weighted: 60% skills, 40% experience)
        overall_score = (skill_score * 0.6 + experience_score * 0.4) * 100
        
        # Add score interpretation
        if overall_score >= 80:
            feedback.append("🌟 Strong match for the position")
        elif overall_score >= 60:
            feedback.append("👍 Good potential, but some gaps exist")
        else:
            feedback.append("⚠ May need additional experience or skills")
        
        return {
            'skills_match': skill_matches,
            'experience_match': {
                'experiences': experience_details,
                'total_years': total_years
            },
            'overall_score': overall_score,
            'detailed_feedback': feedback
        } 