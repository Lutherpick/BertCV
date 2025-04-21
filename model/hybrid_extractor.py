import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
import spacy
from typing import List, Set, Dict
import re
from dataclasses import dataclass
from .common_skills import COMMON_SKILLS

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
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() and self.device == 'cuda' else -1
        )
        
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

    def get_skill_confidence(self, skill: str, text: str) -> float:
        """Get confidence score for a specific skill in the text."""
        # Get embedding for the skill
        skill_embedding = self._get_text_embedding(skill)
        
        # Get embedding for relevant parts of the text
        doc = self.nlp(text)
        max_confidence = 0.0
        
        # Check each sentence for the skill
        for sent in doc.sents:
            # Skip very short sentences
            if len(sent.text.split()) < 3:
                continue
            
            # Rule-based check (direct match)
            if skill.lower() in sent.text.lower():
                max_confidence = max(max_confidence, 0.9)
                continue
            
            # BERT-based semantic similarity
            sent_embedding = self._get_text_embedding(sent.text)
            similarity = float(F.cosine_similarity(sent_embedding, skill_embedding))
            max_confidence = max(max_confidence, similarity)
        
        return max_confidence 