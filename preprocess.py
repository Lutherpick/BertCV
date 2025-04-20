import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import spacy

class CVPreprocessor:
    def __init__(self, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.nlp = spacy.load('en_core_web_sm')
        
    def preprocess_text(self, text):
        """Clean and preprocess the text"""
        # Basic text cleaning
        text = text.lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, texts, labels=None):
        """Tokenize texts and create input features for BERT"""
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if labels is not None:
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            }
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    def extract_skills(self, text):
        """Extract potential skills using spaCy"""
        doc = self.nlp(text)
        skills = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                skills.append(token.text)
        return list(set(skills))
    
    def prepare_dataset(self, cv_texts, skill_labels):
        """Prepare the dataset for training"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in cv_texts]
        
        # Tokenize
        tokenized_data = self.tokenize(processed_texts, skill_labels)
        
        return tokenized_data

if __name__ == "__main__":
    # Example usage
    preprocessor = CVPreprocessor()
    
    # Example CV text
    cv_text = """
    Experienced Python developer with 5 years of experience in machine learning.
    Proficient in TensorFlow, PyTorch, and scikit-learn.
    Strong background in natural language processing and computer vision.
    """
    
    # Example skill labels (1 for skill, 0 for non-skill)
    skill_labels = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    
    # Process the data
    processed_data = preprocessor.prepare_dataset([cv_text], [skill_labels])
    
    print("Processed data structure:", processed_data.keys()) 