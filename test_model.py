import torch
from preprocess import CVPreprocessor
from model import SkillExtractionBERT, predict_skills

def test_model(cv_text):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize preprocessor and model
    preprocessor = CVPreprocessor()
    model = SkillExtractionBERT()
    model.to(device)
    
    # Load pre-trained weights (if available)
    try:
        model.load_state_dict(torch.load('skill_extraction_model.pth'))
        print("Loaded pre-trained model weights")
    except:
        print("No pre-trained weights found. Using randomly initialized model.")
    
    # Get predictions
    predictions = predict_skills(model, cv_text, preprocessor, device)
    
    # Get the original tokens to show what was predicted
    tokenized = preprocessor.tokenizer(cv_text, return_tensors='pt', truncation=True)
    tokens = preprocessor.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
    
    # Print results
    print("\nCV Text:")
    print(cv_text)
    print("\nPredicted Skills:")
    for token, pred in zip(tokens, predictions[0]):
        if pred == 1:  # If predicted as a skill
            print(f"- {token}")

if __name__ == "__main__":
    # Example CV texts to test
    test_cvs = [
        """
        Senior Data Scientist with 5 years of experience in machine learning and AI.
        Proficient in Python, TensorFlow, and PyTorch.
        Strong background in natural language processing and computer vision.
        Experience with AWS, Docker, and Kubernetes.
        """,
        
        """
        Full Stack Developer specializing in web technologies.
        Expert in JavaScript, React, Node.js, and MongoDB.
        Experience with cloud platforms and DevOps tools.
        Strong knowledge of SQL and NoSQL databases.
        """,
        
        """
        Machine Learning Engineer with expertise in deep learning.
        Skilled in Python, scikit-learn, and data analysis.
        Experience with big data technologies and cloud computing.
        """
    ]
    
    # Test each CV
    for i, cv in enumerate(test_cvs, 1):
        print(f"\n{'='*50}")
        print(f"Testing CV {i}")
        print(f"{'='*50}")
        test_model(cv) 