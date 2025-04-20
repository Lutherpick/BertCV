import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocess import CVPreprocessor
from model import SkillExtractionBERT, train_model, predict_skills

def create_dataloader(data, batch_size=8):
    """Create PyTorch DataLoader from processed data"""
    dataset = TensorDataset(
        data['input_ids'],
        data['attention_mask'],
        data['labels']
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize preprocessor
    preprocessor = CVPreprocessor()
    
    # Example CV texts and labels (in a real scenario, you would load this from your dataset)
    cv_texts = [
        """
        Experienced Python developer with 5 years of experience in machine learning.
        Proficient in TensorFlow, PyTorch, and scikit-learn.
        Strong background in natural language processing and computer vision.
        """,
        """
        Senior Software Engineer specializing in web development.
        Expert in JavaScript, React, Node.js, and MongoDB.
        Experience with AWS and Docker.
        """
    ]
    
    # Example labels (1 for skill, 0 for non-skill)
    skill_labels = [
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    ]
    
    # Prepare dataset
    processed_data = preprocessor.prepare_dataset(cv_texts, skill_labels)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(cv_texts))
    train_data = {
        'input_ids': processed_data['input_ids'][:train_size],
        'attention_mask': processed_data['attention_mask'][:train_size],
        'labels': processed_data['labels'][:train_size]
    }
    val_data = {
        'input_ids': processed_data['input_ids'][train_size:],
        'attention_mask': processed_data['attention_mask'][train_size:],
        'labels': processed_data['labels'][train_size:]
    }
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_data)
    val_dataloader = create_dataloader(val_data)
    
    # Initialize model
    model = SkillExtractionBERT()
    model.to(device)
    
    # Train model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=3
    )
    
    # Example prediction
    test_cv = """
    Data Scientist with expertise in Python, machine learning, and big data.
    Experience with SQL, Tableau, and cloud platforms.
    """
    
    predictions = predict_skills(model, test_cv, preprocessor, device)
    print("\nPredicted skills for test CV:")
    print(predictions)

if __name__ == "__main__":
    main() 