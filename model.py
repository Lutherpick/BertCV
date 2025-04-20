import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

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