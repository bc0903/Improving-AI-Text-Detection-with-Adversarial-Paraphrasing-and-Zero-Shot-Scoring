"""
Core AI-generated text detection functionality using RoBERTa.
"""
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Import AdamW from torch.optim instead
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Set random seed for reproducibility
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
np.random.seed(seed_val)

def get_device():
    """
    Get the device to use for tensor operations.
    
    Returns:
        torch.device: CPU or CUDA device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_model_and_tokenizer(device=None):
    """
    Load the pre-trained RoBERTa tokenizer and model.
    
    Args:
        device (torch.device, optional): Device to place model on.
        
    Returns:
        tuple: (tokenizer, model)
    """
    if device is None:
        device = get_device()
        
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)

    return tokenizer, model

def preprocess_data(df, tokenizer, max_length=512):
    """
    Tokenize the text data and prepare it for the model.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'text' and 'label' columns.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
        max_length (int): Maximum sequence length.
        
    Returns:
        torch.utils.data.TensorDataset: Dataset ready for training/evaluation.
    """
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Truncate very long texts first to avoid memory issues
    truncated_texts = [text[:10000] if isinstance(text, str) else "" for text in texts]
    
    # Tokenize the text
    encodings = tokenizer(
        truncated_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)
    
    return TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        labels_tensor
    )

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_dataset (TensorDataset): Training dataset.
        val_dataset (TensorDataset): Validation dataset.
        test_dataset (TensorDataset): Test dataset.
        batch_size (int): Batch size for dataloaders.
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader

def evaluate_model(model, dataloader, device=None):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model (transformers.PreTrainedModel): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader with evaluation data.
        device (torch.device, optional): Device to place tensors on.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    predictions = []
    true_labels = []
    logits_all = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            logits_all.extend(logits.detach().cpu().numpy())
            
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    # Calculate ROC-AUC
    probs = torch.softmax(torch.tensor(logits_all), dim=1)[:, 1].numpy()
    auroc = roc_auc_score(true_labels, probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probs
    }

def train_model(model, train_dataloader, val_dataloader, epochs=4, device=None):
    """
    Train the model using the provided dataloaders.
    
    Args:
        model (transformers.PreTrainedModel): Model to train.
        train_dataloader (torch.utils.data.DataLoader): Training data.
        val_dataloader (torch.utils.data.DataLoader): Validation data.
        epochs (int): Number of training epochs.
        device (torch.device, optional): Device to place tensors on.
        
    Returns:
        list: List of dictionaries with training statistics.
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Calculate total steps
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    training_stats = []
    
    for epoch in range(epochs):
        print(f"======== Epoch {epoch + 1} / {epochs} ========")
        
        # Training
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation on validation set
        val_metrics = evaluate_model(model, val_dataloader, device)
        
        # Save stats
        training_stats.append({
            'epoch': epoch + 1,
            'training_loss': avg_train_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auroc': val_metrics['auroc']
        })
        
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Validation AUROC: {val_metrics['auroc']:.4f}")
    
    return training_stats

def plot_metrics(training_stats):
    """
    Plot training and validation metrics.
    
    Args:
        training_stats (list): List of dictionaries with training statistics.
    """
    stats_df = pd.DataFrame(training_stats)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=stats_df, x='epoch', y='training_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracy and F1
    plt.subplot(1, 2, 2)
    sns.lineplot(data=stats_df, x='epoch', y='val_accuracy', label='Accuracy')
    sns.lineplot(data=stats_df, x='epoch', y='val_f1', label='F1')
    sns.lineplot(data=stats_df, x='epoch', y='val_auroc', label='AUROC')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def predict_text(text, model, tokenizer, device=None):
    """
    Make a prediction on a single text input.
    
    Args:
        text (str): Text to classify.
        model (transformers.PreTrainedModel): Trained model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer.
        device (torch.device, optional): Device to place tensors on.
        
    Returns:
        dict: Prediction results.
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get prediction
    prediction = torch.argmax(probs, dim=-1).item()
    human_prob = probs[0, 0].item()
    ai_prob = probs[0, 1].item()
    
    return {
        "prediction": "AI-generated" if prediction == 1 else "Human-written",
        "human_probability": human_prob,
        "ai_probability": ai_prob
    }

def save_model(model, tokenizer, output_dir="roberta_ai_detector"):
    """
    Save the trained model and tokenizer.
    
    Args:
        model (transformers.PreTrainedModel): Model to save.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to save.
        output_dir (str): Directory to save to.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def split_data(df, test_size=0.3, random_state=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'text' and 'label' columns.
        test_size (float): Proportion of data to use for test+validation.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Split into train and temp (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    # Split temp into val and test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=random_state, 
        stratify=temp_df['label']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df