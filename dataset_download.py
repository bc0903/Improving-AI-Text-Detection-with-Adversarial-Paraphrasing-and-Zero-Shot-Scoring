"""
Dataset downloading and preparation utilities for AI-generated text detection.
"""
import os
import pandas as pd
import requests
import random
import re
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def download_dataset(dataset_name="HC3", subset="en"):
    """
    Download a dataset for AI-generated text detection.
    Returns the path to the downloaded dataset file.
    
    Args:
        dataset_name (str): Name of the dataset to download.
        subset (str): Subset of the dataset to download. 
                    For HC3, options include "en", "finance", "all"
    
    Returns:
        str: Path to the downloaded dataset file, or None if download fails.
    """
    print(f"Downloading dataset {dataset_name}, subset {subset}...")
    
    # Try using Hugging Face datasets library first (preferred method)
    try:
        print(f"Loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(f"Hello-SimpleAI/{dataset_name}", subset)
        output_path = f"{dataset_name}_{subset}_hf.csv"
        
        # Extract human answers and chatgpt answers
        print("Extracting and saving data...")
        data = []
        
        # Process each example in the dataset
        for split in dataset.keys():
            for example in dataset[split]:
                human_answers = example.get('human_answers', [])
                chatgpt_answers = example.get('chatgpt_answers', [])
                
                # Add each human answer with label 0
                for answer in human_answers:
                    if answer and isinstance(answer, str) and len(answer.strip()) > 0:
                        data.append({
                            'text': answer,
                            'label': 0  # Human-written
                        })
                
                # Add each ChatGPT answer with label 1
                for answer in chatgpt_answers:
                    if answer and isinstance(answer, str) and len(answer.strip()) > 0:
                        data.append({
                            'text': answer,
                            'label': 1  # AI-generated
                        })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Total samples: {len(df)}, Human: {sum(df['label'] == 0)}, AI: {sum(df['label'] == 1)}")
        return output_path
        
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Falling back to direct download method...")

def prepare_dataset(file_path, sample_size=2000):
    """
    Prepare the dataset for the AI text detection task.
    
    Args:
        file_path (str): Path to the dataset file.
        sample_size (int): Maximum number of samples to include from each class.
        
    Returns:
        str: Path to the prepared dataset file.
    """
    print(f"Preparing {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check if file is already in the expected format (text, label)
    if 'text' in df.columns and 'label' in df.columns:
        print("Dataset already in expected format.")
        
        # Limit sample size if needed
        if len(df) > 2 * sample_size:
            # Make sure we have enough samples of each class
            human_count = sum(df['label'] == 0)
            ai_count = sum(df['label'] == 1)
            
            if human_count >= sample_size and ai_count >= sample_size:
                human_df = df[df['label'] == 0].sample(n=sample_size, random_state=RANDOM_SEED)
                ai_df = df[df['label'] == 1].sample(n=sample_size, random_state=RANDOM_SEED)
                combined_df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            else:
                # If we don't have enough of one class, take what we have
                human_sample_size = min(sample_size, human_count)
                ai_sample_size = min(sample_size, ai_count)
                
                human_df = df[df['label'] == 0].sample(n=human_sample_size, random_state=RANDOM_SEED)
                ai_df = df[df['label'] == 1].sample(n=ai_sample_size, random_state=RANDOM_SEED)
                combined_df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        else:
            combined_df = df
        
        # Save the prepared dataset
        output_path = "prepared_dataset.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Dataset prepared with {len(combined_df)} samples:")
        print(f"  - Human texts: {sum(combined_df['label'] == 0)}")
        print(f"  - AI texts: {sum(combined_df['label'] == 1)}")
        print(f"Saved to {output_path}")
        
        return output_path
    
    # If not in expected format, try to extract from HC3 format
    try:
        # Extract human and AI responses
        human_texts = []
        ai_texts = []
        
        # HC3 format has human_answers and chatgpt_answers columns
        for _, row in df.iterrows():
            # Get human answers (may be multiple)
            if 'human_answers' in df.columns and isinstance(row['human_answers'], str) and row['human_answers'] != '[]':
                try:
                    human_answers = eval(row['human_answers'])
                    if isinstance(human_answers, list):
                        human_texts.extend(human_answers)
                    else:
                        human_texts.append(human_answers)
                except:
                    pass
            
            # Get ChatGPT answers (may be multiple)
            if 'chatgpt_answers' in df.columns and isinstance(row['chatgpt_answers'], str) and row['chatgpt_answers'] != '[]':
                try:
                    chatgpt_answers = eval(row['chatgpt_answers'])
                    if isinstance(chatgpt_answers, list):
                        ai_texts.extend(chatgpt_answers)
                    else:
                        ai_texts.append(chatgpt_answers)
                except:
                    pass
        
        # Ensure we have text content
        human_texts = [text for text in human_texts if isinstance(text, str) and len(text) > 50]
        ai_texts = [text for text in ai_texts if isinstance(text, str) and len(text) > 50]
        
        # Limit sample size if needed
        sample_size = min(sample_size, min(len(human_texts), len(ai_texts)))
        
        # Randomly sample texts
        if len(human_texts) > sample_size:
            human_sample = random.sample(human_texts, sample_size)
        else:
            human_sample = human_texts
        
        if len(ai_texts) > sample_size:
            ai_sample = random.sample(ai_texts, sample_size)
        else:
            ai_sample = ai_texts
        
        # Create DataFrame with human (0) and AI (1) labels
        human_df = pd.DataFrame({'text': human_sample, 'label': 0})
        ai_df = pd.DataFrame({'text': ai_sample, 'label': 1})
        
        # Combine and shuffle
        combined_df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        print(f"Dataset prepared with {len(combined_df)} samples:")
        print(f"  - Human texts: {len(human_df)}")
        print(f"  - AI texts: {len(ai_df)}")
        
        # Save the prepared dataset
        output_path = "prepared_dataset.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        print("Creating a synthetic dataset instead...")
        return create_larger_sample_dataset(sample_size)

def get_dataset(dataset_name="HC3", subset="all", sample_size=2000):
    """
    Convenience function to download and prepare a dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download.
        subset (str): Subset of the dataset to download. 
                      For HC3, options include "en", "finance", "all"
        sample_size (int): Maximum number of samples to include from each class.
        
    Returns:
        pandas.DataFrame: Prepared dataset.
    """
    # Download the dataset
    dataset_path = download_dataset(dataset_name=dataset_name, subset=subset)
    
    if dataset_path is None:
        print("Dataset download failed. Using synthetic dataset.")
        dataset_path = create_larger_sample_dataset(sample_size=sample_size)
    
    # Prepare the dataset
    output_path = prepare_dataset(dataset_path, sample_size=sample_size)
    
    # Load the prepared dataset
    return pd.read_csv(output_path)