from datasets import load_dataset
import pandas as pd
import torch
from ai_detector import (
    preprocess_data, evaluate_model, load_model_and_tokenizer, get_device
)

# 1. Load HumanEval
dataset = load_dataset("openai_humaneval")
human_data = dataset['test']

# 2. Construct the evaluation DataFrame
# Use prompt + canonical_solution as full text
texts = [
    ex['prompt'] + ex['canonical_solution'] for ex in human_data
    if isinstance(ex['prompt'], str) and isinstance(ex['canonical_solution'], str)
]
df_humaneval = pd.DataFrame({
    'text': texts,
    'label': [0] * len(texts)  # All are human-written
})

# 3. Load tokenizer and model
device = get_device()
tokenizer, model = load_model_and_tokenizer(device)

# 4. Preprocess dataset
eval_dataset = preprocess_data(df_humaneval, tokenizer)

# 5. Create DataLoader
from torch.utils.data import DataLoader, TensorDataset
eval_loader = DataLoader(eval_dataset, batch_size=8)

# 6. Evaluate
results = evaluate_model(model, eval_loader, device)

# 7. Print metrics (note: no AI examples, so AUROC may be undefined)
print("Evaluation on HumanEval (human-written code only):")
print(f"Accuracy (should be close to 1.0): {results['accuracy']:.4f}")
print(f"F1 Score (human class): {results['f1']:.4f}")
print(f"AI Prob Mean: {results['ai_probability']:.4f}")
print(f"Human Prob Mean: {results['human_probability']:.4f}")
