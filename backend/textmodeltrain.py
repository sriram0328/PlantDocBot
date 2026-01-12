# train_text_model_38_classes.py
"""
Script to retrain the text model on all 38 PlantVillage classes
using the ButterChicken98/plantvillage-image-text-pairs dataset
"""

import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import ast

# -------------------- CONFIGURATION --------------------
MODEL_NAME = "bert-base-uncased"  # or "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
OUTPUT_DIR = "./plant_disease_text_model_38classes"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# -------------------- LOAD DATASET --------------------
print("Loading dataset from HuggingFace...")
dataset = load_dataset("ButterChicken98/plantvillage-image-text-pairs")

# Convert to pandas for easier manipulation
df = pd.DataFrame(dataset['train'])

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\nFirst few rows:")
print(df.head())

# -------------------- PREPROCESS --------------------
# The dataset has: 'image', 'caption' (label), 'captions' (list of descriptions)
# 'caption' is the disease label
# 'captions' is a list of text descriptions we'll use for training

# Normalize labels (same as your backend)
def normalize_label(label):
    return str(label).replace("___", " ").replace("_", " ").lower().strip()

# Extract the disease label from 'caption' column
df['disease_label'] = df['caption']
df['normalized_label'] = df['disease_label'].apply(normalize_label)

# The 'captions' column contains lists of text descriptions
# We need to expand this so each text gets its own row
expanded_rows = []
for idx, row in df.iterrows():
    disease_label = row['disease_label']
    normalized_label = row['normalized_label']
    
    # Handle the captions - it might be a string representation of a list or an actual list
    captions = row['captions']
    if isinstance(captions, str):
        try:
            # Try to parse if it's a string representation of a list
            captions = ast.literal_eval(captions)
        except:
            # If it fails, treat it as a single caption
            captions = [captions]
    elif not isinstance(captions, list):
        captions = [str(captions)]
    
    # Create a row for each caption
    for caption_text in captions:
        if caption_text and str(caption_text).strip():  # Skip empty captions
            expanded_rows.append({
                'text': str(caption_text).strip(),
                'disease_label': disease_label,
                'normalized_label': normalized_label
            })

# Create new dataframe with expanded rows
df_expanded = pd.DataFrame(expanded_rows)

print(f"\nOriginal dataset: {len(df)} samples")
print(f"Expanded dataset: {len(df_expanded)} samples (with multiple captions per image)")
print(f"\nUnique classes: {df_expanded['normalized_label'].nunique()}")
print(f"\nClass distribution:")
print(df_expanded['normalized_label'].value_counts().sort_index())

# Remove any empty texts
df_expanded = df_expanded[df_expanded['text'].str.strip() != '']

print(f"\nFinal dataset size: {len(df_expanded)} samples")

# Check if we have all 38 classes
unique_classes = df_expanded['normalized_label'].unique()
print(f"\nTotal unique classes found: {len(unique_classes)}")
if len(unique_classes) < 38:
    print(f"⚠ Warning: Expected 38 classes but found {len(unique_classes)}")
    print("Missing classes might affect model performance")

# -------------------- ENCODE LABELS --------------------
label_encoder = LabelEncoder()
df_expanded['label_encoded'] = label_encoder.fit_transform(df_expanded['normalized_label'])

print(f"\nLabel encoder classes: {len(label_encoder.classes_)}")
print(f"\nAll classes:")
for i, cls in enumerate(sorted(label_encoder.classes_)):
    count = (df_expanded['normalized_label'] == cls).sum()
    print(f"  {i+1:2d}. {cls:50s} ({count:5d} samples)")

# Save label encoder
with open("label_encoder_38classes.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("\n✓ Saved label encoder to 'label_encoder_38classes.pkl'")

# -------------------- TRAIN/VAL SPLIT --------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_expanded['text'].tolist(),
    df_expanded['label_encoded'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_expanded['label_encoded']
)

print(f"\nTrain samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# -------------------- TOKENIZATION --------------------
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PlantDiseaseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = PlantDiseaseDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = PlantDiseaseDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

print("✓ Datasets created")

# -------------------- MODEL --------------------
num_labels = len(label_encoder.classes_)
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

print(f"✓ Model loaded with {num_labels} output classes")

# -------------------- TRAINING --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=LEARNING_RATE,
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n" + "="*60)
print("Starting training...")
print("="*60)

trainer.train()

# -------------------- SAVE MODEL --------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "="*60)
print(f"✓ Model saved to {OUTPUT_DIR}")
print("="*60)

# -------------------- EVALUATION --------------------
print("\nFinal evaluation:")
results = trainer.evaluate()
print(results)

# -------------------- TEST PREDICTIONS --------------------
print("\n" + "="*60)
print("Testing predictions...")
print("="*60)

test_cases = [
    "Yellow spots on leaves with brown edges",
    "Healthy green leaves with smooth texture",
    "Powdery white substance covering the leaf surface",
    "Dark circular lesions with yellow halo around them",
    "Wilting leaves with brown spots",
    "Mosaic pattern of light and dark green on leaves",
    "Leaf edges turning brown and crispy",
    "Small dark spots with concentric rings"
]

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"\nDevice: {device}")
print("\nTest predictions:")

for i, text in enumerate(test_cases, 1):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        
        # Get top 3 predictions
        top3_probs, top3_idx = probs.topk(3, dim=1)
        
        predicted_class = logits.argmax(-1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = probs[0][predicted_class].item()
    
    print(f"\n{i}. Text: '{text}'")
    print(f"   Predicted: {predicted_label} (confidence: {confidence:.3f})")
    print(f"   Top 3:")
    for j, (idx, prob) in enumerate(zip(top3_idx[0], top3_probs[0]), 1):
        label = label_encoder.inverse_transform([idx.item()])[0]
        print(f"      {j}. {label}: {prob.item():.3f}")

print("\n" + "="*60)
print("✓ Training complete!")
print("="*60)
print(f"\nOutput files:")
print(f"  - Model: {OUTPUT_DIR}/")
print(f"  - Label Encoder: label_encoder_38classes.pkl")
print(f"  - Training logs: ./logs/")
print("\nYou can now use this model in your backend by updating:")
print("  1. Tokenizer path: 'plant_disease_text_model_38classes'")
print("  2. Model path: 'plant_disease_text_model_38classes'")
print("  3. Label encoder: 'label_encoder_38classes.pkl'")