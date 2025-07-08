import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import json

# Dataset class for loading tokenized text and labels
class WeiboDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# Load dataset from CSV
def load_data(path):
    df = pd.read_csv(path, quoting=1, quotechar='"', escapechar='\\', encoding='utf-8')
    df = df.dropna(subset=['text', 'label'])  # Remove rows with missing data
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    return list(df['text']), list(df['label'])

# Metric computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-chinese', help='Pretrained model name or path')
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--val_file', type=str, default='val.csv')
    parser.add_argument('--output_dir', type=str, default='./model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Load data
    train_texts, train_labels = load_data(args.train_file)
    val_texts, val_labels = load_data(args.val_file)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Tokenize data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # Build datasets
    train_dataset = WeiboDataset(train_encodings, train_labels)
    val_dataset = WeiboDataset(val_encodings, val_labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=len(set(train_labels)))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()

