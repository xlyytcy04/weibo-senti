import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import json
from datetime import datetime


# Dataset class
class WeiboDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

    def __len__(self):
        return len(self.labels)


# Load dataset from CSV
def load_data(path):
    df = pd.read_csv(path, quoting=1, quotechar='"', escapechar="\\", encoding="utf-8")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return list(df["text"]), list(df["label"])


# Compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-chinese",
        help="Pretrained model name or path",
    )
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--val_file", type=str, default="val.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="",
        help="Optional description of the experiment",
    )
    args = parser.parse_args()

    # Generate unique experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.experiment_tag.replace(" ", "_")
    experiment_name = f"{timestamp}_{args.model_name.replace('/', '_')}_bs{args.batch_size}_ep{args.epochs}"
    if tag:
        experiment_name += f"_{tag}"
    experiment_dir = os.path.join("experiments", experiment_name)
    model_output_dir = os.path.join(experiment_dir, "model")
    log_output_dir = os.path.join(experiment_dir, "logs")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    # Load dataset
    train_texts, train_labels = load_data(args.train_file)
    val_texts, val_labels = load_data(args.val_file)

    # Load tokenizer and tokenize
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=128
    )
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # Prepare datasets
    train_dataset = WeiboDataset(train_encodings, train_labels)
    val_dataset = WeiboDataset(val_encodings, val_labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(set(train_labels))
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=log_output_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()

    # Save model
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # Save metrics
    with open(os.path.join(experiment_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save config
    config = vars(args)
    config["experiment_dir"] = experiment_dir
    with open(os.path.join(experiment_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("✅ Training complete.")
    print("🧪 Experiment saved to:", experiment_dir)
    print("📊 Final Evaluation:", metrics)


if __name__ == "__main__":
    main()
