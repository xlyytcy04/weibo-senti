import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import json

# Load model and tokenizer
model_path = "./model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label mapping
with open("label_mapping.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

# Predict function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_class]

# Example usage
if __name__ == "__main__":
    text = input("Enter a Weibo comment: ")
    result = predict(text)
    print(f"Predicted emotion: {result}")

