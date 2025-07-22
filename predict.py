import argparse
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load label mapping
with open("label_mapping.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_class]


def main():
    parser = argparse.ArgumentParser(description="Predict emotion for a Weibo comment")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model",
        help="Path to the trained model directory",
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    text = input("Enter a Weibo comment: ")
    result = predict(text, tokenizer, model)
    print(f"Predicted emotion: {result}")


if __name__ == "__main__":
    main()

