# Weibo Sentiment Analysis

This repository provides scripts for training a BERT-based sentiment classifier and for making predictions on single Weibo comments.

## Training

Run the training script with default settings:

```bash
python train.py --model_name bert-base-chinese
```

The trained model and logs will be saved under `experiments/`.

## Prediction

After training finishes you can predict the emotion of a new comment with `predict.py`. Provide the path to the saved model directory using `--model_path`:

```bash
python predict.py --model_path experiments/your_experiment/model
```

If omitted, `--model_path` defaults to `./model`.

You will be prompted to enter a comment and the predicted emotion will be printed.
