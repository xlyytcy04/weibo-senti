# Weibo Sentiment Classification

This project contains scripts for preparing a Chinese Weibo sentiment dataset, training a BERT-based classifier and running inference.

## Dataset structure

The `dataset/` directory is organised as follows:

```
dataset/
    train/                # training data
        usual_train.txt
        usual_train.xlsx
        virus_train.txt
        virus_train.xlsx
    eval/                 # validation data
        usual_eval.txt
        usual_eval.xlsx
        usual_eval_labeled.txt
        usual_eval_labeled.xlsx
        virus_eval.txt
        virus_eval.xlsx
        virus_eval_labeled.txt
        virus_eval_labeled.xlsx
    test/
        mixed/            # mixed with dummy data
            usual_test.txt
            usual_test.xlsx
            usual_test_labeled.xlsx
            virus_test.txt
            virus_test.xlsx
            virus_test_labeled.txt
            virus_test_labeled.xlsx
        real/             # real evaluation data
            usual_test.txt
            usual_test.xlsx
            usual_test_labeled.txt
            usual_test_labeled.xlsx
            virus_test.txt
            virus_test.xlsx
            virus_test_labeled.txt
            virus_test_labeled.xlsx
```

The raw `*.txt` files store data in JSON array format with fields `id`, `content` and optionally `label`. The accompanying `*.xlsx` files contain the same information in Excel format. The dataset README (see `dataset/readme.txt`) notes the dataset sizes and file naming scheme【F:dataset/readme.txt†L1-L14】 as well as a sample JSON structure【F:dataset/readme.txt†L17-L34】.

A mapping from emotion labels to numeric ids is stored in `label_mapping.json`:

```
{
  "angry": 0,
  "fear": 1,
  "happy": 2,
  "neutral": 3,
  "sad": 4,
  "surprise": 5
}
```

## Preparing training data

Run `prepare_data.py` to convert the raw JSON data into `train.csv` and `val.csv` with numeric labels. The script also writes `label_mapping.json`.

```bash
python prepare_data.py
```

## Training

`train.py` uses HuggingFace Transformers to fine-tune a BERT-based classifier. Example command:

```bash
python train.py \
  --model_name bert-base-chinese \
  --epochs 3 \
  --batch_size 64
```

Each training run saves results under `experiments/<timestamp>_bert-base-chinese_bs64_ep3/` and updates `experiments_summary.csv`.

## Inference

After training, copy the desired model checkpoint to `./model` (or point `predict.py` to a directory containing `pytorch_model.bin` and `vocab.txt`). Run:

```bash
python predict.py
```

The script will prompt for a sentence and output the predicted emotion label.

## Requirements

Install required packages via pip:

```bash
pip install torch pandas scikit-learn transformers
```

PyTorch version should match your CUDA/CPU environment.

