import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# open dataset files
def main():
    base_dir = os.path.dirname(__file__)

    with open(os.path.join(base_dir, "dataset/train/usual_train.txt"), encoding='utf-8') as f:
        train_data = json.load(f)

    with open(os.path.join(base_dir, "dataset/eval/usual_eval_labeled.txt"), encoding='utf-8') as f:
        val_data = json.load(f)

# turn into DataFrame
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)

# drop NA
    df_train = df_train[df_train['label'].notna()]
    df_val = df_val[df_val['label'].notna()]

# encoding labels
    le = LabelEncoder()
    df_train['label'] = le.fit_transform(df_train['label'])
    df_val['label'] = le.transform(df_val['label'])

# save label mapping
    label_map = {label: int(idx) for label, idx in zip(le.classes_, le.transform(le.classes_))}
    with open(os.path.join(base_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

# rename columns
    df_train = df_train.rename(columns={"content": "text"})
    df_val = df_val.rename(columns={"content": "text"})

# save as csv
    df_train[['text', 'label']].to_csv(os.path.join(base_dir, "train.csv"), index=False)
    df_val[['text', 'label']].to_csv(os.path.join(base_dir, "val.csv"), index=False)


if __name__ == "__main__":
    main()

