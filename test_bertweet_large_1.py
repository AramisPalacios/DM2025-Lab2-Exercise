import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
from transformers import EarlyStoppingCallback

# Load data
with open('data/competition/final_posts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
posts = [item['root']['_source']['post'] for item in data]
df_tweets = pd.DataFrame(posts)
df_splits = pd.read_csv('data/competition/data_identification.csv')
df_emotions = pd.read_csv('data/competition/emotion.csv')
df_merged = df_tweets.merge(df_splits, left_on='post_id', right_on='id').drop(columns=['post_id'])
df_merged = df_merged.merge(df_emotions, how='left', on='id')
df_train = df_merged[df_merged['split'] == 'train'].copy()
df_test = df_merged[df_merged['split'] == 'test'].copy()

# Labels
labels = sorted(df_train["emotion"].dropna().unique())
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
df_train["label"] = df_train["emotion"].map(label2id)

# Dataset
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist() if "label" in df.columns else None
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx])
        return item

# Train/val split
df_train_, df_val_ = train_test_split(df_train, test_size=0.15, random_state=42, stratify=df_train["label"])

# Model & tokenizer
model_name = "vinai/bertweet-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))

# Datasets
train_dataset = TweetDataset(df_train_, tokenizer)
val_dataset = TweetDataset(df_val_, tokenizer)
test_dataset = TweetDataset(df_test, tokenizer)

# Metric
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    return {"f1": f1_score(p.label_ids, preds, average="macro")}

# Training
training_args = TrainingArguments(
    output_dir="./bertweet_large_emotion",
    num_train_epochs=3,                  # max epochs, early stopping will cut off if no improvement
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    save_total_limit=2,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # stops if val F1 doesn't improve for 3 evals
)

trainer.train()

# Predict test
preds = trainer.predict(test_dataset).predictions
pred_labels = [id2label[i] for i in preds.argmax(axis=1)]

# Save
pd.DataFrame({"id": df_test["id"].tolist(), "emotion": pred_labels}).to_csv("BERTweet_large_1.csv", index=False)
print("Saved BERTweet_large_1.csv")


