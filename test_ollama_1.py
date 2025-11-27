import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
X = df_train["text"].tolist()
y = df_train["emotion"].tolist()

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)

# Embeddings
def embed_text(text, model="snowflake-arctic-embed"):
    resp = ollama.embeddings(model=model, prompt=text)
    return resp["embedding"]

def embed_texts(texts, model="snowflake-arctic-embed", max_workers=8):
    vectors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_text, t, model): t for t in texts}
        for f in tqdm(as_completed(futures), total=len(texts), desc="Embedding texts"):
            vectors.append(f.result())
    return np.array(vectors)

print("▶ Generating TRAIN embeddings...")
train_emb = embed_texts(X_train)

print("▶ Generating VAL embeddings...")
val_emb = embed_texts(X_val)

print("▶ Generating TEST embeddings...")
test_emb = embed_texts(df_test["text"].tolist())

# Training
clf = xgb.XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",           # GPU-compatible
    predictor="gpu_predictor"
)

clf.fit(train_emb, y_train)

# Validate
val_pred = clf.predict(val_emb)
f1 = f1_score(y_val, val_pred, average="macro")
print(f"\nVALIDATION MACRO F1: {f1:.4f}\n")

# Predict test
test_pred_ids = clf.predict(test_emb)
test_pred_labels = label_enc.inverse_transform(test_pred_ids)

# Save
df_out = pd.DataFrame({
    "id": df_test["id"].tolist(),
    "emotion": test_pred_labels
})

df_out.to_csv("ollama_1.csv", index=False)
print("Saved to ollama_1.csv")
