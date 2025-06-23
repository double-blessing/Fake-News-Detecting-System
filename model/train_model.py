import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json # Import json for saving metrics

# --- Load Dataset ---
fake_path = os.path.join('..', 'data', 'Fake.csv')
true_path = os.path.join('..', 'data', 'True.csv')

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

df = pd.concat([fake_df, true_df], axis=0)
df['content'] = df['title'] + " " + df['text']
df = df[['content', 'label']].dropna()

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# --- Vectorization ---
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Model Training ---
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test_tfidf)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.4f}")

# Precision
precision = precision_score(y_test, y_pred, pos_label='REAL')
print(f"Model Precision: {precision:.4f}")

# Recall
recall = recall_score(y_test, y_pred, pos_label='REAL')
print(f"Model Recall: {recall:.4f}")

# F1-score
f1 = f1_score(y_test, y_pred, pos_label='REAL')
print(f"Model F1-score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Save Model and Metrics ---
MODEL_DIR = os.path.join('..', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, 'fake_news_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

# Save model accuracy to a JSON file
metrics = {"model_accuracy": round(acc, 4)}
with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f)

print("[✔] Model, vectorizer, and metrics saved successfully.")