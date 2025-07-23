#Import Dependencies ===
!pip install --quiet scikit-learn pandas numpy matplotlib

import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# === Load Data ===

train_df = pd.read_csv('/content/train.csv')   
test_df  = pd.read_csv('/content/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test  shape: {test_df.shape}")
train_df.head()

# === Data Cleaning Function ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)            # remove URLs
    text = re.sub(r"\@\w+|\#", "", text)                           # remove mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                       # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()                       # collapse whitespace
    return text

# Apply cleaning
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df ['clean_text'] = test_df ['text'].apply(clean_text)

# === Label Encoding ===
# Drop any rows with missing sentiment
train_df.dropna(subset=['sentiment'], inplace=True)

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['sentiment'])
print("Classes:", list(le.classes_))

# === Train/Validation Split ===
X = train_df['clean_text']
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Val:", X_val.shape)

# === TF-IDF Vectorization ===
tfidf = TfidfVectorizer(
    max_features=20000,       # limit vocab size
    ngram_range=(1,2),        # unigrams + bigrams
    min_df=5,                 # ignore super-rare words
    stop_words='english'
)

# Learn vocabulary on train & apply to train/val/test
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(test_df['clean_text'])

# === Baseline Model — Logistic Regression ===
clf = LogisticRegression(
    solver='saga',
    penalty='l2',
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train_tfidf, y_train)

# Evaluate on validation set
y_val_pred = clf.predict(X_val_tfidf)
print("F1-score (macro):", f1_score(y_val, y_val_pred, average='macro'))
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# === Hyperparameter Tuning ===
param_grid = {
    'C': [0.1, 1.0, 5.0],
    'penalty': ['l2'],
    'solver': ['saga']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    param_grid,
    scoring='f1_macro',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train_tfidf, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

# Re-evaluate best model on validation
best_clf = grid.best_estimator_
y_val_pred = best_clf.predict(X_val_tfidf)
print("Val F1 (macro):", f1_score(y_val, y_val_pred, average='macro'))

# === Final Train on Full Training Set & Predict ===
# (Re-vectorize on full train set)
X_full_tfidf = tfidf.fit_transform(train_df['clean_text'])
y_full = train_df['label']

final_clf = LogisticRegression(
    solver='saga', C=1.0, max_iter=1000,
    class_weight='balanced', random_state=42
)
final_clf.fit(X_full_tfidf, y_full)

# Transform test set & predict
X_test_tfidf = tfidf.transform(test_df['clean_text'])
test_preds = final_clf.predict(X_test_tfidf)
test_labels = le.inverse_transform(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_labels
})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")