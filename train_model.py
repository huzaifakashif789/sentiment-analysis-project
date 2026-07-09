import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

print("="*50)
print("SENTIMENT ANALYSIS - MACHINE LEARNING PROJECT")
print("="*50)

# Step 1: Load Dataset
print("\nStep 1: Loading dataset...")
df = pd.read_csv('Tweets.csv')

print(f"Dataset loaded! Total tweets: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Step 2: Data Exploration
print("\nStep 2: Data Exploration...")
print(f"\nSentiment Distribution:")
print(df['airline_sentiment'].value_counts())

# Step 3: Text Preprocessing
print("\nStep 3: Text Preprocessing...")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
print(" Text cleaning completed!")

# Step 4: Features and Target
print("\nStep 4: Preparing features...")
X = df['cleaned_text']
y = df['airline_sentiment']

# Step 5: Train-Test Split
print("\nStep 5: Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 6: Feature Extraction (TF-IDF)
print("\nStep 6: Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Features shape: {X_train_tfidf.shape}")

# Step 7: Model Training
print("\nStep 7: Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training completed!")

# Step 8: Predictions
print("\nStep 8: Making predictions...")
y_pred = model.predict(X_test_tfidf)

# Step 9: EVALUATION METRICS
print("\n" + "="*50)
print("STEP 9: MODEL EVALUATION METRICS")
print("="*50)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Precision, Recall, F1-Score (per class)
print("\nDetailed Classification Report:")
print("-"*40)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Macro and Weighted Averages
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\nMacro Average (all classes equally important):")
print(f"   Precision: {precision_macro:.4f}")
print(f"   Recall: {recall_macro:.4f}")
print(f"   F1-Score: {f1_macro:.4f}")

print("\nWeighted Average (accounts for class imbalance):")
print(f"   Precision: {precision_weighted:.4f}")
print(f"   Recall: {recall_weighted:.4f}")
print(f"   F1-Score: {f1_weighted:.4f}")

# Step 10: Confusion Matrix
print("\nConfusion Matrix:")
print("-"*40)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix - Sentiment Analysis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Step 11: Save Model
print("\nStep 11: Saving model and vectorizer...")
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved as 'sentiment_model.pkl'")
print("Vectorizer saved as 'vectorizer.pkl'")

# Step 12: Final Summary
print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)
print("\nFINAL RESULTS SUMMARY:")
print(f"   • Accuracy: {accuracy*100:.2f}%")
print(f"   • Precision (macro): {precision_macro:.4f}")
print(f"   • Recall (macro): {recall_macro:.4f}")
print(f"   • F1-Score (macro): {f1_macro:.4f}")
print(f"\n   • Model saved: sentiment_model.pkl")
print(f"   • Vectorizer saved: vectorizer.pkl")
print(f"   • Confusion matrix: confusion_matrix.png")