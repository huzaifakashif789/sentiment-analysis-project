# Social Media Sentiment Analyzer

## Project Overview
Machine Learning project that analyzes social media posts to detect sentiment: **Positive, Negative, or Neutral**.

## Dataset
- **Source:** Twitter US Airline Sentiment (Kaggle)
- **Total tweets:** 14,640
- **Distribution:** Negative (63%), Neutral (21%), Positive (16%)

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 79.34% |
| Precision | 76.87% |
| Recall | 68.22% |
| F1-Score | 71.39% |

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run app
streamlit run app.py