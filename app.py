import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="wide"
)

# Title
st.title("Social Media Sentiment Analyzer")
st.markdown("*Machine Learning Project - Analyze tweets, reviews, and comments for sentiment*")

# Load model
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses **Logistic Regression** model trained on Twitter US Airline Sentiment dataset.

    **Model Performance:**
    - Accuracy: 79.34%
    - Precision: 76.87%
    - Recall: 68.22%
    - F1-Score: 71.39%
    """)

    st.header("Example Texts")
    st.write("**Positive:** *I love this product! Amazing quality!*")
    st.write("**Negative:** *Worst service ever, very disappointed.*")
    st.write("**Neutral:** *The package arrived on time.*")

# Main area - two tabs
tab1, tab2 = st.tabs(["🔍 Single Text Analysis", "📁 Batch Analysis"])

# TAB 1: Single Text Analysis
with tab1:
    st.subheader("Enter your text below:")

    user_input = st.text_area(
        "",
        height=150,
        placeholder="Write a tweet, review, or comment here...",
        key="single_input"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Analyze Sentiment", use_container_width=True)

    if analyze_button and user_input:
        with st.spinner("Analyzing..."):
            try:
                model, vectorizer = load_model()

                cleaned = clean_text(user_input)
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]
                probs = model.predict_proba(features)[0]

                # FIX: Handle both string and integer predictions
                if isinstance(prediction, str):
                    if prediction.lower() == 'positive':
                        sentiment = "Positive"
                    elif prediction.lower() == 'negative':
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                    confidence = max(probs) * 100
                else:
                    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    sentiment = label_map[prediction]
                    confidence = max(probs) * 100

                st.markdown("---")
                st.subheader("Analysis Result")

                col1, col2 = st.columns([2, 1])

                with col1:
                    if sentiment == "Positive":
                        st.success(f"**Sentiment: Positive**")
                        st.balloons()
                    elif sentiment == "Negative":
                        st.error(f"**Sentiment: Negative**")
                    else:
                        st.warning(f"**Sentiment: Neutral**")

                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.progress(int(confidence))

                with col2:
                    st.write("**All probabilities:**")
                    if len(probs) == 3:
                        st.write(f"- Negative: {probs[0]*100:.1f}%")
                        st.write(f"- Neutral: {probs[1]*100:.1f}%")
                        st.write(f"- Positive: {probs[2]*100:.1f}%")
                    else:
                        for i, p in enumerate(probs):
                            st.write(f"- Class {i}: {p*100:.1f}%")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Try restarting the app or check if model files exist.")

    elif analyze_button and not user_input:
        st.warning("Please enter some text to analyze!")

# TAB 2: Batch Analysis
with tab2:
    st.subheader("Upload CSV file for batch analysis")
    st.markdown("CSV file should have a **'text'** column with the tweets/reviews.")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview of uploaded data:**")
        st.dataframe(df.head())

        if 'text' in df.columns:
            if st.button("Process Batch"):
                with st.spinner("Processing all texts..."):
                    model, vectorizer = load_model()
                    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

                    results = []
                    for text in df['text']:
                        cleaned = clean_text(str(text))
                        features = vectorizer.transform([cleaned])
                        pred = model.predict(features)[0]

                        if isinstance(pred, str):
                            results.append(pred.capitalize())
                        else:
                            results.append(label_map[pred])

                    df['sentiment'] = results

                    st.success("Processing complete!")
                    st.write("**Results:**")
                    st.dataframe(df)

                    sentiment_counts = df['sentiment'].value_counts()
                    fig, ax = plt.subplots()
                    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                    bar_colors = [colors.get(sentiment, 'blue') for sentiment in sentiment_counts.index]
                    sentiment_counts.plot(kind='bar', color=bar_colors, ax=ax)
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Distribution")
                    st.pyplot(fig)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
        else:
            st.error("! CSV must have a 'text' column!")

st.markdown("---")
st.markdown(
    "<center>Built with Streamlit | Logistic Regression Model | Accuracy: 79.34%</center>",
    unsafe_allow_html=True
)