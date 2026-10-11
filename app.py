
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)

st.title("Email Spam Classifier")
st.write(
    "Enter a message to classify it as spam or legitimate "
    "using a machine learning model."
)

# Load and validate the dataset
df = pd.read_csv("messages.csv")

if not {"Message", "Label"}.issubset(df.columns):
    st.error("The dataset must contain Message and Label columns.")
    st.stop()

df = df.dropna(subset=["Message", "Label"])
df["Message"] = df["Message"].astype(str)
df["Label"] = df["Label"].str.strip().str.lower()

if set(df["Label"].unique()) != {"spam", "ham"}:
    st.error("The dataset must contain both spam and ham labels.")
    st.stop()

if df["Label"].value_counts().min() < 2:
    st.error("Each label needs at least two examples.")
    st.stop()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df["Message"],
    df["Label"],
    test_size=0.25,
    random_state=42,
    stratify=df["Label"]
)

# Create and train the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

# Evaluate on held-out test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Messages", len(df))
col2.metric("Training Messages", len(X_train))
col3.metric("Test Accuracy", f"{accuracy:.1%}")

st.warning(
    "This dataset is very small and illustrative. "
    "The accuracy shown is not a reliable measure of real-world "
    "spam detection performance."
)

st.divider()

# Message classification
st.subheader("Test a Message")

message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! Claim your free prize now.",
    height=130
)

if st.button("Classify Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM (legitimate)")

        probabilities = model.predict_proba([message])[0]
        classes = model.named_steps["classifier"].classes_
        scores = dict(zip(classes, probabilities))

        st.write(
            f"Model spam score: {scores.get('spam', 0):.1%}"
        )
        st.caption(
            "This score is the model's estimated probability, "
            "not a guarantee that the message is spam."
        )

st.divider()

# Confusion matrix
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["ham", "spam"],
    display_labels=["Not Spam", "Spam"],
    ax=ax,
    colorbar=False
)
ax.set_title("Test Set Classification Results")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Dataset preview
st.subheader("Sample Dataset")
st.dataframe(df, use_container_width=True, hide_index=True)
