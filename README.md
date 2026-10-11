
# Email Spam Classifier

## Overview
A machine learning web application that classifies text messages as spam or legitimate.

## Features
- Text preprocessing using TF-IDF
- Multinomial Naive Bayes classification
- Training and testing data split
- Test accuracy metric
- Interactive message classification
- Spam probability score
- Confusion matrix visualization
- Dataset preview

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## How to Run

Install dependencies:

pip install -r requirements.txt

Launch the application:

streamlit run app.py

## Dataset
The project uses a small, fictional sample dataset stored in `messages.csv`.

## Machine Learning Approach
TF-IDF converts text into numerical features, and Multinomial Naive Bayes learns to classify messages as spam or ham.

## Limitations
The dataset is small and intended for educational purposes. The model is not suitable for dependable real-world spam filtering without a larger, representative dataset and further evaluation.
