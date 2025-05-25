# Email Spam Detection

[Live Demo](https://email-spam-detection-grserxnqxeenspkgtcfff6.streamlit.app/)  
🎯 Automatically identify and flag spam emails using machine learning and NLP techniques.

This project classifies emails as spam or ham (not spam) using a supervised learning pipeline. It leverages Natural Language Processing (NLP) to transform email text into numerical features and applies a classification algorithm to accurately detect spam.

Ideal for learning how spam filters work and how text classification can be applied to real-world problems.



## Overview

This is a classic binary classification task. The dataset contains labeled emails marked as spam or not, and the model is trained to predict the correct label using:
- Text cleaning & preprocessing
- TF-IDF vectorization
- Naive Bayes classification
It's a simple yet effective machine learning pipeline that gives solid results and demonstrates the core ideas behind spam filtering systems.



## Features

-  Cleans and preprocesses raw email data
-  Converts email text into numerical features using TF-IDF
-  Trains a Naive Bayes classifier to detect spam
-  Evaluates performance using accuracy, precision, recall, and F1-score
-  Test set prediction and confusion matrix analysis



## Installation

Clone the repository and install the required libraries:

```bash

git clone https://github.com/Sreelakshmi-rv/Email-Spam-detection.git
cd Email-Spam-detection

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```




## Project Structure

```bash
├── spam.csv                 # Dataset containing labeled emails
├── spam_detection.ipynb     # Jupyter Notebook with complete pipeline
├── requirements.txt         # List of Python dependencies
└── README.md                # Project documentation
```



## Model Info

- Text Preprocessing: Lowercasing, punctuation removal, stopword removal, stemming.
- Vectorization: TF-IDF (Term Frequency–Inverse Document Frequency).
- Classifier: Multinomial Naive Bayes (ideal for text classification tasks).
- Evaluation: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.

## Sample Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 97%   |
| Precision | 96%   |
| Recall    | 95%   |
| F1 Score  | 95.5% |


## Future Improvements

- Add support for additional classifiers (e.g., Logistic Regression, SVM)
- Deploy the model using Streamlit for live predictions
- Use advanced NLP techniques (e.g., word embeddings, LSTM)
- Integrate with email clients for real-time spam filtering



## 🙌 Acknowledgments

- Dataset from Kaggle's Email Spam Classification dataset
- Inspired by real-world spam filtering systems
- Built as a portfolio project @ Digital University Kerala 🎓
