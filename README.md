
# AI-Based Customer Support Ticket Classification & Priority Prediction

An NLP and Machine Learning based system that automatically classifies customer support tickets into the appropriate support queue and predicts their priority level.

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to automate customer support ticket classification.

The system predicts:

- Support Queue — the department responsible for handling the ticket
- Priority — Low, Medium, or High

The trained models are integrated into a Streamlit web application.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF
- Linear Support Vector Machine (SVM)
- Streamlit
- Joblib
- Google Colab
- GitHub

## Machine Learning Workflow

Customer Support Dataset
→ Data Cleaning
→ Exploratory Data Analysis
→ Text Preprocessing
→ TF-IDF Feature Extraction
→ Model Training
→ Hyperparameter Tuning
→ Model Evaluation
→ Streamlit Application

## Models

The project evaluates multiple machine learning algorithms including:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine

Linear SVM was selected for the final classification tasks.

## Final Test Performance

| Task | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Queue Classification | Balanced Linear SVM | 50.44% | 48.50% |
| Priority Prediction | Linear SVM | 59.50% | 57.23% |

Macro F1 was considered along with accuracy because the support queue classes are imbalanced.

## Streamlit Application

The application allows users to enter a customer support ticket and receive predictions for:

- Support Queue
- Priority Level

Example:

Input:
My application keeps crashing whenever I try to open it.

Output:
Support Queue: IT Support
Priority: High

## Project Structure

AI-Customer-Support-Ticket-Classifier/

├── app.py
├── queue_model.pkl
├── queue_tfidf.pkl
├── priority_model.pkl
├── priority_tfidf.pkl
├── requirements.txt
└── README.md

## Future Improvements

- Add multilingual classification
- Improve performance on minority classes
- Experiment with transformer-based NLP models
- Add calibrated probability predictions
- Add automated ticket summarization
- Deploy the application publicly

## Author

Bhagyasree S B

BCA Data Science
Amrita Vishwa Vidyapeetham, Kochi Campus
