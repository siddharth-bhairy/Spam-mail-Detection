# Spam Detection using Logistic Regression

# Overview
This project implements a spam detection system using a Logistic Regression model trained on email data. The goal is to classify emails as either spam or ham (not spam) based on their content. The project involves data preprocessing, feature extraction using TF-IDF, model training, evaluation, and visualization of results.

# Table of Contents
Dataset
Installation
Project Structure
Code Walkthrough
Evaluation Metrics
Visualizations
How to Use
Conclusion
Dataset
The dataset used for this project is mail_data.csv, which contains email messages and their respective labels:
Category: Either "spam" or "ham".
Message: The content of the email.
The dataset is preprocessed to handle missing values and to label encode the categories (0 for spam, 1 for ham).

# Code Walkthrough
Data Collection & Preprocessing:

Load the dataset and handle missing values.
Encode the labels: "spam" as 0 and "ham" as 1.
Splitting the Data:

Split the dataset into training (80%) and testing (20%) sets.
Feature Extraction:

Use TF-IDF Vectorizer to transform the email content into numerical features.
Model Training:

Train a Logistic Regression model on the training data.
Model Evaluation:

Evaluate the model on both training and testing datasets.
Calculate the accuracy and generate a confusion matrix.
Visualization:

Plot class distribution, top features, accuracy comparison, and confusion matrix.
Evaluation Metrics
Accuracy:

Training Accuracy: 0.98776 (example value)
Test Accuracy: 0.96543 (example value)
Confusion Matrix:

Provides insights into the number of true positives, true negatives, false positives, and false negatives.
# Visualizations
Class Distribution:

A pie chart showing the proportion of spam vs. ham emails in the dataset.
Top 10 Important Features:

A bar chart showing the top 10 words with the highest TF-IDF scores.
Accuracy Comparison:

A bar plot comparing the accuracy of the model on the training and testing datasets.
Confusion Matrix:

A heatmap showing the performance of the model in terms of true/false positives and negatives.


# Conclusion
This project demonstrates a simple yet effective approach for detecting spam emails using logistic regression. The model achieves good accuracy and provides interpretable results. Future improvements could include using more advanced models like Naive Bayes, Support Vector Machine, or deep learning approaches for better performance.
