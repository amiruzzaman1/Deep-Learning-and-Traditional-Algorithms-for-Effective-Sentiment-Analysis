# Deep-Learning-and-Traditional-Algorithms-for-Effective-Sentiment-Analysis
## Overview
This project focuses on performing sentiment analysis on Twitter data using a variety of machine learning and deep learning techniques. It compares the effectiveness of traditional algorithms and deep learning models in analyzing sentiment trends across a large-scale dataset.

## Key Features

- **Algorithmic Variety:** Incorporates a range of traditional machine learning models (RF, SVM, DT, LR, NB) and deep learning architectures (CNN, ANN, RNN) for comprehensive sentiment analysis.
- **Data Preprocessing:** Rigorous preprocessing steps including tokenization, stemming/lemmatization, and removal of stopwords and special characters ensure clean and standardized text data.
- **Feature Representation:** Utilizes Bag of Words (BoW) and potentially other feature representation techniques to transform text into numerical vectors suitable for machine learning models.
- **Model Evaluation:** Detailed evaluation metrics such as accuracy, precision, recall, and F1-score provide insights into the performance and generalization capabilities of each model.

## Dataset

The Twitter sentiment analysis dataset comprises 69,491 unique entries categorized into Negative, Positive, Neutral, and Irrelevant sentiments. This balanced distribution facilitates a comprehensive analysis of sentiment trends and opinions expressed on Twitter.

### Sentiment Categories:
- **Negative:** 29.9%
- **Positive:** 28.2%
- **Neutral:** 24.6%
- **Irrelevant:** 17.3%

## Colab Notebook (Click to View)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sq6CLAzN6u2vu3VnSiC7hsQqoK1fgHWk?usp=sharing)
## Dataset Overview
**Dataset Link:** [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

![image](https://github.com/user-attachments/assets/8ecc7aa2-6618-4915-84fa-9c4356034acb)
![image](https://github.com/user-attachments/assets/6c22dd13-836f-418f-8cf0-7abbf4ae8e3c)
![image](https://github.com/user-attachments/assets/b21b6ef9-527f-489d-9756-2ca130cba0e9)
![image](https://github.com/user-attachments/assets/76add8ad-3cb9-4eac-a67d-48fddb6fcbb0)
![image](https://github.com/user-attachments/assets/d17a2d55-c4f8-4496-896c-0049ac258822)

## Algorithm Implementation
The project begins with thorough dataset preparation, including cleaning, tokenization, and transformation into a Bag of Words (BoW) representation. This structured approach ensures the data is optimized for training across various machine learning models: Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), Recurrent Neural Networks (RNN), Random Forest (RF), Decision Tree (DT), Logistic Regression (LR), and Support Vector Machine (SVM). Models are trained using an 80-20 split for training and testing datasets to evaluate their performance metrics like accuracy, precision, recall, and F1-score. This systematic evaluation aids in selecting the most effective model for accurately predicting and classifying emotional sentiments from Twitter data.


## Result
### TRADITIONAL ALGORITHMS

| Algorithm              | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Random Forest (RF)     | 0.90     | 0.91      | 0.90   | 0.90     |
| Support Vector Machine | 0.90     | 0.77      | 0.77   | 0.77     |
| Decision Tree (DT)     | 0.75     | 0.75      | 0.75   | 0.75     |
| Logistic Regression (LR)| 0.73    | 0.73      | 0.73   | 0.73     |
| Naive Bayes (NB)       | 0.71     | 0.67      | 0.71   | 0.69     |

### DEEP LEARNING ALGORITHMS

| Model                  | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|------------------------|-------------------|---------------------|---------------|-----------------|
| CNN                    | 0.9597            | 0.9550              | 0.0919        | 0.3552          |
| ANN                    | 0.9491            | 0.9220              | 0.1227        | 0.7047          |
| RNN                    | 0.9414            | 0.9210              | 0.1446        | 0.5074          |


#### CNN
![image](https://github.com/user-attachments/assets/bcb77852-3f62-4f1a-95da-aff6a7e8196c)
#### ANN
![image](https://github.com/user-attachments/assets/e73edb5e-dce9-4e1a-baa9-64f48cca2296)
#### RNN
![image](https://github.com/user-attachments/assets/957edace-911b-4cd7-b9a7-d9c8da91c466)
