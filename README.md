# Spam Email Classification Using Python

## Objective

The objective of this project is to build a simple machine learning classifier that can classify messages as Spam or Ham (Not Spam).

## Dataset

The dataset contains sample messages labeled as:

- Spam
- Ham

The `message` column contains the text of the message and the `label` column contains the classification.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Naive Bayes

## Methodology

The project follows these steps:

1. Load the dataset using Pandas.
2. Separate the messages and their labels.
3. Divide the dataset into training and testing data.
4. Convert the text into numerical values using TF-IDF.
5. Train a Multinomial Naive Bayes classifier.
6. Predict the labels of the testing messages.
7. Calculate the accuracy of the classifier.

## Machine Learning Algorithm

### Naive Bayes

Multinomial Naive Bayes is used as the classification algorithm. It is a simple and commonly used algorithm for text classification.

### TF-IDF

TF-IDF is used to convert text messages into numerical features that can be processed by the machine learning algorithm.

## Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The training data is used to train the classifier, while the testing data is used to evaluate its performance.

## How to Run

First install the required libraries:

```bash
pip install -r requirements.txt
