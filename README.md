# Spam Email Classifier

## About the Project

This is a basic machine learning project made in Python to identify whether a message is **Spam** or **Ham (Not Spam)**.

I used a small dataset of email messages and trained a **Naive Bayes classifier** to make the predictions.

## Dataset

The dataset is stored in `spam_emails.csv`.

It has two columns:

* `message` – contains the email/message text
* `label` – tells whether the message is `spam` or `ham`

The dataset contains **99 messages** in total.

* 49 Spam messages
* 50 Ham messages

## Tools and Libraries Used

* Python
* Pandas
* Scikit-learn

The project uses **TF-IDF** to convert text into numerical data and **Multinomial Naive Bayes** for classification.

## How the Model Works

The basic process is:

1. Read the dataset using Pandas.
2. Separate the messages and their labels.
3. Split the data into training and testing sets.
4. Convert the messages into numerical values using TF-IDF.
5. Train the Naive Bayes classifier.
6. Test the classifier with the testing data.
7. Calculate the accuracy.

## Training and Testing

The dataset is divided into:

* **80% training data**
* **20% testing data**

The training data is used to teach the model, while the testing data is used to check how accurately the model can classify new messages.

## Algorithm Used

### Multinomial Naive Bayes

I used Multinomial Naive Bayes because it is a simple and commonly used algorithm for text classification.

### TF-IDF

TF-IDF is used to convert the words in the messages into numerical values so that the machine learning model can work with the text.

## How to Run

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the Python file:

```bash
python classifier.py
```

The program will display the number of messages used for training and testing along with the accuracy of the classifier.

## Results

The model was tested using 20% of the dataset.

The classifier achieved an accuracy of:

**95.00%**

This means that the model correctly classified **19 out of 20 test messages**.

The result shows that the model performed well on this small dataset.

## Conclusion

This project helped me understand the basic process of using a machine learning classifier for a real-world problem. It shows how text can be converted into numerical data using TF-IDF and then classified into Spam or Ham using Naive Bayes.

The accuracy of **95.00%** shows that the classifier was able to make correct predictions on most of the test messages. However, the dataset is small, so this result should not be considered representative of a real-world spam filtering system.

## Project Files

```text
spam-email-classifier/
│
├── spam_emails.csv
├── classifier.py
├── requirements.txt
└── README.md
```
