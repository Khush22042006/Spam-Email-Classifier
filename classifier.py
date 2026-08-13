import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Read the dataset
data = pd.read_csv("spam_emails.csv")


# Separate messages and labels
X = data["message"]
y = data["label"]


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Create and train the Naive Bayes classifier
model = MultinomialNB()

model.fit(X_train, y_train)


# Make predictions on the testing data
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


print("Spam Email Classification")
print("-------------------------")
print("Total messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
print()
print("Accuracy:", round(accuracy * 100, 2), "%")
