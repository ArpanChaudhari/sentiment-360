import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Data
train_df = pd.read_csv("data/laptop_train.csv")

X = train_df["text"]
y = train_df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Learn vocabulary fro, training data and trasform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using the same vocabulaty
X_test_tfidf = vectorizer.transform(X_test)

print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)


# Initialize and fit the logistic regression model

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
predictions = model.predict(X_test_tfidf)

# evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# step 5 Detailed report card
print(classification_report(y_test, predictions))
