import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

"""
Predictive Maintenance Model

This script generates a synthetic dataset using scikit-learn's `make_classification` function to
simulate sensor readings and a binary target indicating failure.
It then splits the data into training and testing sets, fits a logistic regression model,
and outputs accuracy and a classification report.

To run this script:
    python predictive_maintenance_model.py

It will print metrics to the console.
"""

def main():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, random_state=42)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()
