import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a dataset
np.random.seed(42)
data_size = 1000

data = {
    "CustomerAge": np.random.randint(18, 70, data_size),
    "AnnualIncome": np.random.randint(20000, 120000, data_size),
    "CreditScore": np.random.randint(300, 850, data_size),
    "LoanAmount": np.random.randint(1000, 50000, data_size),
    "LoanApproved": np.random.choice([0, 1], data_size, p=[0.5, 0.5])  # Binary target variable
}

ml_dataset = pd.DataFrame(data)

# Step 2: Machine Learning Process
# Splitting data into features and target
X = ml_dataset[["CustomerAge", "AnnualIncome", "CreditScore", "LoanAmount"]]
y = ml_dataset["LoanApproved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model creation and training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
