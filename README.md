# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset: Import the dataset containing car prices along with relevant features.
3. Data Preprocessing: Manage missing data and select key features for the model, if required.
4. Split Data: Divide the dataset into training and testing subsets.
5. Train Model: Build a linear regression model and train it using the training data.
6. Make Predictions: Apply the model to predict outcomes for the test set.
7. Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9. Output Results: Present the predictions and evaluation metrics.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("food_items.csv")

# Inspect the dataset
print("Name:Barath B")
print("Reg. No:25009091")
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# Model parameters
penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000

# Define logistic regression model
l2_model = LogisticRegression(
    random_state=123,
    penalty=penalty,
    multi_class=multi_class,
    solver=solver,
    max_iter=max_iter
)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

```

## Output:
<img width="806" height="645" alt="image" src="https://github.com/user-attachments/assets/60e573d9-c85e-4fb9-a96e-083697c4bc39" />
<img width="574" height="649" alt="image" src="https://github.com/user-attachments/assets/cd316f85-59dc-432b-9178-4dd5494ceb98" />
<img width="597" height="381" alt="image" src="https://github.com/user-attachments/assets/5269a1a0-b801-465f-a253-a02fa8fd0114" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
