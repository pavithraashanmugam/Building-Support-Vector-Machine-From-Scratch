
# SVM Classifier from Scratch

This repository contains an implementation of a Support Vector Machine (SVM) classifier built from scratch using Python. It does not rely on libraries like `sklearn` for classification but instead uses core `numpy` functions and a custom implementation of gradient descent to train the model.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [How the SVM Classifier Works](#how-the-svm-classifier-works)
- [Dataset](#dataset)
- [Steps to Train the Model](#steps-to-train-the-model)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

The SVM classifier here is designed to classify diabetes using a dataset containing medical information. The key steps are:
1. **Hyperplane Equation**: We aim to find a hyperplane that separates the classes (diabetic or not) in a high-dimensional space.
2. **Gradient Descent**: We optimize the weights and bias of the hyperplane using the gradient descent algorithm to minimize the loss function.

## Dependencies

The following Python libraries are required to run this model:

- `numpy`
- `pandas`
- `sklearn` (for dataset handling and splitting)
- `matplotlib` (optional, for visualization)

To install these dependencies, run:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How the SVM Classifier Works

### Equation of the Hyperplane
The equation of the hyperplane used in the SVM is:

\[
y = wx - b
\]

Where:
- `y` is the predicted label (either 0 or 1 for diabetes).
- `w` are the weights of the features.
- `x` are the input features.
- `b` is the bias term.

### Gradient Descent
We use gradient descent to minimize the hinge loss function. The update rules for the weights and bias are:

\[
w = w - \alpha \cdot dw
\]
\[
b = b - \alpha \cdot db
\]

Where:
- `dw` and `db` are the gradients with respect to the weight and bias, respectively.
- `\alpha` (alpha) is the learning rate.

The gradient descent step helps in finding the optimal values of `w` and `b` by moving towards minimizing the loss function.

## Dataset

The dataset used for training the model is the **Pima Indians Diabetes Database** which contains features related to diabetes diagnosis:

- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0 for non-diabetic, 1 for diabetic)

This dataset has been pre-processed for missing values and scaled to standardize the feature range.

## Steps to Train the Model

1. **Load and Preprocess the Data**:
   - Import the dataset.
   - Handle missing values (if any).
   - Standardize the features using `StandardScaler` for improved training performance.

2. **Train-Test Split**:
   - Split the dataset into training and testing sets using `train_test_split` from `sklearn`.

3. **Initialize Model Parameters**:
   - Initialize the weights (`w`) and bias (`b`) to zeros.
   - Define the learning rate (`alpha`), number of iterations, and regularization parameter (`lambda`).

4. **Gradient Descent for Model Optimization**:
   - Update weights (`w`) and bias (`b`) iteratively using the gradient descent rules.

5. **Evaluate the Model**:
   - Evaluate the accuracy on both training and testing datasets.

## Model Evaluation

After training, evaluate the accuracy of the model using:

- **Training Accuracy**: The proportion of correct predictions on the training data.
- **Testing Accuracy**: The proportion of correct predictions on the testing data.

## Usage

To use the classifier, follow these steps:

### 1. Import the Required Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

### 2. Load and Prepare the Dataset

```python
# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Split data into features (X) and target (Y)
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

### 3. Define the SVM Classifier

```python
class SVM_classifier():
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)
        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        return np.where(predicted_labels <= -1, 0, 1)
```

### 4. Train the Model

```python
classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
classifier.fit(X_train, Y_train)
```

### 5. Evaluate the Model

```python
# Evaluate on training data
train_predictions = classifier.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')

# Evaluate on testing data
test_predictions = classifier.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f'Testing Accuracy: {test_accuracy}')
```

### 6. Predict New Data

To make predictions on new data, standardize the input and use the trained model:

```python
input_data = [4, 110, 92, 0, 0, 37.6, 0.191, 30]
input_data_reshaped = np.asarray(input_data).reshape(1, -1)
input_data_standardized = scaler.transform(input_data_reshaped)

prediction = classifier.predict(input_data_standardized)
print(f'The person is {"diabetic" if prediction[0] == 1 else "not diabetic"}')
```

## Conclusion

This implementation of an SVM classifier is based on custom gradient descent for optimization. It works well for small to medium datasets like the Diabetes dataset used in this example. You can tweak the hyperparameters (learning rate, number of iterations, and regularization) to improve model performance for other datasets.
