
---

# Customer Churn Prediction using Artificial Neural Network

## Overview

This project involves building an Artificial Neural Network (ANN) for predicting customer churn. The dataset used contains various customer attributes, and the ANN is trained to predict whether a customer is likely to leave the bank.

![1698802120424](https://github.com/vishal815/Customer-Churn-Prediction-using-Artificial-Neural-Network/assets/83393190/a7c7acbd-df24-4877-b4d4-7dc3ab0a5da6)


## Files

- **artificial_neural_network(Customer Churn Prediction).ipynb**: Jupyter Notebook containing the code for data preprocessing, model building, training, and evaluation.
- **Churn_Modelling.csv**: Dataset used for training and testing the ANN.

## Workflow

1. **Importing Libraries**: Necessary libraries such as NumPy, Pandas, TensorFlow, and Keras are imported.
2. **Data Preprocessing**: The dataset is loaded, and data preprocessing steps include handling categorical data, label encoding, one-hot encoding, splitting the dataset, and feature scaling.
3. **Building the ANN**: A Sequential model is created using TensorFlow and Keras. The model architecture consists of an input layer, two hidden layers with ReLU activation, and an output layer with sigmoid activation.
4. **Training the ANN**: The model is compiled using the Adam optimizer and binary crossentropy loss. It is then trained on the training set for 100 epochs.
5. **Making Predictions and Evaluating the Model**: Predictions are made on the test set, and the model's performance is evaluated using a confusion matrix and accuracy score.

## Results

- **Accuracy**: The trained model achieves an accuracy of approximately 86.3% on the test set.

## Prediction Example

An example is provided where the model predicts whether a customer with specific attributes will leave the bank. The model predicts that the customer stays.

## Important Notes

1. Ensure input values are formatted as a double pair of square brackets for predictions.
2. For categorical variables, use one-hot encoding, and be careful about the order of columns.

### Option 1: Google Colab

1. Open the Jupyter Notebook in Google Colab by clicking on [artificial_neural_network(Customer_Churn_Prediction).ipynb](https://colab.research.google.com/github/vishal815/Customer-Churn-Prediction-using-Artificial-Neural-Network/blob/main/artificial_neural_network(Customer_Churn_Prediction).ipynb).

2. Execute each cell in order.

### Option 2: Local Environment

1. Download the Jupyter Notebook [artificial_neural_network(Customer_Churn_Prediction).ipynb](https://github.com/vishal815/Customer-Churn-Prediction-using-Artificial-Neural-Network/blob/main/artificial_neural_network(Customer_Churn_Prediction).ipynb) and open it in a Jupyter Notebook environment with the required dependencies.
