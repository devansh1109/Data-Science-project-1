# Credit Card Fraud Detection

This repository contains a project that implements a Random Forest classifier for credit card fraud detection using Python. The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets?search=credit+card+fraud) available in CSV format.

## Project Overview

- **Data Loading**: The dataset is loaded and initial exploration is performed to understand the data structure.
- **Data Exploration**: Summary statistics and fraud detection metrics are calculated.
- **Feature Engineering**: Data is split into features (`X`) and target (`Y`), and correlation analysis is performed.
- **Model Training**: A Random Forest Classifier is trained on the dataset.
- **Evaluation**: The model's performance is evaluated using accuracy, precision, recall, and F1-Score.

## Installation

Ensure you have the required Python libraries installed. You can use the following command to install them:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
# Usage
- **Load the dataset**: Make sure the creditcard.csv file is located in the specified path.

- **Run the notebook**: Execute the code to perform the following:

Load and explore the dataset.
Preprocess the data.
Train and evaluate the Random Forest Classifier.
- **View Results**: The performance metrics of the model will be displayed.
