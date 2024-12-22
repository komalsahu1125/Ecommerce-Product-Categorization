# Ecommerce Product Categorization

## Project Overview

This project is part of a Data Science Hackathon focused on categorizing ecommerce products based on their descriptions. Using Logistic Regression and TF-IDF vectorization, the model predicts the product category from textual descriptions. The main goal of this project is to classify products into predefined categories using machine learning techniques.

## Description
This project was developed as part of a Data Science Hackathon to address the challenges of categorizing products in an ecommerce setting. Accurate product categorization is essential for improving customer experiences, reducing search friction, and enhancing product discoverability.

## Objectives
The key objectives of the project were:
- Data Exploration and Preparation
- Descriptive Analysis
- Feature Engineering/Text to Features
- Predictive Modeling
- Fine Tuning
- Enhancing Categorization Accuracy

## Features
- **Data Exploration and Preparation**: Understand key features, detect missing data, and handle inconsistencies.
- **Descriptive Analysis**: Identify data patterns and category distributions, visualize insights using word clouds and other techniques.
- **Feature Engineering**: Convert text descriptions into numerical features using techniques like TF-IDF and word embeddings.
- **Predictive Modeling**: Develop machine learning and deep learning models for product categorization.
- **Model Evaluation**: Evaluate models using metrics such as accuracy, F1 score, precision, and recall.
- **Comparison of Models**: Compare the performance of machine learning and deep learning models, highlighting that machine learning models provided better accuracy for this task.
---

## Dataset

The dataset used for this project is a collection of ecommerce product descriptions with their associated product categories.

### Features:
- `description`: Textual descriptions of products.
- `product_category_tree`: Target variable indicating the category of the product.

### Preprocessing Steps:
1. **Handle Missing Data**: Drop the `brand` column due to a high number of missing values and remove rows with any missing data.
2. **Label Encoding**: Convert the target variable (`product_category_tree`) to numerical labels using `LabelEncoder`.
3. **Word Embedding**: Convert the product descriptions into vectors for numerical representation of text data.

---

## Dependencies

The following Python libraries are required:
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

You can install these packages using:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

---

## Model Pipeline

### 1. Data Preprocessing
- **Dropping Columns**: Dropped the `brand` column with many missing values.
- **Removing Missing Rows**: Rows with missing values were dropped.

### 2. Label Encoding
- Converted the `product_category_tree` into numerical labels to be used by the Logistic Regression model.
- 
### 3. Train-Test Split
- Split the data into 80% training and 20% testing sets.
- 
### 4. Model Selection: 
- Trained the model on the vectorized data.

### 6. Hyperparameter Tuning: GridSearchCV
- Tuned the model's hyperparameters (`C` and `solver`) using grid search.

### 7. Evaluation
- Used the test dataset to evaluate the model's performance using accuracy, classification report, and confusion matrix.

### 8. Save Predictions
- Saved predictions for unseen test data into a CSV file.
- 
---

### Confusion Matrix

The confusion matrix provides a visual breakdown of how well the model performs across different categories. It shows the number of true positive, false positive, true negative, and false negative predictions for each category.

In this confusion matrix:
- **True Labels** are represented along the y-axis.
- **Predicted Labels** are represented along the x-axis.
- Diagonal elements represent correct predictions (true positives).
- Off-diagonal elements represent misclassifications (false positives and false negatives).

---

## Conclusion

The model performed exceptionally well, achieving a high accuracy and excellent precision, recall, and F1-scores across most categories. Categories like **Clothing**, **Jewellery**, and **Watches** saw perfect classification results, while categories like **Toys & School Supplies** had slightly lower performance.

---
