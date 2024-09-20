# Task-03: Decision Tree Classifier for Predicting Customer Purchases

## Project Overview

In this project, we build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. For this purpose, we use the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

The dataset consists of information gathered from direct marketing campaigns of a Portuguese banking institution. The classification goal is to predict whether a customer will subscribe to a term deposit.

## Dataset

The **Bank Marketing dataset** contains the following key features:
- **Age**: Age of the customer.
- **Job**: Type of job.
- **Marital**: Marital status.
- **Education**: Education level.
- **Default**: Whether the customer has credit in default.
- **Balance**: The customer's account balance.
- **Housing**: Whether the customer has a housing loan.
- **Loan**: Whether the customer has a personal loan.
- **Contact**: Communication type (e.g., cellular, telephone).
- **Day**: Last contact day of the month.
- **Month**: Last contact month of the year.
- **Duration**: Last contact duration, in seconds.
- **Campaign**: Number of contacts performed during this campaign.
- **Pdays**: Number of days since the client was last contacted from a previous campaign.
- **Previous**: Number of contacts performed before this campaign.
- **Poutcome**: Outcome of the previous marketing campaign.
- **Y**: Whether the client subscribed to a term deposit (Yes/No).

## Steps Involved

### 1. Data Preprocessing
- **Handling Missing Data**: Identify and handle missing values, if any.
- **Encoding Categorical Features**: Convert categorical variables (e.g., job, marital status) into numerical representations using techniques like one-hot encoding or label encoding.
- **Feature Scaling**: Apply scaling techniques (e.g., MinMaxScaler or StandardScaler) to ensure uniformity in data ranges, which can improve classifier performance.
  
### 2. Model Building
- **Train/Test Split**: Split the data into training and testing sets using an 80/20 or 70/30 ratio.
- **Decision Tree Classifier**: Implement a decision tree classifier using Scikit-learn's `DecisionTreeClassifier`.
- **Hyperparameter Tuning**: Use grid search or random search to optimize the hyperparameters like `max_depth`, `min_samples_split`, `min_samples_leaf`, etc.

### 3. Model Evaluation
- **Accuracy Score**: Calculate the accuracy of the model on the test data.
- **Confusion Matrix**: Plot the confusion matrix to visualize the performance in terms of true positives, false positives, true negatives, and false negatives.
- **Precision, Recall, F1-Score**: Evaluate the model using precision, recall, and F1-score to understand how well the classifier balances between positive and negative predictions.
  
### 4. Visualizing the Decision Tree
- Plot the decision tree structure to visualize how the features are split and what rules are applied to classify the data.
  
### 5. Insights and Interpretations
After building and evaluating the decision tree classifier, we can interpret:
- Which features are the most important in predicting customer purchases.
- How different customer demographics or behaviors influence the likelihood of a purchase.
  
## Tools and Libraries
- **Python**: Programming language used for data analysis and model building.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For model building, evaluation, and hyperparameter tuning.
- **Matplotlib & Seaborn**: For visualization of the decision tree and performance metrics.

## Instructions for Running the Code
1. Clone or download the project files.
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. Download the dataset from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and place it in the project folder.
4. Open the `decision_tree_classifier.ipynb` file in Jupyter Notebook or Google Colab.
5. Run each cell to preprocess the data, build the classifier, and evaluate the results.

## File Structure
```
project_folder/
│
├── decision_tree_classifier.ipynb   # Jupyter Notebook for model building and analysis
├── bank_marketing.csv               # Bank Marketing dataset
└── README.md                        # This readme file
```

## Future Work
- Implement other classification models (e.g., Random Forest, SVM, Logistic Regression) to compare performance.
- Apply feature engineering techniques to improve model accuracy.
- Explore advanced decision tree algorithms like Gradient Boosting or XGBoost.
