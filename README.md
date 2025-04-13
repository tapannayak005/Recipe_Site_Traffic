Data Scientist Professional Practical Exam Submission
Use this template to write up your summary for submission. Code in Python or R needs to be included.

📝 Task List
Your written report should include both code, output and written text summaries of the following:

Data Validation:
Describe validation and cleaning steps for every column in the data
Exploratory Analysis:
Include two different graphics showing single variables only to demonstrate the characteristics of data
Include at least one graphic showing two or more variables to represent the relationship between features
Describe your findings
Model Development
Include your reasons for selecting the models you use as well as a statement of the problem type
Code to fit the baseline and comparison models
Model Evaluation
Describe the performance of the two models based on an appropriate metric
Business Metrics
Define a way to compare your model performance to the business
Describe how your models perform using this approach
Final summary including recommendations that the business should undertake
Start writing report here..

Data Validation & Cleaning
Missing Values Handling:

Columns affected: calories, carbohydrate, sugar, and protein had missing values.

Solution: Missing values were replaced with column mean to maintain data consistency.

servings had some non-numeric values ("4 as a snack", "6 as a snack").

Solution: Converted to numeric and replaced invalid values with the median.

High Traffic was provided as "High" and "Low", converted to binary encoding (1 = High, 0 = Low).

Exploratory Data Analysis (EDA)
Visualizations Used:

Box Plot for calories and servings (to detect outliers).

Scatter Plot between carbohydrates and protein (to check relationships).

Heatmap (correlation matrix to find feature relationships).

Findings:

Recipes with higher protein and lower sugar tend to be more popular.

Certain categories (e.g., Breakfast, One Dish Meals) had a higher proportion of high-traffic recipes.

Model Development
Problem Type:

This is a binary classification problem (high_traffic: 1 or 0).

Models Used:

Baseline Model: Logistic Regression

Comparison Model: Random Forest

Code for Model Fitting:

from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.linear_model import LogisticRegression from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import recall_score

Splitting the dataset
X = df.drop(columns=['recipe', 'category', 'high_traffic']) y = df['high_traffic'] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Scaling the data
scaler = StandardScaler() X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test)

Logistic Regression Model
log_model = LogisticRegression() log_model.fit(X_train_scaled, y_train) y_pred_log = log_model.predict(X_test_scaled) recall_log = recall_score(y_test, y_pred_log)

Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) rf_model.fit(X_train, y_train) y_pred_rf = rf_model.predict(X_test) recall_rf = recall_score(y_test, y_pred_rf)

print(f'Logistic Regression Recall: {recall_log:.4f}') print(f'Random Forest Recall: {recall_rf:.4f}')

Model Evaluation
Since the business goal is 80% recall, Logistic Regression meets the requirement (98.23%), whereas Random Forest (77.88%) does not.

Business Metrics & Recommendations
Metric to Monitor:

Recall should remain above 80% for predicting popular recipes.

Track precision to balance avoiding unpopular recipe misclassification.

Final Recommendations:

Use Logistic Regression for selecting homepage recipes.

Monitor model performance and update regularly with new data.

Optimize recipe selection based on ingredient trends and user engagement data.
