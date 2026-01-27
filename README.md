# DA_CLASSIFICATION

# Breast Cancer Classification using Logistic Regression

Overview:
This notebook demonstrates a machine learning workflow to predict breast cancer diagnosis (Malignant or Benign) using a Logistic Regression model. The data is sourced from a Snowflake database and processed using Python's data science libraries.

Data Source:
The dataset is retrieved from a Snowflake database, specifically from the CANCER table within the breast_cancer_db.PUBLIC schema.

Preprocessing Steps:
Diagnosis Mapping: The 'DIAGNOSIS' column, originally categorical ('M' for Malignant, 'B' for Benign), is converted into numerical representation (1 for Malignant, 0 for Benign).
Feature and Target Split: The dataset is split into features (X) and the target variable (y), with 'ID' and 'DIAGNOSIS' columns excluded from features.
Train-Test Split: The data is divided into training and testing sets with a 80/20 ratio (test_size=0.2) to evaluate model performance.
Feature Scaling: Numerical features are standardized using StandardScaler to ensure that all features contribute equally to the model training process.
Model Training:
A Logistic Regression model is used for classification. The model is trained on the scaled training data (x_train_scaled) and their corresponding labels (y_train).

Model Performance:
After training, the model's performance is evaluated on the test set. The key metrics are as follows:

Accuracy
0.9737

Confusion Matrix
[[70  1]
 [ 2 41]]
Classification Report
              precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Conclusion:
The Logistic Regression model achieved high accuracy in predicting breast cancer diagnosis, as indicated by the performance metrics.

# Diabetes Prediction using Decision Tree Classifier


Overview:
This notebook demonstrates the process of building and evaluating a Decision Tree Classifier for predicting diabetes based on various health indicators. The analysis involves data loading, initial exploration, model training, and performance evaluation.

Dataset:
The dataset used in this notebook is diabetes.csv, which contains various diagnostic measurements and a binary outcome variable indicating whether the patient has diabetes (1) or not (0).

Features:
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age in years
Target Variable:
Outcome: Class variable (0 or 1) indicating non-diabetic or diabetic
Analysis Steps
Load Data: The diabetes.csv file is loaded into a pandas DataFrame.
Data Exploration: Basic data checks are performed, including head(), isnull().sum(), info(), and describe() to understand the dataset structure, check for missing values, and view statistical summaries.
Data Splitting: The dataset is split into features (X) and target (y), and then further divided into training and testing sets using train_test_split.
Model Training (Gini Impurity): A Decision Tree Classifier is trained using the default gini criterion.
Model Evaluation (Gini Impurity): The model's accuracy, confusion matrix, and classification report are calculated and displayed.
Model Training (Entropy): Another Decision Tree Classifier is trained using the entropy criterion.
Model Evaluation (Entropy): The accuracy, confusion matrix, and classification report for the entropy-based model are calculated and displayed.
Visualization: Decision trees for both models are visualized using export_graphviz and graphviz.

# EEG Eye State Classification using Random Forest

Project Overview:
This project aims to classify the eye state (open or closed) of an individual based on Electroencephalogram (EEG) signals. Using machine learning techniques, specifically a Random Forest Classifier, we predict whether a person's eyes are open or closed based on various EEG sensor readings.

Dataset:
The dataset used for this project is EEG_Eye_State_Classification.csv. It contains 15 features, all of which are EEG measurements from different electrodes, and a target variable eyeDetection which indicates the eye state (0 for closed, 1 for open).

Features:

AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4: These are EEG measurements from various channels, representing brain activity.
Target Variable:

eyeDetection: Binary variable indicating the eye state (0: eyes closed, 1: eyes open).
Data Exploration Summary
The dataset contains 14980 entries with no missing values.
All features are numerical (float64), and the target variable is an integer (int64).
The describe() output showed some unusually high maximum values for certain EEG channels (e.g., AF3, FC5, P7, O1, AF4), which might indicate outliers or noise in the data, but for this initial model, they were not explicitly handled.

Model:
Algorithm:
A Random Forest Classifier was chosen for this classification task. Random Forests are an ensemble learning method for classification and regression that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

Implementation Details:
The data was split into training and testing sets with a 70/30 ratio using train_test_split.
The RandomForestClassifier was initialized with n_estimators=100 (100 decision trees) and random_state=42 for reproducibility.
The model was trained on the x_train and y_train datasets.
Results:
After training and evaluating the Random Forest Classifier on the test set, the following performance metrics were obtained:

Accuracy: 0.9226 (approximately 92.26%)
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.96      0.93      2386
           1       0.95      0.88      0.91      2108

    accuracy                           0.92      4494
   macro avg       0.93      0.92      0.92      4494
weighted avg       0.92      0.92      0.92      4494
Confusion Matrix:
[[2285  101]
 [ 247 1861]]

# EEG Eye State Classification using SVM

Overview:
This repository contains a Jupyter Notebook for classifying eye states (open or closed) based on Electroencephalogram (EEG) signals. The project utilizes Support Vector Machine (SVM) models with both linear and Radial Basis Function (RBF) kernels to predict the eyeDetection target variable.

Dataset:
The dataset used for this project is EEG_Eye_State_Classification.csv. It contains 14 features, which are EEG signal measurements from various electrodes, and a target variable eyeDetection (0 for eye closed, 1 for eye open).

Features:

AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 (EEG sensor readings)
Target:

eyeDetection: 0 (eyes closed) or 1 (eyes open)
Methodology
1. Data Loading and Initial Exploration
The dataset was loaded into a pandas DataFrame.
Initial checks were performed for missing values (isnull().sum()), data types (info()), and descriptive statistics (describe()). No missing values were found.
2. Data Preprocessing
Feature-Target Split: The dataset was split into features (X) and target (y), where eyeDetection is the target variable.
Train-Test Split: The data was divided into training and testing sets with a 80/20 ratio (test_size=0.2) and random_state=1 for reproducibility.
Feature Scaling: StandardScaler was applied to the features to standardize them, which is crucial for SVM models.
3. Model Training and Evaluation
Two Support Vector Machine (SVM) models were trained and evaluated:

a) Linear SVM
A Support Vector Classifier (SVC) with a linear kernel was trained on the scaled training data.

Evaluation Metrics:

Accuracy Score: 0.6175
Classification Report:
              precision    recall  f1-score   support

   0       0.61      0.88      0.72      1661
   1       0.66      0.29      0.41      1335
accuracy 0.62 2996 macro avg 0.63 0.59 0.56 2996

weighted avg 0.63 0.62 0.58 2996 ``` - A confusion matrix was also generated.

b) RBF SVM
A Support Vector Classifier (SVC) with an rbf (Radial Basis Function) kernel was trained on the scaled training data.

Evaluation Metrics:

Accuracy Score: 0.7250
Classification Report:
              precision    recall  f1-score   support

   0       0.70      0.90      0.78      1661
   1       0.80      0.51      0.62      1335
accuracy : 0.72 2996 macro avg 0.75 0.70 0.70 2996

weighted avg : 0.74 0.72 0.71 2996 ``` - A confusion matrix was also generated.

Results and Conclusion:
The RBF kernel SVM model significantly outperformed the linear kernel SVM model, achieving an accuracy of approximately 0.725 compared to 0.6175. This suggests that the relationship between the EEG features and eye state is non-linear, which the RBF kernel is better equipped to handle.
