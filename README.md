# Breast Cancer Wisconsin (Diagnostic) Data Analysis

## Introduction
This project involves an analysis of the Breast Cancer Wisconsin (Diagnostic) Data Set to predict whether the cancer is benign or malignant using various machine learning techniques.

## Data Description
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features**: 30 numeric features describing the characteristics of the cell nuclei in images.
- **Target**: Binary classification - 0 (benign) or 1 (malignant).
## About The Dataset
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

## Setup and Installation
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Vinyo1/Machine_Learning_Breast_Cancer_Detection.git
    cd Machine_Learning_Breast_Cancer_Detection
    ```
2. Follow the steps in the notebook to reproduce the analysis.
## Contributing
I welcome contributions! Please see `CONTRIBUTING.md` for more details.

## NB//: The full code for this project is in the attached notebook
## STEP 1 - Data Preprocessing and Dealing With Missing Values
```python
        print(data.head())
        print(data.shape)
        print(data.describe())
        print(data.info())
        data.isnull().values.any()
        # drop column with null values
        data = data.drop(columns = 'Unnamed: 32')
```
### 1a. Using One Hot Encoding To Deal With Categorical Data
```python
data = pd.get_dummies(data=data, drop_first=True)
data.head()
```
### 1b Developing A Corelation Matrix And Plotting A Heat Map
```python
        data_2.corrwith(data['diagnosis_M']).plot.bar(
        figsize=(20, 10), title = 'Correlated with diagnosis_M', rot=90, grid=True
    )
```
### 1c Splitting Data Into Train and Test
```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)
```
### 1d - Feature Scaling
```python
    # create an instance of the class
    scaler = StandardScaler()
    # Fitting X_train and X_test to the scaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
```
## STEP 2 - MODEL BUILDING - Logistic Regression
```python
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=42)
        # Fit training data
        lr.fit(X_train, y_train)
```
### 2a -Cross Validation
```python
        accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
        print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
        print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
```
### 2b Model Selection -Random Forest
```python
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    # creating an instance
    rf.fit(X_train, y_train)
```
### 2c -Cross Validation
```python
        #Analyze perfomance of Random Forest model
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
```python
        ### Comparing both models
        compare_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
        compare_models.style.background_gradient(cmap='Blues')
```
### 2d Verifying True Confusion Matrix
```python
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat
```
## STEP 3 - Hyper Parameter Tuning Using Randomized SearchCV and GridSearchCV
``` python
    accuracies = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10)
    print(accuracies)
```
### 3a - Randomizing Search To Find The Best Parameters For Logistic Regression
```python
    from sklearn.model_selection import RandomizedSearchCV
    parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.25, 0.50, 0.75, 1, 1.25, 1.50, 2.0],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    # defining an instance
    random_search = RandomizedSearchCV(estimator=lr, param_distributions=parameters, n_iter=10, scoring='roc_auc', n_jobs= -1, verbose= 3, cv=10, random_state=42)
    random_search.fit(X_train, y_train)
    #Finding the best score
    random_search.best_score_
    #Finding the best parameters
    random_search.best_params_
```
### 3b - GridSearch Cv to Find The Best Parameters For Logistic Regression
    from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
```python
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.25, 0.50, 0.75, 1, 1.25, 1.50, 2.0],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 1000, 2500, 5000]
    }
    
    
    # Set up the grid search
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='roc_auc',
                               n_jobs=-1, verbose=3, cv=10)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print("Best estimator:", grid_search.best_estimator_)
```
## Step 4 - Part 4 Deciding on The Model - Logistic Regression
```python
    # Selecting the best parameters for the Logistic Regression-(gRIDsEARCHCV Parameters)
    from sklearn.linear_model import LogisticRegression
    final_lr = LogisticRegression(C=0.25, penalty='l2',  max_iter =  100, solver='liblinear')
    final_lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    #Analyze perfomance of logistic model
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    final_regression_results = pd.DataFrame([['Final Logistic Regression', acc, f1, recall, precision, roc_auc]], columns = ['Model', 'Accuracy',
    'F1 Score', 'Recall', 'Precision', 'ROC_AUC'])
```
## Step 5 - Comparing The Report of All Three Models
```python
    # Comparing Models
    final_results = pd.concat([lr_results, rf_results, final_regression_results], axis=0).reset_index(drop=True)
    final_results.style.background_gradient(cmap='Blues')
```
### 5a - Cross Validation 
```python
    accuracies = cross_val_score(estimator = final_lr, X = X_train, y = y_train, cv = 10)
    print(accuracies)
    print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
    print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
```

- Summary statistics
  ``` data.describe() ```


## Findings and Conclusion
Best performing model was the Logistic regression model it had an **accuracy of 97.36 %** and a **Standard Deviation of 1.93 %**
The random forest model had an **accuracy of 96.26 % and a Standard Deviation of 2.41 %**
After hyper parameter tuning, the best parameter was 
```python
    # Selecting the best parameters for the Logistic Regression-(gRIDsEARCHCV Parameters)
    from sklearn.linear_model import LogisticRegression
    final_lr = LogisticRegression(C=0.25, penalty='l2',  max_iter =  100, solver='liblinear')
```
This gave an **accuracy of 98.02 %** and a **Standard Deviation of 1.54 %**
|S/N |Model |Accuracy |F1 Score	|Recall	|Precision	|ROC_AUC
|----|----|----|----|----|----|----|
|0	|Logistic Regression	|0.973684	|0.964706	|0.953488	|0.976190	|0.969702
|1	|Random Forest	|0.964912	|0.952381	|0.930233	|0.975610	|0.958074
|2	|Final Logistic Regression	|0.991228	|0.988235	|0.976744	|1.000000	|0.988372

#### Key takeaways from the analysis
- Apart from fractal_dimension_mean, texture_se, smoothness_se, symmetry_se, all other features are postively corelated with the target variable.

#### Potential improvements and future work
- Consider dimensionality reduction techniques like PCA (Principal Component Analysis) to reduce the number of features while retaining most of the variance.
- Perform more exhaustive hyperparameter tuning using techniques like Grid Search or Random Search with cross-validation to find the optimal model parameters
- Combine multiple models to create an ensemble (e.g., using techniques like bagging, boosting, or stacking) which can often result in better performance than individual models.
- Implement anomaly detection algorithms to identify and handle outliers more effectively, which can improve model performance.
- Regularly audit the model for fairness and ethical considerations.



## License
This project is licensed under the MIT License. See the [Lincense](url) file for more details.


## Acknowledgments
- [Kaggle](https://www.kaggle.com) for providing the dataset and an indepth description about the dataset

