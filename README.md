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

- Summary statistics
  ``` data.describe() ```
- Data visualization (histograms, scatter plots, etc.)
- Key findings

## Data Preprocessing
- Handling missing values
- Feature scaling using StandardScaler
- Encoding categorical variables (if any)

## Modeling
- Logistic Regression
- Decision Tree
- Random Forest
- Hyperparameter tuning using GridSearchCV
- Evaluation metrics: accuracy, precision, recall, F1 score

## Results
- Model performance comparison
- Confusion matrix
- Best performing model

## Conclusion
- Key takeaways from the analysis
- Potential improvements and future work

## How to Use
1. Run the Jupyter notebook:
    ```bash
    jupyter notebook Breast_Cancer_Analysis.ipynb
    ```
2. Follow the steps in the notebook to reproduce the analysis.
## Contributing
I welcome contributions! Please see `CONTRIBUTING.md` for more details.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Kaggle](https://www.kaggle.com) for providing the dataset and an indepth description about the dataset
- Contributors and maintainers of the libraries used in this project.
