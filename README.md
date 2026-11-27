![](UTA-DataScience-Logo.png)

# Introverts vs. Extroverts Classification

**One-Sentence Summary:** This project uses the XGBoost machine learning algorithm to predict whether an individual is an introvert or extrovert based on behavioral and social activity features from the Kaggle *Predict the Introverts from the Extroverts* competition.
(https://www.kaggle.com/competitions/playground-series-s5e7).

## Overview

The objective of this project is to classify individuals as Introverts or Extroverts using behavioral and social activity information. The dataset includes features such as time spent alone, social event attendance, stage fear, going outside, feeling drained after socializing, friend circle size, and posting frequency.

This project formulates the task as a binary classification problem. The data was explored and preprocessed by handling missing values and encoding categorical variables. An original XGBoost classifier was trained as a baseline, and GridSearchCV was used to tune selected hyperparameters using five-fold cross-validation. The two models were then evaluated using a held-out validation dataset.

The tuned XGBoost model achieved **96.90% validation accuracy**, slightly improving upon the original model's 96.84% accuracy. The final tuned model was also used to generate predictions for the Kaggle competition test dataset.

## Summary of Work Done

### Data

**Type:**
- **Input:** CSV file containing behavioral and social activity features.
- **Output:** Personality label (Introvert or Extrovert).
- **Features:**
  - `Time_spent_Alone`
  - `Stage_fear`
  - `Social_event_attendance`
  - `Going_outside`
  - `Drained_after_socializing`
  - `Friends_circle_size`
  - `Post_frequency`

**Dataset Size:**
- Training dataset: 18,524 records
- Competition test dataset: 6,175 records
- Total records provided: 24,699

**Training, Validation, and Test Split:**
- Original training data: 18,524 instances
- Validation data: 3,705 instances (20% of the training set after splitting)
- Model training data: 14,819 instances (80% of the training set after splitting)
- Competition test data: 6,175 instances

The training dataset was split into training and validation subsets using an 80/20 stratified split. Stratification was used to preserve the original class distribution across both subsets. The competition test dataset was kept separate and used for generating final predictions.

### Preprocessing / Clean Up

The dataset was examined for missing values, data types, duplicate records, categorical values, and numerical feature distributions before model training.

**1. Missing Value Handling**

Missing values were present in several features. Removing all rows containing missing values would have reduced the training dataset from 18,524 rows to 10,189 rows, resulting in the loss of approximately 45% of the original data.

To retain more data, imputation was used:

- Numerical features were filled using the median.
- Categorical features were filled using the most frequent category (mode).

The imputation values were fitted using the training subset and then applied to the training, validation, and competition test datasets.

**2. Categorical Encoding**

The categorical features `Stage_fear` and `Drained_after_socializing` contained Yes/No values. These were converted into numerical values:

- No = 0
- Yes = 1

The target variable `Personality` was encoded using LabelEncoder:

- Extrovert = 0
- Introvert = 1

**3. Feature Selection**

The `id` column was excluded from model training because it serves as an identifier rather than a behavioral predictor.

The model used seven behavioral features, consisting of five numerical features and two categorical features.

**4. Train-Validation Split**

The training dataset was split using:

- Validation size: 20%
- Training size: 80%
- Random state: 2008
- Stratification: Enabled

This resulted in 14,819 training observations and 3,705 validation observations.

**5. Duplicate Records**

The dataset was checked for duplicate records. No duplicate rows were found.

### Data Visualization

Several visualizations were created to explore the dataset and understand patterns in the behavioral features.

**1. Personality Distribution**

A count plot was generated to examine the distribution of the target variable.

| Personality | Count | Percentage |
|---|---:|---:|
| Extrovert | 13,699 | 73.95% |
| Introvert | 4,825 | 26.05% |
<img width="350" height="200" alt="download" src="https://github.com/user-attachments/assets/81e89a37-9672-4acc-af13-15eb8103a948" />

The dataset is imbalanced, with Extroverts representing the larger class. Approximately 74% of the observations are labeled Extrovert, while 26% are labeled Introvert.

**2. Numerical Feature Histograms**

Histograms were generated to examine the distributions of the five numerical features:

- `Time_spent_Alone`
- `Social_event_attendance`
- `Going_outside`
- `Friends_circle_size`
- `Post_frequency`

<img width="400" height="300" alt="download" src="https://github.com/user-attachments/assets/e66f0662-3dc4-496c-bb4d-99d386246e81" />

The observations showed the following patterns:

- **Time_spent_Alone:** Most observations were concentrated between 0 and 3.
- **Social_event_attendance:** Values were spread across the range, with many observations around 4–9.
- **Going_outside:** Many observations fell between 3 and 7.
- **Friends_circle_size:** Values were distributed across a relatively wide range.
- **Post_frequency:** Many observations fell between 3 and 9.

**3. Numerical Feature Comparisons by Personality**

Normalized histograms were used to compare numerical feature distributions between Introverts and Extroverts.

The visualizations suggested the following differences:

- Introverts generally had higher values of `Time_spent_Alone`.
- Extroverts generally had higher values of `Social_event_attendance`.
- Extroverts generally had higher values of `Going_outside`.
- Extroverts generally had larger `Friends_circle_size` values.
- Extroverts generally had higher `Post_frequency` values.

**4. Categorical Feature Comparisons**

Grouped percentage plots and cross-tabulations were generated to compare `Stage_fear` and `Drained_after_socializing` between personality groups.

| Feature | Personality | No | Yes |
|---|---|---:|---:|
| Stage_fear | Extrovert | 98.12% | 1.88% |
| Stage_fear | Introvert | 7.23% | 92.77% |
| Drained_after_socializing | Extrovert | 98.22% | 1.78% |
| Drained_after_socializing | Introvert | 7.54% | 92.46% |

These features showed a strong difference between the two personality groups in the dataset.

**5. Numerical Feature Boxplots**

Boxplots were generated to examine the numerical features and identify potential outliers.

- `Time_spent_Alone` showed some potential higher-value outliers above 8.
- `Social_event_attendance` showed no obvious outliers.
- `Going_outside` showed no obvious outliers.
- `Friends_circle_size` showed no obvious outliers.
- `Post_frequency` showed no obvious outliers.

**6. Correlation Heatmap**

A correlation heatmap was created to examine relationships among the numerical features.

The observed correlations included:

- `Time_spent_Alone` had moderately strong negative correlations with the other numerical features, ranging from approximately −0.60 to −0.64.
- `Social_event_attendance` had positive correlations with other social-activity features, ranging from approximately 0.57 to 0.59.
- `Going_outside` and `Post_frequency` had a positive correlation of approximately 0.58.
- `Friends_circle_size` and `Post_frequency` had a positive correlation of approximately 0.52.

The heatmap suggested that people who spent more time alone tended to have lower values in the other numerical social-activity features in this dataset.

## Problem Formulation

### Input and Output

**Input:** The model uses seven behavioral and social activity features:

- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`

**Output:** The target variable is `Personality`, a binary classification label with two possible classes:

- Extrovert
- Introvert

The objective is to learn patterns from the input features and predict the personality label.

### Models

**XGBoost Classifier**

XGBoost (Extreme Gradient Boosting) was selected as the classification algorithm for this project. It is a tree-based boosting method that builds decision trees sequentially to improve predictions.

Two model configurations were evaluated:

**1. Original XGBoost Model**

The original model was used as the initial benchmark.

Parameters:
- `n_estimators = 100`
- `max_depth = 3`
- `learning_rate = 0.1`
- `objective = 'binary:logistic'`
- `eval_metric = 'logloss'`
- `random_state = 2008`

**2. Tuned XGBoost Model**

GridSearchCV was used to search for a better combination of selected hyperparameters. The tuned model was selected based on five-fold cross-validation accuracy.

The tuned model achieved the highest cross-validation score among the tested combinations and was evaluated again using the held-out validation dataset.

### Loss, Optimizer, and Hyperparameters

**Loss Function**

The model used the binary logistic objective (`binary:logistic`) for binary classification and `logloss` as the evaluation metric.

**Optimization Method**

XGBoost uses gradient boosting, adding trees sequentially to improve the model by minimizing its loss function.

**Hyperparameter Tuning**

GridSearchCV was used to search the following parameter combinations:

| Hyperparameter | Values Tested |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 2, 3, 4 |
| `learning_rate` | 0.05, 0.1, 0.2 |

The grid contained 27 parameter combinations. With five-fold cross-validation, a total of 135 fits were performed.

GridSearchCV settings:

- `scoring = 'accuracy'`
- `cv = 5`
- `n_jobs = -1`
- `verbose = 1`

**Best Hyperparameters**

| Hyperparameter | Selected Value |
|---|---:|
| `learning_rate` | 0.2 |
| `max_depth` | 2 |
| `n_estimators` | 100 |

The best cross-validation accuracy was **96.88%**.

## Training

### How the Model Was Trained

The project was developed using Python in Jupyter Notebook. Pandas and NumPy were used for data manipulation, Matplotlib was used for visualizations, Scikit-learn was used for preprocessing and evaluation, and XGBoost was used for classification.

The training process followed these steps:

1. Load the training and competition test datasets.
2. Examine the data types, missing values, duplicates, and feature distributions.
3. Perform exploratory data analysis and generate visualizations.
4. Separate the input features and target variable.
5. Exclude the `id` column from the model features.
6. Split the training data into training and validation subsets using an 80/20 stratified split.
7. Impute numerical and categorical missing values.
8. Encode the categorical features and target variable.
9. Train the original XGBoost model.
10. Evaluate the original model using validation data.
11. Perform five-fold cross-validation hyperparameter tuning using GridSearchCV.
12. Evaluate the tuned model using the validation subset.
13. Compare the original and tuned model performance.
14. Generate predictions for the competition test dataset.
15. Save the predictions to `submission.csv`.

### Training Time

The exact model training duration was not recorded in the notebook. Therefore, no specific training time is reported.

### Training Curves

Training and validation log-loss curves were generated for the tuned XGBoost model across 100 boosting rounds.

The curves were produced using the training and validation datasets passed to the model through its evaluation sets. The log-loss values were recorded for each boosting round and plotted to observe the loss progression during training.

XGBoost uses boosting rounds rather than neural-network epochs.

### Stopping Criteria

The model was trained for the specified 100 boosting rounds (`n_estimators = 100`). Early stopping was not configured. During hyperparameter tuning, GridSearchCV selected the configuration with the highest mean five-fold cross-validation accuracy.

### Difficulties and Solutions

**1. Missing Data**

Several input features contained missing values. Removing rows with missing values would have resulted in losing approximately 45% of the original training data.

This was addressed by using median imputation for numerical features and most-frequent-value imputation for categorical features.

**2. Class Imbalance**

The target variable contained more Extrovert observations than Introvert observations. Stratified splitting was used to preserve the class distribution in training and validation datasets.

**3. Selecting Hyperparameters**

Choosing the most effective XGBoost settings was a challenge. GridSearchCV was used to systematically evaluate 27 parameter combinations through five-fold cross-validation.

**4. Consistent Preprocessing**

The training, validation, and competition test datasets needed to have the same feature columns and numerical encoding. The same fitted imputers and categorical encoding rules were applied consistently to all three datasets.

## Performance Comparison

### Evaluation Metrics

The models were evaluated using the held-out validation dataset of 3,705 observations.

The following metrics were used:

- **Accuracy:** The proportion of all predictions that were correct.
- **Precision:** The proportion of predicted positive cases that were correct.
- **Recall:** The proportion of actual positive cases that were correctly identified.
- **F1-score:** The harmonic mean of precision and recall.

Weighted precision, recall, and F1-score were reported to account for the class distribution.

### Model Performance Comparison

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Original XGBoost | 96.8421% | 96.8411% | 96.8421% | 96.8416% |
| Tuned XGBoost | 96.8961% | 96.8951% | 96.8961% | 96.8956% |

The tuned model slightly improved the validation performance compared with the original model.

- Original XGBoost accuracy: **96.84%**
- Tuned XGBoost accuracy: **96.90%**
- Best five-fold cross-validation accuracy: **96.88%**

The improvement was modest, but the tuned model achieved the highest validation accuracy among the configurations evaluated in this project.

### Visualization of Results

**1. Confusion Matrix**

The confusion matrix was generated for the tuned XGBoost model using the validation dataset.

Rows represent the actual classes and columns represent predicted classes, in the order Extrovert and Introvert.

| Actual Class | Predicted Extrovert | Predicted Introvert |
|---|---:|---:|
| Extrovert | 2,683 | 57 |
| Introvert | 58 | 907 |

The confusion matrix shows that most validation observations were correctly classified. The model made 115 incorrect predictions out of 3,705 validation observations.

**2. Feature Importance**

Feature importance was generated using the tuned XGBoost model.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Drained_after_socializing` | 0.830210 |
| 2 | `Stage_fear` | 0.069825 |
| 3 | `Time_spent_Alone` | 0.068525 |
| 4 | `Social_event_attendance` | 0.015556 |
| 5 | `Going_outside` | 0.009698 |
| 6 | `Post_frequency` | 0.005014 |
| 7 | `Friends_circle_size` | 0.001172 |

`Drained_after_socializing` had the largest feature importance in the tuned model, followed by `Stage_fear` and `Time_spent_Alone`.

Feature importance represents the model's relative use of features for making predictions. It does not establish causation or prove that a feature independently determines personality.

### Discussion

Both XGBoost models achieved high validation accuracy. Hyperparameter tuning resulted in a small improvement in accuracy, precision, recall, and weighted F1-score.

The validation results showed that the model classified both personality groups well, although the dataset contains more Extrovert than Introvert observations. Feature importance indicated that `Drained_after_socializing` contributed the most to the tuned model, while `Stage_fear` and `Time_spent_Alone` were the next most influential features.

These results describe performance on this dataset and the selected validation split. They should not be interpreted as a definitive or universal measurement of a person's personality.

## Conclusions

This project developed a binary classification model using XGBoost to predict Introvert and Extrovert labels from behavioral and social activity features.

The original XGBoost model achieved 96.84% validation accuracy. Hyperparameter tuning using GridSearchCV improved the accuracy to 96.90%, a modest increase.

The analysis showed that the target classes were imbalanced, with Extroverts making up approximately 73.95% of the training dataset. Missing values were handled through median and most-frequent-value imputation, allowing the project to retain the available observations rather than remove a large portion of the dataset.

Feature importance analysis identified `Drained_after_socializing` as the most influential feature in the tuned model, followed by `Stage_fear` and `Time_spent_Alone`.

Overall, the project demonstrated how data cleaning, exploratory data analysis, model training, cross-validation, hyperparameter tuning, and model evaluation can be combined into a machine learning workflow for a structured classification problem.

## Future Work

Several opportunities exist to extend and improve this project.

- **Evaluate additional models:** Compare XGBoost with Logistic Regression, Random Forest, Support Vector Machines, and other classification algorithms.
- **Explore other tuning approaches:** Use RandomizedSearchCV or other search methods to explore additional hyperparameter combinations.
- **Investigate class imbalance:** Examine whether class weighting or other imbalance-handling methods improve performance, especially for the Introvert class.
- **Feature engineering:** Explore additional features or combinations of behavioral variables to identify whether they improve model performance.
- **Further model analysis:** Examine class-specific performance, prediction errors, and how model performance changes across different validation samples.
- **Deployment:** Develop a simple application that accepts feature inputs and returns model predictions, while making clear that these predictions are based on the dataset and are not definitive personality assessments.

## How to Reproduce Results

### Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### Dataset

Download the datasets from the Kaggle competition:

**Predict the Introverts from the Extroverts**

https://www.kaggle.com/competitions/playground-series-s5e7

Download the following files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Place the files in the same directory as the Jupyter Notebook.

### Running the Project

1. Clone or download the project repository.
2. Navigate to the project directory.
3. Install the required packages.
4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `Introverts_Extroverts.ipynb`.
6. Run all notebook cells from top to bottom.

The notebook will load the data, perform exploratory analysis, preprocess the features, train the original model, tune hyperparameters, evaluate model performance, and generate predictions for the competition test dataset.

### Reproducing Training Results

To reproduce the reported results:

1. Download the Kaggle dataset files.
2. Install the required libraries.
3. Open the Jupyter Notebook.
4. Run all cells in order without modifying the code.
5. Review the validation metrics, feature importance, confusion matrix, and training curves.

The tuned XGBoost model achieved approximately **96.90% validation accuracy** using the 80/20 stratified split with `random_state=2008`.

### Generating Predictions

The notebook applies the fitted preprocessing steps to the competition test dataset and generates predictions using the tuned XGBoost model.

The final output is saved as `submission.csv`, containing:

- `id`
- `Personality`

The generated submission file contains 6,175 predictions and no missing values. It follows the format specified by `sample_submission.csv`.

## Overview of Files in Repository

The project repository contains the notebook, datasets, prediction file, and documentation.

### Directory Structure

```text
Introverts_Extroverts/
│
├── README.md
├── Introverts_Extroverts.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
└── submission.csv
```

### File Descriptions

**README.md**  
Contains the project overview, dataset description, preprocessing methodology, visualizations and observations, model training, evaluation results, conclusions, future work, and instructions for reproducing the project.

**Introverts_Extroverts.ipynb**  
The main Jupyter Notebook containing the complete machine learning workflow, including data inspection, exploratory data analysis, preprocessing, XGBoost training, hyperparameter tuning, model evaluation, and submission generation.

**train.csv**  
The labeled training dataset containing the behavioral input features and the `Personality` target variable.

**test.csv**  
The competition test dataset containing the input features without the target labels. It is used to generate predictions.

**sample_submission.csv**  
The sample Kaggle submission file showing the required submission columns and format.

**submission.csv**  
The prediction file generated by the tuned XGBoost model, containing the IDs and predicted personality labels for the competition test dataset.

## Software Setup

### Required Packages

The project was developed using Python and the following packages:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

### Installing Required Packages

Install the packages using pip:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### System Environment

The project was developed using:

| Software / Environment | Version |
|---|---|
| Python | 3.14.4 |
| Operating System | Linux on WSL2 |
| CPU cores available | 16 |
| Pandas | 3.1.1 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |
| Scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |

The notebook was run locally using Jupyter Notebook within the Windows Subsystem for Linux 2 (WSL2) environment.

### Running the Project

1. Clone or download the repository.
2. Navigate to the project folder:

```bash
cd Introverts_Extroverts
```

3. Install the required packages.
4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `Introverts_Extroverts.ipynb`.
6. Run all cells in order.

## Data

### Data Source

The dataset was obtained from the Kaggle competition:

**Predict the Introverts from the Extroverts**

https://www.kaggle.com/competitions/playground-series-s5e7

The original dataset source is also available here:

https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data

### Dataset Description

The dataset contains behavioral and social activity information used to predict the target personality label.

The seven input features are:

- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`

The target variable is:

- `Personality`

The competition test dataset does not contain the target label. The tuned model was used to generate predictions for this dataset.

### Preprocessing Summary

- Numerical missing values were filled using the median.
- Categorical missing values were filled using the most frequent category.
- Yes/No categorical features were encoded as 0/1.
- The target variable was label encoded.
- The `id` feature was excluded from model training.
- The training dataset was split into training and validation subsets using an 80/20 stratified split.
- The same fitted imputation and encoding rules were applied to the validation and competition test datasets.

## Training

To train the model, ensure that the required packages are installed and the dataset files are available in the project directory.

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open the notebook:

```text
Introverts_Extroverts.ipynb
```

3. Run all cells in order from top to bottom.

The notebook will:

- Load the training and test datasets.
- Examine missing values and data distributions.
- Generate exploratory data visualizations.
- Preprocess the numerical and categorical features.
- Split the labeled dataset into training and validation subsets.
- Train the original XGBoost model.
- Perform five-fold cross-validation hyperparameter tuning.
- Evaluate both models using validation data.
- Display the confusion matrix and feature importance.
- Plot training and validation log-loss across boosting rounds.
- Generate predictions for the competition test dataset.
- Save the predictions to `submission.csv`.

The final tuned XGBoost model achieved approximately **96.90% validation accuracy**.

### Performance Evaluation

Model evaluation is performed using the held-out validation dataset.

The notebook reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Feature importance
- Training and validation log-loss curves

To reproduce the evaluation:

1. Open `Introverts_Extroverts.ipynb`.
2. Run the notebook cells in order.
3. Review the original and tuned XGBoost evaluation results.
4. Compare the performance metrics.
5. Review the confusion matrix, feature importance, and log-loss curves.

The final tuned model achieved 96.90% validation accuracy, compared with 96.84% for the original model.

## Citations

### Dataset Sources

1. Kaggle. **Predict the Introverts from the Extroverts.**  
   https://www.kaggle.com/competitions/playground-series-s5e7

2. Kaggle. **Extrovert vs. Introvert Behavior Data.**  
   https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data

### Software and Libraries

- Pandas — data manipulation and analysis.
- NumPy — numerical operations.
- Matplotlib — data visualization.
- Scikit-learn — preprocessing, train-validation splitting, hyperparameter tuning, and evaluation.
- XGBoost — gradient boosting classification.





