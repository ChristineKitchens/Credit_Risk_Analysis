# Credit Risk Analysis

## Overview
The Python scikit-learn and imbalanced-learn machine learning libraries were used to assess credit card risk based on features such as loan amount, interest, etc. The target for predicted outcome was 'loan_risk', which could be either 'high-risk' or 'low-risk'. Data was analyzed using six different supervised machine learning models:

- [Resampling Models](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#logistic-regression-model-with-resampling)
  - [Naive Random Oversampling](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#naive-random-oversampling)
  - [SMOTE Oversampling](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#smote-oversampling)
  - [Cluster Centroid Undersampling](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#cluster-centroid-undersampling)
  - [SMOTEENN Combination Sampling](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#smoteenn-combination-sampling)
- [Ensemble Models](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#ensemble-models)
  - [Balanced Random Forest Classification](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#balanced-random-forest-classification)
  - [Easy Ensemble with AdaBoosting](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#easy-ensemble-with-adaboost)

The results of each model were assessed based on metrics that included balanced accuracy, precision, and recall scores. The subsequent [Analysis](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#analysis) section is divided up by candidate model. Each section contains tables and screenshots of model assessment metrics (i.e. balanced accuracy scores, precision scores, recall scores, confusion matrices, classification reports). The [Summary](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#summary) section summarizes these results and the resulting model recommendation. Jupyter Notebooks and data can be found in the [Resources](https://github.com/InRegards2Pluto/Credit_Risk_Analysis#resources) section.

## Analysis
### Logistic Regression Model with Resampling
#### Naive Random Oversampling 

- The model had high precision in regards to low-risk outcomes, but low precision in regards to high-risk outcomes.
- The recall scores for both the high-risk and low-risk outcomes were moderately high.
- Overall, the model had a balanced accuracy score of <b>0.6620175698580149</b>

<figcaption><b>Table 1. Naive Random Oversampling Precision and Recall Scores for Target Outcomes</b></figcaption>

| Outcome    | Precision Score | Recall Score  |
|:----------:|:---------------:|:-------------:|
| high-risk  | 0.01            | 0.72          |
| low-risk   | 1.00            | 0.60          |

<figcaption><b>Fig 1. Naive Random Oversampling Model Assessment</b></figcaption>
  
![Results of Logistic Regression and Naive Random Oversampling](images/results_oversampling_naive.png)
#### SMOTE Oversampling
- The model had high precision in regards to low-risk outcomes, but low precision in regards to high-risk outcomes.
- The recall scores for both the high-risk and low-risk outcomes were moderately high. The high-risk recall score dropped relative to the naive random oversampling method, but the low-risk score increased.
- Overall, the model had a balanced accuracy score of <b>0.6568196079430206</b>

<figcaption><b>Table 2. SMOTE Oversampling Precision and Recall Scores for Target Outcomes</b></figcaption>

| Outcome    | Precision Score | Recall Score  |
|:----------:|:---------------:|:-------------:|
| high-risk  | 0.01            | 0.61          |
| low-risk   | 1.00            | 0.70          |

<figcaption><b>Fig 2. SMOTE Oversampling Model Assessment</b></figcaption>

![Results of Logistic Regression and SMOTE Oversampling](images/results_oversampling_smote.png)
#### Cluster Centroid Undersampling
- The model had high precision in regards to low-risk outcomes, but low precision in regards to high-risk outcomes.
- The recall scores for both the high-risk and low-risk outcomes were reduced relative to the prior undersampling methods.
- Overall, the model had a balanced accuracy score of <b>0.6027679241263696</b>

<figcaption><b>Table 3. Cluster Centroid Undersampling Precision and Recall Scores for Target Outcomes</b></figcaption>

| Outcome    | Precision Score | Recall Score  |
|:----------:|:---------------:|:-------------:|
| high-risk  | 0.01            | 0.61          |
| low-risk   | 1.00            | 0.59          |

<figcaption><b>Fig 3. Cluster Centroid Undersampling Model Assessment</b></figcaption>

![Results of Logistic Regression and Cluster Centroid Undersampling](images/results_undersampling_cluster.png)
#### SMOTEENN Combination Sampling
- The model had high precision in regards to low-risk outcomes, but low precision in regards to high-risk outcomes.
- The recall scores for both the high-risk and low-risk outcomes were improved relative to the undersampling method, but not relative to the oversampling methods.
- Overall, the model had a balanced accuracy score of <b>0.639214728301642</b>

<figcaption><b>Table 4. SMOTEENN Combination Sampling Precision and Recall Scores for Target Outcomes</b></figcaption>

| Outcome    | Precision Score | Recall Score  |
|:----------:|:---------------:|:-------------:|
| high-risk  | 0.03            | 0.69          |
| low-risk   | 1.00            | 0.59          |

<figcaption><b>Fig 4. SMOTEENN Combination Sampling Model Assessment</b></figcaption>

![Results of Logistic Regression and SMOTEENN Combination Resampling](images/results_combosampling_smoteenn.png)
### Ensemble Models
#### Balanced Random Forest Classification
- The model had high precision in regards to low-risk outcomes, but low precision in regards to high-risk outcomes.
- The recall scores for both the high-risk and low-risk outcomes were improved relative to the resampling methods.
- Overall, the model had a balanced accuracy score of <b>0.7887512850910909</b>

<figcaption><b>Table 5. Random Forest Model Precision and Recall Scores for Target Outcomes</b></figcaption>

| Outcome    | Precision Score | Recall Score  |
|:----------:|:---------------:|:-------------:|
| high-risk  | 0.03            | 0.70          |
| low-risk   | 1.00            | 0.87          |

<figcaption><b>Fig 5. Random Forest Model Assessment</b></figcaption>

![Results of Random Forest Classification](images/results_rf.png)

<figcaption><b>Table 6. Top 10 Importance Features According to RF Model</b></figcaption>

| Feature          | Importance           |
|:----------------:|:--------------------:|
| total_rec_prncp  | 0.07876809003486353  |
| total_pymnt      | 0.05883806887524815  |
| total_pymnt_inv  | 0.05625613759225244  |
| total_rec_int    | 0.05355513093134745  |
| last_pymnt_amnt  | 0.0500331813446525   |
| int_rate         | 0.02966959508700077  |
| issue_d_Jan-2019 | 0.021129125328012987 |
| installment      | 0.01980242888931366  |
| dti              | 0.01747062730041245  |
| out_prncp_inv    | 0.016858293184471483 |



#### Easy Ensemble with AdaBoost 
- The model had high precision in regards to low-risk outcomes, but low precision in regards to high-risk outcomes.
- The recall scores for both the high-risk and low-risk outcomes were improved relative to all other models.
- Overall, the model had a balanced accuracy score of <b>0.931601605553446</b>

<figcaption><b>Table 7. AdaBoost Model Precision and Recall Scores for Target Outcomes</b></figcaption>

| Outcome    | Precision Score | Recall Score  |
|:----------:|:---------------:|:-------------:|
| high-risk  | 0.09            | 0.92          |
| low-risk   | 1.00            | 0.94          |

<figcaption><b>Fig 6. AdaBoost Model Assessment</b></figcaption>

![Results of Easy Ensemble Classification with AdaBoosting](images/results_ada_boost.png)
## Summary
- Of the 4 resampling machine learning models, it's challenging to identify one model that clearly outperformed the others. Some precision scores were comparable across the board, but more variance was seen in the recall scores. Some had high overall recall scores compared to others, but then their individual high-risk and low-risk scores would be below other candidate models. Looking at the imbalanced accuracy scores (a measure of overall precision/recall score tradeoffs), the naive random oversampling model performed the best of it's cohort.
- Of the 2 ensemble models, the AdaBoosted easy ensemble model outperformed the random forest model across all metrics. However, both ensemble models outperformed the resampling models.
- Across all 6 machine learning models, the AdaBoosted easy ensemble by and far performed best and should be the one selected for use. However, it's important to note that, while the AdaBoosted model outperformed other candidate models across all metrics, the precision score for high-risk loan candidates was still only 0.09. If the primary purpose of the modelling effort was to identify high-risk applicants, alternative machine learning models should be explored.

## Resources
- Data
  - [LoanStats_2019Q1.csv](LoanStats_2019Q1.csv)
- Notebooks
  - [credit_risk_resampling.ipynb](credit_risk_resampling.ipynb)
  - [credit_risk_ensemble.ipynb](credit_risk_ensemble.ipynb)
- Software
  - Jupyter Notebook
