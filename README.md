# Intro
This page (https://irismaxj.github.io/DS_FoodFactAnalysis/) is a showcase of the big-data analytic project -- Food Facts Analysis. 
* Data Source: [Open Food Facts](https://github.com/openfoodfacts), [Ooen Food Facts Official Website](https://world.openfoodfacts.org/discover)
* Note: the project was done in the `.ipynb` notebook environment.

## Data Science Techniques used:
* Data Cleaning
* Exploratory Data Analysis (EDA)
* Data Visualization: wordcloud, heatmap, bar plot, scatter plot, confusion matrix
* Machine Learning:
  * Model Selection:
    * **Classification** (Logistic Regression)
    * **Linear Regression**
    * **Random Forrest**
  * Dimensionality Reduction via Principal Component Analysis (PCA)
  * Resampling
  * Feature Engineering
  * Model training
  * Hyperparameter Tuning

## Preview of the Data Visualization

Correlation Matrix of nutrition score (uk standard) vs nutrition score (french standard)
![png](/image-DataVis/22.png)

Nutrition Grade Distribution Range
![png](/image-DataVis/25.png)

Range of distribution of data points of core nutrients (before outlier removal)
![png](/image-DataVis/39.png)

Range of distribution of data points of core nutrients (after outlier removal)
![png](/image-DataVis/44.png)

Word Cloud of common allergens
![png](/image-DataVis/55.png)

Top 50 Allergens
![png](/image-DataVis/59.png)

Correlation Matrix of Core Nutrients that take numeric measures
![png](/image-DataVis/73.png)

Number of Miussing vs Non-missing Values for Each Nutrients (before imputation)
![png](/image-DataVis/84.png)

Number of Miussing vs Non-missing Values for Each Nutrients (after imputation)
![png](/image-DataVis/88.png)

Cumulated Explained Variance Ratio vs Number of Components
![png](/image-DataVis/101.png)

### Classification Model (Logistic Regression)
`LogisticRegressor` Trained with unbalanced samples vs `LogisticRegressor` Trained with Balanced Samples
![png](/image-DataVis/118.png)

**Untuned** `LogisticRegressor` Trained with unbalanced samples vs **Optimized** `LogisticRegressor` Trained with Balanced Samples
![png](/image-DataVis/128.png)

`RandomForestClassifer()` trained with Balanced samples vs **Optimized** `LogisticRegressor()` trained with Balanced Samples
![png](/image-DataVis/135.png)

### Linear Regression Model
**OLS** Linear Regression, untuned
![png](/image-DataVis/152.png)

**Optimized + L1 Regularized** Linear Regression vs. **Untuned + OLS** Linear Regression
![png](/image-DataVis/159.png)

`RandomForestRegressor()` vs **Optimized + L1 Regularized** Linear Regression
![png](/image-DataVis/164.png)

# Insights gained from the Analysis
## For Basis Analysis of the Open Food Fact Dataset:
* We made discovery about the common allergens among the given dataset.
* We revealed the brands that produce high-sugar content foods, as well as brands that make energy-dense foods.
* We spotted the trend that Foods sold exclusively in France are a bit healthier than foods which exclusively sold in the US.
* We revealed the correlations amoung the various nutrients.

## For Classification Models:
* `RandomForestClassifier()` outperformed all constructs of `Logistic Regression` Models.
"Random forests are one of the most popular and accurate method classifiers for big data"
* Performance Overview:
  * Untuned LogisticReg Trained with Unbalanced Samples: ROC_AUC = 0.7657(Accuracy = 62.88%)
  * Untuned LogisticReg Trained with Balanced Samples: ROC_AUC = 0.7730 (Accuracy = 62.35%)
  * Optimized LogisticReg Trained with Balanced Samples: ROC_AUC = 0.7742 (Accuracy 62.41%)
  * C = 0.01, solver = lbfgs, regularization mode = L2 (Ridge Regularization)
RandomForestClassifer() Trained with Balanced Samples: ROC_AUC = 0.9111 (Accuracy 85.60%)

* Reflection: We found that the balanced training set together with Lasso (L2) Regularization produced the best LogisticRegression model, even though the ROC_AUC scores among the three Logistic Regression models were very close. We attribute such improvement to:

1) Ridge's ability to shrink the coefficient(s)/weight(s) or non-important feature(s) towards zero. Greatly suppressed the effect of trivial features.
2) A balanced traning set can help teach the model to learn making categorical predictions in a more unbias way (i.e. the model doesn't favor the majority class just because it contains more data.)
For Regression Models:

## For Linear Regression Models:
* `RandomForestRegressor()` outperformed all constructs of Linear Regression Models.
* Performce Overview:
  * Linear Regression (OLS): R2 Score = 0.6546 
  * Linear Regression (L1): R2 Score = 0.6569
    * alpha= 0.1, l1_ratio= 1.0 (Lasso Regularization)
  * RandomForestRegressor(): R2 Score 0.9505
* Reflection: We found that the balanced training set together with Lasso (L1) Regularization produced the best LinearRegression model, even though the R2 scores among the two LinearRegression models were very close. We attribute such improvement to:
  * Lasso's ability to push the coefficient(s)/weight(s) or non-important feature(s) to zero (i.e. feature selection).
