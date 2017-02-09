Boston housing dataset is original hosted at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing), and can be easily import by `sklearn.datasets.load_boston()`. It has:

- 506 instances
- 14 attributes (including 1 lable feature)

Udacity has a [modified version](https://github.com/udacity/machine-learning/tree/master/projects/boston_housing), which is stored in a cvs file. It has:

- 489 instances
- 4 attributes (including 1 lable feature)
- RM: average number of rooms per dwelling 
- LSTAT: % lower status of the population 
- PTRATIO: pupil-teacher ratio by town 
- MEDV: Median value of owner-occupied homes in $1000's

Note: the median housing price is 439 k, which has been scaled to account for 35 years of market inflation. ( A median value of 21 k in 1978) 

## workflow in Machine Learning Nanodegree project

1. `data = pd.read_csv()` and calculate statistics
2. `data.corr()` to see feature correlations
3. `sklearn.metrics.r2_score()` # coefficient of determination
4. `sklearn.tree.DecisionTreeRegressor`
5. `sklearn.model_selection.learning_curves`
6. `sklearn.model_selection.validation_curve`
7. Bias-Variance Tradeoff in terms of max_depth
8. `sklearn.model_selection.GridSearchCV`