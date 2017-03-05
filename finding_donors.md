Original hosted at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). 

- 45 k instances
- 14 attributes (including one binary lable feature)

workflow:

1. feature scaling for ['capital-gain', 'capital-loss'].`df.apply(lambda x: np.log(x + 1))`

2. feature normalize for ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'] `sklearn.preprocessing.MinMaxScaler.fit_transform(df)`

3. one hot encode: `pd.get_dummies (df)`

4. `sklearn.model_selection.train_test_split()`

5. fit, predict, `sklearn.metrics.accuracy_score(y_test,y_predict)`

6. `sklearn.grid_search.GridSearchCV`

7. feature importance. It's interesting that different models have different preferences.

   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
   model = DecisionTreeClassifier()
   #model = AdaBoostClassifier()
   #model = RandomForestClassifier()
   model.fit(X_train, y_train)
   importances = model.feature_importances_
   indices = np.argsort(importances)[::-1]
   columns = X_train.columns.values[indices[:5]]
   values = importances[indices][:5]
   print columns
   print values
   ```

8. feature selection. slice data by top 5 important features

9. clone previous clf and train on new features. `clf = sklearn.base.clone(best_clf)`