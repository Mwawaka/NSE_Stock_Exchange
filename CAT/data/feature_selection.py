# feature selection
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=2)
selector.fit(X_train, y_train)
print('Number of input features:', selector.n_features_in_)
print('Input features Names  :', selector.feature_names_in_)
print('Input features scores :', selector.scores_)
print('Input features pvalues:', selector.pvalues_)
print('Output features Names :', selector.get_feature_names_out())

# Based on the output ['Day Low','Day High'] have a huge impact on the classes

# selecting features based on f-score
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)