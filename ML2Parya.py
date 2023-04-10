#import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate ,GridSearchCV
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor
from scipy.stats import ttest_rel
from sklearn.linear_model import Ridge

# Load the dataset
dataset = datasets.fetch_openml(data_id=43250, as_frame=True, parser="auto")

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
data = imputer.fit_transform(dataset.data)
data = pd.DataFrame(data, columns=dataset.feature_names)

# Handle nominal features
nominal_features = ["quarter", "department", "day", "date"]
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), nominal_features)], remainder="passthrough")
new = ct.fit_transform(data)
ct.get_feature_names_out()

# Create a new Dataset with the transformed data
feature_names = ct.get_feature_names_out()
data = pd.DataFrame(new, columns=feature_names)
x, y = new, dataset.target

# Tune decision Tree and K nearest neighbor
dtparameters = [{"min_samples_leaf":[1,10,50,200,500]}]
tuned_DecisionTree= GridSearchCV(DecisionTreeRegressor(), dtparameters, scoring="neg_root_mean_squared_error", cv=5)
knnparameters = [{"n_neighbors":[1,10,50,200,500]}]
tuned_KNeighbors= GridSearchCV(KNeighborsRegressor(), knnparameters, scoring="neg_root_mean_squared_error", cv=5)
lr_parameters = [{"alpha": [0, 0.1, 1, 20]}]
tuned_LinearRegression = GridSearchCV(Ridge(), lr_parameters, scoring="neg_root_mean_squared_error", cv=5)

# Create an empty dictionary to store the RMSE scores for each model
rmse_scores = {}

# Train all models on the entire dataset
models = {
    "Linear Regression": tuned_LinearRegression.fit(x,y),
    "Decision Tree": tuned_DecisionTree.fit(x,y),
    "K-Nearest Neighbors": tuned_KNeighbors.fit(x,y),
    "Support Vector Regression": SVR(),
    "Bagged Linear Regression": BaggingRegressor(estimator=tuned_LinearRegression.fit(x,y)),
    "Bagged Decision Tree": BaggingRegressor(estimator=tuned_DecisionTree.fit(x,y)),
    "Bagged K-Nearest Neighbors": BaggingRegressor(estimator=tuned_KNeighbors.fit(x,y)),
    "Bagged Support Vector Regression": BaggingRegressor(estimator=SVR()),
    "Boosted Linear Regression": AdaBoostRegressor(estimator=tuned_LinearRegression.fit(x,y)),
    "Boosted Decision Tree": AdaBoostRegressor(estimator=tuned_DecisionTree.fit(x,y)),
    "Boosted K-Nearest Neighbors": AdaBoostRegressor(estimator=tuned_KNeighbors.fit(x,y)),
    "Boosted Support Vector Regression": AdaBoostRegressor(estimator=SVR()),
    "Voting Regression": VotingRegressor([
        ("Linear Regression", tuned_LinearRegression.fit(x,y)),
        ("Decision Tree", tuned_DecisionTree.fit(x,y)),
        ("K-Nearest Neighbors", tuned_KNeighbors.fit(x,y)),
        ("Support Vector Regression", SVR())
    ])
}

# Perform cross-validation for each model and compute RMSE scores
for name, model in models.items():
    scores = cross_validate(model, new, dataset.target, cv=10, scoring="neg_root_mean_squared_error", return_train_score=False)
    rmse = 0 - scores["test_score"].mean()
    rmse_scores[name] = scores["test_score"]
    print("Mean RMSE score for", name, ":", rmse)

# Define the pairs of models to compare
pairs = [("Linear Regression", "Bagged Linear Regression"),
         ("Linear Regression", "Boosted Linear Regression"),
         ("Decision Tree", "Bagged Decision Tree"),
         ("Decision Tree", "Boosted Decision Tree"),
         ("K-Nearest Neighbors", "Bagged K-Nearest Neighbors"),
         ("K-Nearest Neighbors", "Boosted K-Nearest Neighbors"),
         ("Support Vector Regression", "Bagged Support Vector Regression"),
         ("Support Vector Regression", "Boosted Support Vector Regression")]

# Perform t-tests between pairs of models
for name1, name2 in pairs:
    t_stat, p_value = ttest_rel(rmse_scores[name1], rmse_scores[name2])
    print("t-test between", name1, "and", name2, "has p-value of", p_value)

