#import libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
import sys
import warnings

warnings.filterwarnings("ignore")

seed = 42
model = "xgboost"
topk = int(sys.argv[1])

#import train and test data
Train_data = pd.read_csv(os.getcwd() + "/../Dataprep/train.csv", sep="\t")
Test_data = pd.read_csv(os.getcwd() + "/../Dataprep/test.csv", sep="\t")

Train_data.drop("Unnamed: 0", axis=1, inplace=True)
Test_data.drop("Unnamed: 0", axis=1, inplace=True)

Train_data["MP_Ratio"] = np.round(np.log(Train_data["MP_Ratio"] + 1), 4)
Test_data["MP_Ratio"] = np.round(np.log(Test_data["MP_Ratio"] + 1), 4)

print("Null in Train", Train_data.isnull().values.any())
print("Null in Test", Test_data.isnull().values.any())

print(Train_data["MP_Ratio"].describe())
print(Test_data["MP_Ratio"].describe())

y_train_reg = Train_data["MP_Ratio"]
y_test_reg = Test_data["MP_Ratio"]

print("Train_data: ", Train_data.shape)
print("Test_data: ", Test_data.shape)

Train_data.drop("MP_Ratio", axis=1, inplace=True)
Test_data.drop("MP_Ratio", axis=1, inplace=True)

# Declare path to export the results and features
path = os.getcwd() + "/Results"
resultpath = path + "/" + model + "/Top" + str(topk)
plotpath = resultpath + "/Plots"
others = resultpath + "/others"
featurepath = resultpath + "/featurepath"

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(resultpath):
    os.makedirs(resultpath)

if not os.path.exists(plotpath):
    os.makedirs(plotpath)

if not os.path.exists(others):
    os.makedirs(others)

if not os.path.exists(featurepath):
    os.makedirs(featurepath)

Non_zero_var_col = []
Zero_var_col = []
for i in list(Train_data.columns):
    if Train_data[i].var():
        Non_zero_var_col.append(i)
    else:
        Zero_var_col.append(i)

#Remove Zero Variance columns
Train_data = Train_data[Non_zero_var_col]
Test_data = Test_data[Non_zero_var_col]
print("Shape", Train_data.shape, " ", Test_data.shape)

# Data Pre-processing
scaling = StandardScaler()
scaling.fit(Train_data)
Scaled_data = scaling.transform(Train_data)
Train = pd.DataFrame(Scaled_data, columns=list(Train_data.columns))
print(Train.shape)

test_Scaled_data = scaling.transform(Test_data)
Test = pd.DataFrame(test_Scaled_data, columns=list(Test_data.columns))
print(Test.shape)

#Random Forest model for feature selection
feature_selection = SelectFromModel(
    RandomForestRegressor(n_estimators=500, random_state=seed),
    max_features=topk,
)
feature_selection.fit(Train, y_train_reg)
Fs_Train = feature_selection.transform(Train)
Fs_Test = feature_selection.transform(Test)

feature_idx = feature_selection.get_support()
feature_name = Train.columns[feature_idx]

Final_col_name = []
for column in feature_name.tolist():
    Final_col_name.append(column)

train = pd.DataFrame(Fs_Train, columns=Final_col_name)
test = pd.DataFrame(Fs_Test, columns=Final_col_name)

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

with open(featurepath + "/Final_Features.txt", "w") as fp2:
    for columns in Final_col_name:
        fp2.write("%s\n" % columns)
    fp2.close()

# Grid Search Cross Validation
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=seed)
Result = {}

xg_regr = GradientBoostingRegressor()
score_list = ["neg_root_mean_squared_error", "r2"]
xg_regr_parameters = {
    "loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
    "max_depth": [1, 2, 3, 5, None],
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [100, 200, 300, 400, 500],
    "criterion": ["squared_error", "friedman_mse"],
    "max_features": ["sqrt", "log2", "auto", None],
    "random_state": [42],
}
xg_regr_model = GridSearchCV(
    xg_regr,
    xg_regr_parameters,
    cv=cv,
    scoring=score_list,
    refit="neg_root_mean_squared_error",
).fit(train, y_train_reg)
print(xg_regr_model.best_score_, " ", xg_regr_model.best_params_)

cv_df = pd.DataFrame(xg_regr_model.cv_results_)
cv_df.to_csv(others + "/CV_results.csv", sep="\t")

best_params = pd.DataFrame(xg_regr_model.best_params_, index=[0])
best_params.to_csv(others + "/best_params.csv", sep="\t")

# Generation and Training of Xgboost model
xg_model = GradientBoostingRegressor(
    loss=xg_regr_model.best_params_["loss"],
    learning_rate=xg_regr_model.best_params_["learning_rate"],
    n_estimators=xg_regr_model.best_params_["n_estimators"],
    criterion=xg_regr_model.best_params_["criterion"],
    max_depth=xg_regr_model.best_params_["max_depth"],
    max_features=xg_regr_model.best_params_["max_features"],
    random_state=xg_regr_model.best_params_["random_state"],
).fit(train, y_train_reg)

pickle.dump(xg_model, open(resultpath + "/" + model, "wb"))

train_dict = {"Model": model, "Topk": topk}
test_dict = {"Model": model, "Topk": topk}

Train_r2 = xg_model.score(train, y_train_reg)
Train_MSE = mean_squared_error(y_train_reg, np.round(xg_model.predict(train), 4))
Train_RMSE = mean_squared_error(y_train_reg, np.round(xg_model.predict(train), 4), squared=False)

Test_r2 = xg_model.score(test, y_test_reg)
Test_MSE = mean_squared_error(y_test_reg, np.round(xg_model.predict(test), 4))
Test_RMSE = mean_squared_error(y_test_reg, np.round(xg_model.predict(test), 4), squared=False)

train_dict["R2"] = Train_r2
train_dict["MSE"] = Train_MSE
train_dict["RMSE"] = Train_RMSE

test_dict["R2"] = Test_r2
test_dict["MSE"] = Test_MSE
test_dict["RMSE"] = Test_RMSE

#Export results to csv files
train_results = pd.DataFrame(train_dict, index=[0])
train_results.to_csv(resultpath + "/train_results.csv", sep="\t")

test_results = pd.DataFrame(test_dict, index=[0])
test_results.to_csv(resultpath + "/test_results.csv", sep="\t")
