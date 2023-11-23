"""
Program to build Xgboost model for classification of chemical compounds.
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
from sklearn.metrics import plot_confusion_matrix
seed = 42

model = "xgboost"
topk = int(sys.argv[1])

#import train and test data
Train_data = pd.read_csv(os.getcwd() + "/../Dataset/train.csv", sep="\t")
Test_data = pd.read_csv(os.getcwd() + "/../Dataset/test.csv", sep="\t")

Train_data.drop("Unnamed: 0", axis=1, inplace=True)
Test_data.drop("Unnamed: 0", axis=1, inplace=True)

#Generate the risk feature for classification of chemical compounds
def y_class(x):
    if x["MP_Ratio"] < 1:
        return 0
    else:
        return 1


Train_target = Train_data.apply(y_class, axis=1)
Test_target = Test_data.apply(y_class, axis=1)
print(Train_target.value_counts(), "\n", Test_target.value_counts())

Train_data.drop("MP_Ratio", inplace=True, axis=1)
Test_data.drop("MP_Ratio", inplace=True, axis=1)
print(Train_data.shape, " ", Test_data.shape)

"""Remove Zero Variance"""

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

#Remove Zero Variance Features
Non_zero_var_col = []
Zero_var_col = []
for i in list(Train_data.columns):
    if Train_data[i].var():
        Non_zero_var_col.append(i)
    else:
        Zero_var_col.append(i)

Train_data = Train_data[Non_zero_var_col]
Test_data = Test_data[Non_zero_var_col]
print(Train_data.shape, " ", Test_data.shape)

with open(featurepath + "/Zero_variance_columns.txt", "w") as fp:
    for columns in Zero_var_col:
        fp.write("%s\n" % columns)
    fp.close()

with open(featurepath + "/Non_Zero_variance_columns.txt", "w") as fp1:
    for columns in Non_zero_var_col:
        fp1.write("%s\n" % columns)
    fp1.close()

# Data Pre-processing
scaling = StandardScaler()
scaling.fit(Train_data)
Scaled_Train_data = scaling.transform(Train_data)
Train = pd.DataFrame(Scaled_Train_data, columns=list(Train_data.columns))

Scaled_Test_data = scaling.transform(Test_data)
Test = pd.DataFrame(Scaled_Test_data, columns=list(Test_data.columns))

print(Train.shape, " ", Test.shape)

forest = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)
feature_selection = BorutaPy(forest, n_estimators='auto', random_state=seed)
print(type(Train), " ",type(Train_target))


feature_selection.fit(Train.to_numpy(),Train_target.to_numpy())

Final_col_name = []

features = list(Train.columns)

Rank = feature_selection.ranking_

col_dict = {}
for i in range(len(Rank)):
    col_dict[features[i]]=Rank[i]

col_dict = dict(sorted(col_dict.items(), key=lambda item: item[1]))

boruta_results = pd.DataFrame(col_dict, index=[0])
boruta_results.T.to_csv(featurepath + "/boruta_results.csv", sep="\t")

for key,values in col_dict.items():
    if values <= topk:
        Final_col_name.append(key)

train = Train[Final_col_name]
test = Test[Final_col_name]

with open(featurepath + "/Final_Features.txt", "w") as fp2:
    for columns in Final_col_name:
        fp2.write("%s\n" % columns)
    fp2.close()

train.head(5)

print(train.shape, " ", test.shape)

"""Grid Search"""

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=seed)
Result = {}

xg_boost_classifier = GradientBoostingClassifier()
xg_boost_parameters = {
    "loss": ["deviance", "exponential"],
    "criterion": ["friedman_mse", "squared_error", "mse"],
    "n_estimators": [100, 500, 1000],
    "min_samples_split": [2, 3, 4],
    "max_depth": [3, 5, 7],
    "max_features": ["auto", "sqrt", "log2"],
    "n_iter_no_change": [3, 5],
    "random_state": [42],
}
xg_boost_model = GridSearchCV(xg_boost_classifier, xg_boost_parameters, cv=cv).fit(
    train, Train_target
)
print(xg_boost_model.best_score_, " ", xg_boost_model.best_params_)

cv_df = pd.DataFrame(xg_boost_model.cv_results_)
cv_df.to_csv(others + "/CV_results.csv", sep="\t")

best_params = pd.DataFrame(xg_boost_model.best_params_, index=[0])
best_params.to_csv(others + "/best_params.csv", sep="\t")

# Xgboost Model Generation and Training
xg_model = GradientBoostingClassifier(
    loss=xg_boost_model.best_params_["loss"],
    n_iter_no_change=xg_boost_model.best_params_["n_iter_no_change"],
    criterion=xg_boost_model.best_params_["criterion"],
    max_depth=xg_boost_model.best_params_["max_depth"],
    min_samples_split=xg_boost_model.best_params_["min_samples_split"],
    n_estimators=xg_boost_model.best_params_["n_estimators"],
    max_features=xg_boost_model.best_params_["max_features"],
    random_state=seed,
).fit(train, Train_target)
xg_boost_train_accuracy = xg_model.score(train, Train_target)
print(xg_boost_train_accuracy)
pickle.dump(xg_model, open(resultpath + "/xg_model", "wb"))

train_dict = {"Model": model, "Topk": topk}
test_dict = {"Model": model, "Topk": topk}

xg_pred_train = xg_model.predict(train)
(
    xg_boost_train_tn,
    xg_boost_train_fp,
    xg_boost_train_fn,
    xg_boost_train_tp,
) = confusion_matrix(Train_target, xg_pred_train).ravel()
print(
    "TN: ",
    xg_boost_train_tn,
    "FP: ",
    xg_boost_train_fp,
    "FN: ",
    xg_boost_train_fn,
    "TP: ",
    xg_boost_train_tp,
)

(
    xg_boost_train_precision,
    xg_boost_train_recal,
    xg_boost_train_fscore,
    xg_boost_train_support,
) = precision_recall_fscore_support(Train_target, xg_pred_train, average="binary")
print(
    "Precision: ",
    xg_boost_train_precision,
    " ",
    "Recall: ",
    xg_boost_train_recal,
    " ",
    "Fscore: ",
    xg_boost_train_fscore,
    " ",
)

xg_boost_train_accuracy = accuracy_score(Train_target, xg_pred_train)
print("Accuracy: ", xg_boost_train_accuracy)

conf_matrix = confusion_matrix(y_true=Train_target, y_pred=xg_pred_train)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(
            x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
        )

plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Actuals", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.savefig(plotpath + "/train_confusion_matrix.png")
plt.savefig(plotpath + "/train_confusion_matrix.pdf")
# plt.show()
plt.close()

train_dict["Accuracy"] = xg_boost_train_accuracy
train_dict["TN"] = xg_boost_train_tn
train_dict["FP"] = xg_boost_train_fp
train_dict["FN"] = xg_boost_train_fn
train_dict["TP"] = xg_boost_train_tp

train_dict["Precision"] = xg_boost_train_precision
train_dict["Recal"] = xg_boost_train_recal
train_dict["fscore"] = xg_boost_train_fscore

xg_pred = xg_model.predict(test)
xg_boost_tn, xg_boost_fp, xg_boost_fn, xg_boost_tp = confusion_matrix(
    Test_target, xg_pred
).ravel()
print(
    "TN: ", xg_boost_tn, "FP: ", xg_boost_fp, "FN: ", xg_boost_fn, "TP: ", xg_boost_tp
)
(
    xg_boost_precision,
    xg_boost_recal,
    xg_boost_fscore,
    xg_boost_support,
) = precision_recall_fscore_support(Test_target, xg_pred, average="binary")
print(
    "Precision: ",
    xg_boost_precision,
    " ",
    "Recall: ",
    xg_boost_recal,
    " ",
    "Fscore: ",
    xg_boost_fscore,
    " ",
)

xg_boost_test_accuracy = accuracy_score(Test_target, xg_pred)
print("Accuracy: ", xg_boost_test_accuracy)

conf_matrix = confusion_matrix(y_true=Test_target, y_pred=xg_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(
            x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
        )

plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Actuals", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.savefig(plotpath + "/test_confusion_matrix.png")
plt.savefig(plotpath + "/test_confusion_matrix.pdf")
# plt.show()
plt.close()

test_dict["Accuracy"] = xg_boost_test_accuracy
test_dict["TN"] = xg_boost_tn
test_dict["FP"] = xg_boost_fp
test_dict["FN"] = xg_boost_fn
test_dict["TP"] = xg_boost_tp

test_dict["Precision"] = xg_boost_precision
test_dict["Recal"] = xg_boost_recal
test_dict["fscore"] = xg_boost_fscore

train_results = pd.DataFrame(train_dict, index=[0])
train_results.to_csv(resultpath + "/train_results.csv", sep="\t")

test_results = pd.DataFrame(test_dict, index=[0])
test_results.to_csv(resultpath + "/test_results.csv", sep="\t")

############### ROC-AUC Plot #########

Roc_prob = xg_model.predict_proba(test)
Roc_prob = Roc_prob[:, 1]

print("shapes ", Test_target.shape," ",Roc_prob.shape)
xg_auc = roc_auc_score(Test_target, Roc_prob)
print('Xg boost: ROC AUC=%.3f' % (xg_auc))

xg_fpr, xg_tpr, _ = roc_curve(Test_target, Roc_prob)

line_probs = [0 for _ in range(len(Test_target))]
line_auc = roc_auc_score(Test_target, line_probs)
line_fpr, line_tpr, _ = roc_curve(Test_target, line_probs)


plt.plot(line_fpr, line_tpr, linestyle='--')
plt.plot(xg_fpr, xg_tpr, label="AUC="+str(round(xg_auc, 4)))
# axis labels
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
# show the legend
plt.legend(loc=4)
plt.title("Xg boost ROC", fontsize=18)
plt.savefig(plotpath + "/ROC_XG.png")
plt.savefig(plotpath + "/ROC_XG.pdf")
plt.close()

######### PR Curve ######

xg_precision, xg_recall, xg_thresholds = precision_recall_curve(Test_target, Roc_prob)
print(len(xg_thresholds))
xg_auprc = auc(xg_recall, xg_precision)

xg_avg_precision = average_precision_score(Test_target, Roc_prob)
print("XG", xg_auprc," ",xg_avg_precision)
plt.plot(xg_recall, xg_precision, label='Xg boost = %0.2f' % xg_auprc)
plt.plot([0, 1], [sum(Test_target)/len(Test_target)]*2, linestyle='--')

# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend(loc='upper right')
plt.title("Precision-Recall Curve", fontsize=12)

plt.savefig(plotpath + "/PR_curve.png")
plt.savefig(plotpath + "/PR_curve.pdf")
plt.close()
