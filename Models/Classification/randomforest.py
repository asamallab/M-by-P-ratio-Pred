#import libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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

model = "randomforest"
topk = int(sys.argv[1])

#import train and test data
Train_data = pd.read_csv(os.getcwd() + "/../Dataprep/train.csv", sep="\t")
Test_data = pd.read_csv(os.getcwd() + "/../Dataprep/test.csv", sep="\t")

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

Non_zero_var_col = []
Zero_var_col = []
for i in list(Train_data.columns):
    if Train_data[i].var():
        Non_zero_var_col.append(i)
    else:
        Zero_var_col.append(i)

#Remove Zero Variance Features
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

rf_classifier = RandomForestClassifier()
rf_parameters = {
    "class_weight": ["balanced", "balanced_subsample"],
    "criterion": ["gini", "entropy", "log_loss"],
    "n_estimators": [100, 500, 1000],
    "min_samples_split": [2, 3, 4],
    "max_depth": [3, 5, 7, None],
    "max_features": ["auto", "sqrt", None],
    "random_state": [42],
}

rf_model = GridSearchCV(rf_classifier, rf_parameters, cv=cv).fit(train, Train_target)
print(rf_model.best_score_, " ", rf_model.best_params_)

cv_df = pd.DataFrame(rf_model.cv_results_)
cv_df.to_csv(others + "/CV_results.csv", sep="\t")

best_params = pd.DataFrame(rf_model.best_params_, index=[0])
best_params.to_csv(others + "/best_params.csv", sep="\t")

# Random Forest Model Generation and Training for Classification
random_forest_model = RandomForestClassifier(
    class_weight=rf_model.best_params_["class_weight"],
    criterion=rf_model.best_params_["criterion"],
    max_depth=rf_model.best_params_["max_depth"],
    min_samples_split=rf_model.best_params_["min_samples_split"],
    n_estimators=rf_model.best_params_["n_estimators"],
    max_features=rf_model.best_params_["max_features"],
    random_state=seed,
).fit(train, Train_target)
rf_train_accuracy = random_forest_model.score(train, Train_target)
print(rf_train_accuracy)
pickle.dump(random_forest_model, open(resultpath + "/" + model, "wb"))

train_dict = {"Model": model, "Topk": topk}
test_dict = {"Model": model, "Topk": topk}

rf_pred_train = random_forest_model.predict(train)
(
    rf_train_tn,
    rf_train_fp,
    rf_train_fn,
    rf_train_tp,
) = confusion_matrix(Train_target, rf_pred_train).ravel()
print(
    "TN: ",
    rf_train_tn,
    "FP: ",
    rf_train_fp,
    "FN: ",
    rf_train_fn,
    "TP: ",
    rf_train_tp,
)

(
    rf_train_precision,
    rf_train_recal,
    rf_train_fscore,
    rf_train_support,
) = precision_recall_fscore_support(Train_target, rf_pred_train, average="binary")
print(
    "Precision: ",
    rf_train_precision,
    " ",
    "Recall: ",
    rf_train_recal,
    " ",
    "Fscore: ",
    rf_train_fscore,
    " ",
)

rf_train_accuracy = accuracy_score(Train_target, rf_pred_train)
print("Accuracy: ", rf_train_accuracy)

conf_matrix = confusion_matrix(y_true=Train_target, y_pred=rf_pred_train)
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

train_dict["Accuracy"] = rf_train_accuracy
train_dict["TN"] = rf_train_tn
train_dict["FP"] = rf_train_fp
train_dict["FN"] = rf_train_fn
train_dict["TP"] = rf_train_tp

train_dict["Precision"] = rf_train_precision
train_dict["Recal"] = rf_train_recal
train_dict["fscore"] = rf_train_fscore

rf_pred = random_forest_model.predict(test)
rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(Test_target, rf_pred).ravel()
print("TN: ", rf_tn, "FP: ", rf_fp, "FN: ", rf_fn, "TP: ", rf_tp)
(
    rf_precision,
    rf_recal,
    rf_fscore,
    rf_support,
) = precision_recall_fscore_support(Test_target, rf_pred, average="binary")
print(
    "Precision: ",
    rf_precision,
    " ",
    "Recall: ",
    rf_recal,
    " ",
    "Fscore: ",
    rf_fscore,
    " ",
)

rf_test_accuracy = accuracy_score(Test_target, rf_pred)
print("Accuracy: ", rf_test_accuracy)

conf_matrix = confusion_matrix(y_true=Test_target, y_pred=rf_pred)
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

test_dict["Accuracy"] = rf_test_accuracy
test_dict["TN"] = rf_tn
test_dict["FP"] = rf_fp
test_dict["FN"] = rf_fn
test_dict["TP"] = rf_tp

test_dict["Precision"] = rf_precision
test_dict["Recal"] = rf_recal
test_dict["fscore"] = rf_fscore

train_results = pd.DataFrame(train_dict, index=[0])
train_results.to_csv(resultpath + "/train_results.csv", sep="\t")

test_results = pd.DataFrame(test_dict, index=[0])
test_results.to_csv(resultpath + "/test_results.csv", sep="\t")

############### ROC-AUC Plot #########

Roc_prob = random_forest_model.predict_proba(test)
Roc_prob = Roc_prob[:, 1]

print("shapes ", Test_target.shape," ",Roc_prob.shape)
rf_auc = roc_auc_score(Test_target, Roc_prob)
print('RF: ROC AUC=%.3f' % (rf_auc))

rf_fpr, rf_tpr, _ = roc_curve(Test_target, Roc_prob)

line_probs = [0 for _ in range(len(Test_target))]
line_auc = roc_auc_score(Test_target, line_probs)
line_fpr, line_tpr, _ = roc_curve(Test_target, line_probs)


plt.plot(line_fpr, line_tpr, linestyle='--')
plt.plot(rf_fpr, rf_tpr, label="AUC="+str(round(rf_auc, 4)))
# axis labels
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
# show the legend
plt.legend(loc=4)
plt.title("RF ROC", fontsize=18)
plt.savefig(plotpath + "/ROC_RF.png")
plt.savefig(plotpath + "/ROC_RF.pdf")
plt.close()

######## PR Curve #########
rf_precision, rf_recall, rf_thresholds = precision_recall_curve(Test_target, Roc_prob)
print(len(rf_thresholds))
rf_auprc = auc(rf_recall, rf_precision)

rf_avg_precision = average_precision_score(Test_target, Roc_prob)
print("RF", rf_auprc," ",rf_avg_precision)
plt.plot(rf_recall, rf_precision, label='Random Forest = %0.2f' % rf_auprc)
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