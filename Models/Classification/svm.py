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
from sklearn.svm import SVC
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

model = "svm"
topk = int(sys.argv[1])

#import train and test data
Train_data = pd.read_csv(os.getcwd() + "/../Dataset/train.csv", sep="\t")
Test_data = pd.read_csv(os.getcwd() + "/../Dataset/test.csv", sep="\t")

Train_data.drop("Unnamed: 0", axis=1, inplace=True)
Test_data.drop("Unnamed: 0", axis=1, inplace=True)


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

########change #############
SVM_classifier = SVC()
svm_parameters = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "class_weight": ["balanced"],
    "random_state": [seed],
    "gamma": ["auto", "scale"],
    "probability" : [True]
}
svm_model = GridSearchCV(SVM_classifier, svm_parameters, cv=cv).fit(train, Train_target)
print(svm_model.best_score_, " ", svm_model.best_params_)

cv_df = pd.DataFrame(svm_model.cv_results_)
cv_df.to_csv(others + "/CV_results.csv", sep="\t")

best_params = pd.DataFrame(svm_model.best_params_, index=[0])
best_params.to_csv(others + "/best_params.csv", sep="\t")
### Change 1 ########
# SVM Model Generation and Training 
svc_model = SVC(
    kernel=svm_model.best_params_["kernel"],
    gamma=svm_model.best_params_["gamma"],
    class_weight=svm_model.best_params_["class_weight"],
    probability = True,
    random_state=seed,
).fit(train, Train_target)
svm_train_accuracy = svc_model.score(train, Train_target)
print(svm_train_accuracy)
pickle.dump(svc_model, open(resultpath + "/" + model, "wb"))

train_dict = {"Model": model, "Topk": topk}
test_dict = {"Model": model, "Topk": topk}

svc_pred_train = svc_model.predict(train)
svc_train_tn, svc_train_fp, svc_train_fn, svc_train_tp = confusion_matrix(
    Train_target, svc_pred_train
).ravel()
print(
    "TN: ",
    svc_train_tn,
    "FP: ",
    svc_train_fp,
    "FN: ",
    svc_train_fn,
    "TP: ",
    svc_train_tp,
)

(
    svc_train_precision,
    svc_train_recal,
    svc_train_fscore,
    svc_train_support,
) = precision_recall_fscore_support(Train_target, svc_pred_train, average="binary")
print(
    "Precision: ",
    svc_train_precision,
    " ",
    "Recall: ",
    svc_train_recal,
    " ",
    "Fscore: ",
    svc_train_fscore,
    " ",
)

svc_train_accuracy = accuracy_score(Train_target, svc_pred_train)
print("Accuracy: ", svc_train_accuracy)

conf_matrix = confusion_matrix(y_true=Train_target, y_pred=svc_pred_train)
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
## change 2 #######
plt.close()

train_dict["Accuracy"] = svc_train_accuracy
train_dict["TN"] = svc_train_tn
train_dict["FP"] = svc_train_fp
train_dict["FN"] = svc_train_fn
train_dict["TP"] = svc_train_tp

train_dict["Precision"] = svc_train_precision
train_dict["Recal"] = svc_train_recal
train_dict["fscore"] = svc_train_fscore

svc_pred = svc_model.predict(test)
svc_tn, svc_fp, svc_fn, svc_tp = confusion_matrix(Test_target, svc_pred).ravel()
print("TN: ", svc_tn, "FP: ", svc_fp, "FN: ", svc_fn, "TP: ", svc_tp)
svc_precision, svc_recal, svc_fscore, svc_support = precision_recall_fscore_support(
    Test_target, svc_pred, average="binary"
)
print(
    "Precision: ",
    svc_precision,
    " ",
    "Recall: ",
    svc_recal,
    " ",
    "Fscore: ",
    svc_fscore,
    " ",
)

svc_test_accuracy = accuracy_score(Test_target, svc_pred)
print("Accuracy: ", svc_test_accuracy)

conf_matrix = confusion_matrix(y_true=Test_target, y_pred=svc_pred)
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
#### Change 3####
plt.close()

test_dict["Accuracy"] = svc_test_accuracy
test_dict["TN"] = svc_tn
test_dict["FP"] = svc_fp
test_dict["FN"] = svc_fn
test_dict["TP"] = svc_tp

test_dict["Precision"] = svc_precision
test_dict["Recal"] = svc_recal
test_dict["fscore"] = svc_fscore

train_results = pd.DataFrame(train_dict, index=[0])
train_results.to_csv(resultpath + "/train_results.csv", sep="\t")

test_results = pd.DataFrame(test_dict, index=[0])
test_results.to_csv(resultpath + "/test_results.csv", sep="\t")

############### ROC-AUC Plot #########

Roc_prob = svc_model.predict_proba(test)
Roc_prob = Roc_prob[:, 1]

print("shapes ", Test_target.shape," ",Roc_prob.shape)
svc_auc = roc_auc_score(Test_target, Roc_prob)
print('SVC: ROC AUC=%.3f' % (svc_auc))

svc_fpr, svc_tpr, _ = roc_curve(Test_target, Roc_prob)
print("TPR FPR")
#print(svc_fpr, svc_tpr)

line_probs = [0 for _ in range(len(Test_target))]
line_auc = roc_auc_score(Test_target, line_probs)
line_fpr, line_tpr, _ = roc_curve(Test_target, line_probs)


plt.plot(line_fpr, line_tpr, linestyle='--')
plt.plot(svc_fpr, svc_tpr, label="AUC="+str(round(svc_auc, 4)))
# axis labels
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
# show the legend
plt.legend(loc=4)
plt.title("SVC ROC", fontsize=18)
plt.savefig(plotpath + "/ROC_SVC.png")
plt.savefig(plotpath + "/ROC_SVC.pdf")
plt.close()

############# PR Curve ##############

svm_precision, svm_recall, svm_thresholds = precision_recall_curve(Test_target, Roc_prob)
print(len(svm_thresholds))
svm_auprc = auc(svm_recall, svm_precision)

svm_avg_precision = average_precision_score(Test_target, Roc_prob)
print("SVM", svm_auprc," ",svm_avg_precision)
plt.plot(svm_recall, svm_precision, label='SVM = %0.2f' % svm_auprc)
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
