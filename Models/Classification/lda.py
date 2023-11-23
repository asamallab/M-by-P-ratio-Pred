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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

model = "lda"
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


lda_classifier = LinearDiscriminantAnalysis()
lda_parameters = {
    "solver": ["svd", "lsqr", "eigen"],
    "shrinkage": ["auto", "None"],
}
lda_model = GridSearchCV(lda_classifier, lda_parameters, cv=cv).fit(train, Train_target)
print(lda_model.best_score_, " ", lda_model.best_params_)

cv_df = pd.DataFrame(lda_model.cv_results_)
cv_df.to_csv(others + "/CV_results.csv", sep="\t")

best_params = pd.DataFrame(lda_model.best_params_, index=[0])
best_params.to_csv(others + "/best_params.csv", sep="\t")


# LDA Model Generation and Training
lda_model = LinearDiscriminantAnalysis(
    solver=lda_model.best_params_["solver"],
    shrinkage=lda_model.best_params_["shrinkage"],
).fit(train, Train_target)
lda_train_accuracy = lda_model.score(train, Train_target)
print(lda_train_accuracy)
pickle.dump(lda_model, open(resultpath + "/" + model, "wb"))

train_dict = {"Model": model, "Topk": topk}
test_dict = {"Model": model, "Topk": topk}

lda_pred_train = lda_model.predict(train)
lda_train_tn, lda_train_fp, lda_train_fn, lda_train_tp = confusion_matrix(
    Train_target, lda_pred_train
).ravel()
print(
    "TN: ",
    lda_train_tn,
    "FP: ",
    lda_train_fp,
    "FN: ",
    lda_train_fn,
    "TP: ",
    lda_train_tp,
)

(
    lda_train_precision,
    lda_train_recal,
    lda_train_fscore,
    lda_train_support,
) = precision_recall_fscore_support(Train_target, lda_pred_train, average="binary")
print(
    "Precision: ",
    lda_train_precision,
    " ",
    "Recall: ",
    lda_train_recal,
    " ",
    "Fscore: ",
    lda_train_fscore,
    " ",
)

lda_train_accuracy = accuracy_score(Train_target, lda_pred_train)
print("Accuracy: ", lda_train_accuracy)

conf_matrix = confusion_matrix(y_true=Train_target, y_pred=lda_pred_train)
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

train_dict["Accuracy"] = lda_train_accuracy
train_dict["TN"] = lda_train_tn
train_dict["FP"] = lda_train_fp
train_dict["FN"] = lda_train_fn
train_dict["TP"] = lda_train_tp

train_dict["Precision"] = lda_train_precision
train_dict["Recal"] = lda_train_recal
train_dict["fscore"] = lda_train_fscore

lda_pred = lda_model.predict(test)
lda_tn, lda_fp, lda_fn, lda_tp = confusion_matrix(Test_target, lda_pred).ravel()
print("TN: ", lda_tn, "FP: ", lda_fp, "FN: ", lda_fn, "TP: ", lda_tp)
lda_precision, lda_recal, lda_fscore, lda_support = precision_recall_fscore_support(
    Test_target, lda_pred, average="binary"
)
print(
    "Precision: ",
    lda_precision,
    " ",
    "Recall: ",
    lda_recal,
    " ",
    "Fscore: ",
    lda_fscore,
    " ",
)

lda_test_accuracy = accuracy_score(Test_target, lda_pred)
print("Accuracy: ", lda_test_accuracy)

conf_matrix = confusion_matrix(y_true=Test_target, y_pred=lda_pred)
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

test_dict["Accuracy"] = lda_test_accuracy
test_dict["TN"] = lda_tn
test_dict["FP"] = lda_fp
test_dict["FN"] = lda_fn
test_dict["TP"] = lda_tp

test_dict["Precision"] = lda_precision
test_dict["Recal"] = lda_recal
test_dict["fscore"] = lda_fscore

train_results = pd.DataFrame(train_dict, index=[0])
train_results.to_csv(resultpath + "/train_results.csv", sep="\t")

test_results = pd.DataFrame(test_dict, index=[0])
test_results.to_csv(resultpath + "/test_results.csv", sep="\t")


############### ROC-AUC Plot #########

Roc_prob = lda_model.predict_proba(test)
Roc_prob = Roc_prob[:, 1]

print("shapes ", Test_target.shape," ",Roc_prob.shape)
lda_auc = roc_auc_score(Test_target, Roc_prob)
print('LDA: ROC AUC=%.3f' % (lda_auc))

lda_fpr, lda_tpr, _ = roc_curve(Test_target, Roc_prob)

line_probs = [0 for _ in range(len(Test_target))]
line_auc = roc_auc_score(Test_target, line_probs)
line_fpr, line_tpr, _ = roc_curve(Test_target, line_probs)


plt.plot(line_fpr, line_tpr, linestyle='--')
plt.plot(lda_fpr, lda_tpr, label="AUC="+str(round(lda_auc, 4)))
# axis labels
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
# show the legend
plt.legend(loc=4)
plt.title("LDA ROC", fontsize=18)
plt.savefig(plotpath + "/ROC_LDA.png")
plt.savefig(plotpath + "/ROC_LDA.pdf")
plt.close()

######### PR Curve #########

lda_precision, lda_recall, lda_thresholds = precision_recall_curve(Test_target, Roc_prob)
print(len(lda_thresholds))
lda_auprc = auc(lda_recall, lda_precision)

lda_avg_precision = average_precision_score(Test_target, Roc_prob)
print("LDA", lda_auprc," ",lda_avg_precision)
plt.plot(lda_recall, lda_precision, label='LDA = %0.2f' % lda_auprc)
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
