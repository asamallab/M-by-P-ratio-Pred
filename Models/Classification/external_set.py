'''
Evaluates performance of models on the external test set consisting of ExHuMid chemicals.  
'''
#Import libraries
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import statistics as stat

path = sys.argv[1]

#Set threshold for Domain of Applicability
DAP_threshold = 3

#Import External Test Sets
External_test = pd.read_csv(
    os.getcwd() + "/descriptors_1d2d3d_exhumid.csv", sep="\t"
)
External_test.drop("Chemical_Identifier", axis=1, inplace=True)

if "corr_rem" in path:
    Train_data = pd.read_csv(
        os.getcwd() + "/../" + path + "/processed_data/processed_train.csv", sep="\t"
    )
    Train_data.drop("Unnamed: 0.1", axis=1, inplace=True)
else:
    Train_data = pd.read_csv(os.getcwd() + "/../Dataprep/train.csv", sep="\t")

Train_data.drop("Unnamed: 0", axis=1, inplace=True)
Train_data.drop("MP_Ratio", inplace=True, axis=1)

if "Boruta" in path:
    start = 1
    increment = 1
    topk = 6
else:
    start = 5
    increment = 5
    topk = 501

#Scaling Data
scaling = StandardScaler()
scaling.fit(Train_data)
Scaled_Train_data = scaling.transform(Train_data)
Train = pd.DataFrame(Scaled_Train_data, columns=list(Train_data.columns))

if "corr_rem" in path:
    External_test = External_test[list(Train_data.columns)]

Scaled_External_data = scaling.transform(External_test)
External_data = pd.DataFrame(Scaled_External_data, columns=list(External_test.columns))

models = ["svm", "lda", "randomforest", "mlp", "mlp2", "mlp3", "xgboost"]


for model in models:
    for top in range(start, topk, increment):
        model_result_dict = {"Model": model, "Topk": top}

        top_features_path = (
            os.getcwd()
            + "/../"
            + path
            + "/Results/"
            + model
            + "/Top"
            + str(top)
            + "/featurepath/Final_Features.txt"
        )
        top_features = []
        #Extract top features for each model 
        if os.path.exists(top_features_path):
            with open(top_features_path) as file:
                top_features = [line.rstrip() for line in file]

        if len(top_features) > 0:
            print(model," ",top)
            result_path = (
                os.getcwd() + "/../" + path + "/Results/" + model + "/Top" + str(top)
            )
            plot_path = (
                os.getcwd()
                + "/../"
                + path
                + "/Results/"
                + model
                + "/Top"
                + str(top)
                + "/External_test_plots"
            )

            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            External_reduced_data = External_data[top_features]
            External_reduced_data.reset_index(inplace=True, drop=True)
            print("Before: ", External_reduced_data.shape)
            #Assess data based on domain of applicability 
            APD_test = pd.DataFrame()
            for index, rows in External_reduced_data.iterrows():
                row_list = External_reduced_data.iloc[[index]].values.tolist()[0]
                if len(row_list) != len(top_features):
                    print("Error in Size")
                abs_row_list = list(map(abs, row_list))
                max_val = max(abs_row_list)
                if max_val <= DAP_threshold:
                    APD_test = APD_test.append(External_reduced_data.iloc[[index]])
                else:
                    min_val = min(abs_row_list)
                    if min_val > DAP_threshold:
                        continue
                    else:
                        S_new = stat.mean(abs_row_list) + (
                            1.28 * stat.stdev(abs_row_list)
                        )
                        if S_new > DAP_threshold:
                            continue
                        else:
                            APD_test = APD_test.append(
                                External_reduced_data.iloc[[index]]
                            )

            External_reduced_data = APD_test.copy()
            target = np.ones(External_reduced_data.shape[0])
            print("After:", External_reduced_data.shape)

            trained_model = model
            if model == 'xgboost':
                trained_model = 'xg_model'
            model_path = (
                os.getcwd()
                + "/../"
                + path
                + "/Results/"
                + model
                + "/Top"
                + str(top)
                + "/"
                + trained_model
            )
            # Import saved model
            if os.path.exists(model_path):
                loaded_model = pickle.load(open(model_path, "rb"))
            
            #Model Predictions on External set
            model_pred = loaded_model.predict(External_reduced_data)
            print(target.shape, " ", model_pred.shape)

            #Create confusion matrix
            model_tn, model_fp, model_fn, model_tp = confusion_matrix(
                target, model_pred
            ).ravel()
            print(
                "TN: ", model_tn, "FP: ", model_fp, "FN: ", model_fn, "TP: ", model_tp
            )
            (
                model_precision,
                model_recal,
                model_fscore,
                model_support,
            ) = precision_recall_fscore_support(target, model_pred, average="binary")
            print(
                "Precision: ",
                model_precision,
                " ",
                "Recall: ",
                model_recal,
                " ",
                "Fscore: ",
                model_fscore,
                " ",
            )

            model_accuracy = accuracy_score(target, model_pred)
            print("Accuracy: ", model_accuracy)

            conf_matrix = confusion_matrix(y_true=target, y_pred=model_pred)
            #
            # Print the confusion matrix using Matplotlib
            #
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(
                        x=j,
                        y=i,
                        s=conf_matrix[i, j],
                        va="center",
                        ha="center",
                        size="xx-large",
                    )

            plt.xlabel("Predictions", fontsize=18)
            plt.ylabel("Actuals", fontsize=18)
            plt.title("Confusion Matrix", fontsize=18)
            plt.savefig(plot_path + "/External_test_confusion_matrix.png")
            plt.savefig(plot_path + "/External_test_confusion_matrix.pdf")
            # plt.show()

            model_result_dict["Accuracy"] = model_accuracy
            model_result_dict["TN"] = model_tn
            model_result_dict["FP"] = model_fp
            model_result_dict["FN"] = model_fn
            model_result_dict["TP"] = model_tp

            model_result_dict["Precision"] = model_precision
            model_result_dict["Recal"] = model_recal
            model_result_dict["fscore"] = model_fscore

            model_results = pd.DataFrame(model_result_dict, index=[0])
            #Export model results on external set
            model_results.to_csv(result_path + "/External_test_results.csv", sep="\t")
