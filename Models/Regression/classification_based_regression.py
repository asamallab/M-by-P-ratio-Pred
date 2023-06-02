'''
Evaluates performance of models on the internal test set.  
'''
#Import libraries
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
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import sys
import warnings

warnings.filterwarnings("ignore")
path = sys.argv[1]

#Import train and test data
Train_data = pd.read_csv(os.getcwd() + "/Dataprep/train.csv", sep="\t")
Test_data = pd.read_csv(os.getcwd() + "/Dataprep/test.csv", sep="\t")

Train_data.drop("Unnamed: 0", axis=1, inplace=True)
Test_data.drop("Unnamed: 0", axis=1, inplace=True)

#Transform the target variable for regression.
Train_data["MP_Ratio"] = np.round(np.log(Train_data["MP_Ratio"] + 1), 4)
Test_data["MP_Ratio"] = np.round(np.log(Test_data["MP_Ratio"] + 1), 4)

print(Train_data["MP_Ratio"].describe())
print(Test_data["MP_Ratio"].describe())

y_train_reg = Train_data['MP_Ratio']
y_test_reg = Test_data['MP_Ratio']

#Create the Risk variable for classification.
y_training = np.where(Train_data["MP_Ratio"] >= np.round(np.log(2), 4) , 1, 0)
y_testing = np.where(Test_data["MP_Ratio"] >= np.round(np.log(2), 4) , 1, 0)

Train_data.drop("MP_Ratio", axis=1, inplace=True)
Test_data.drop("MP_Ratio", axis=1, inplace=True)


Non_zero_var_col= []
Zero_var_col=[]
#Model selection based on Variance Thresholding, features having 0 variance are removed
for i in list(Train_data.columns):
    if Train_data[i].var():
        Non_zero_var_col.append(i)
    else:
        Zero_var_col.append(i)

Train_data = Train_data[Non_zero_var_col]
Test_data = Test_data[Non_zero_var_col]
print("Shape", Train_data.shape, " ", Test_data.shape)

#Scaling Data
scaling = StandardScaler()
scaling.fit(Train_data)
Scaled_data = scaling.transform(Train_data)
Train = pd.DataFrame(Scaled_data, columns=list(Train_data.columns))
print(Train.shape)

test_Scaled_data = scaling.transform(Test_data)
Test = pd.DataFrame(test_Scaled_data, columns=list(Test_data.columns))
print(Test.shape)

models = ["svm", "mlr", "xgboost", "pls", "randomforest", "mlr2", "mlr3"]
start = 5
topk = 501
increment = 5

for model in models:
    for top in range(start, topk, increment):
        #Import most important features for each model
        feature_path = path+'/Results/'+model+'/Top'+str(top)+'/featurepath/Final_Features.txt'
        trained_model = model
        model_path = path+'/Results/'+model+'/Top'+str(top)+'/'+trained_model
        if os.path.exists(model_path):
            train_dict = {}
            test_dict = {}

            top_features = []
            if os.path.exists(feature_path):
                with open(feature_path) as file:
                    top_features = [line.rstrip() for line in file]
            
            train = Train[top_features]
            test = Test[top_features]
            print(train.shape," ", test.shape)
            #Load saved model
            loaded_model = pickle.load(open(model_path, "rb"))
            
            #Predict model results for training data
            train_prediction = loaded_model.predict(train)
            print("Train Prediction" , train_prediction)
            print("R2 score", r2_score(y_train_reg, train_prediction))  #R2 score of model
            print("Score", loaded_model.score(train, y_train_reg))
            #Based on the milk to plasma ratio risk variable is computed
            train_prediction_class = np.where(train_prediction >= np.round(np.log(2), 4) , 1, 0)
            print(train_prediction_class," ", y_training.shape)

            train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_training, train_prediction_class).ravel()
            train_accuracy = accuracy_score(y_training, train_prediction_class)

            (
                train_precision,
                train_recal,
                train_fscore,
                train_support,
            ) = precision_recall_fscore_support(y_training, train_prediction_class, average="binary")
            train_dict['Model'] = model
            train_dict['Topk'] = top
            train_dict['Train_accuracy'] = train_accuracy
            train_dict['Train_tn'] = train_tn
            train_dict['Train_fp'] = train_fp
            train_dict['Train_fn'] = train_fn
            train_dict['Train_tp'] = train_tp
            train_dict['Train_precision'] = train_precision
            train_dict['Train_recall'] = train_recal
            train_dict['Train_fscore'] = train_fscore
            train_dict['Train_support'] = train_support


            #Predict model results for test data
            test_prediction = loaded_model.predict(test)

            print("R2 score", r2_score(y_test_reg, test_prediction))
            print("Score", loaded_model.score(test, y_test_reg))
            #Compute risk variable for test set based on Milk to Plasma Ratio predicted
            test_prediction_class = np.where(test_prediction >= np.round(np.log(2), 4) , 1, 0)
            print("test Prediction",test_prediction)
            print(test_prediction_class)
            print(y_testing)
            print(test_prediction_class.shape," ", y_testing.shape)
            #Compute Confusion Matrix
            test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_testing, test_prediction_class).ravel()
            test_accuracy = accuracy_score(y_testing, test_prediction_class)

            (
                test_precision,
                test_recal,
                test_fscore,
                test_support,
            ) = precision_recall_fscore_support(y_testing, test_prediction_class, average="binary")

            test_dict['Model'] = model
            test_dict['Topk'] = top
            test_dict['Test_accuracy'] = test_accuracy
            test_dict['Test_tn'] = test_tn
            test_dict['Test_fp'] = test_fp
            test_dict['Test_fn'] = test_fn
            test_dict['Test_tp'] = test_tp
            test_dict['Test_precision'] = test_precision
            test_dict['Test_recall'] = test_recal
            test_dict['Test_fscore'] = test_fscore
            test_dict['Test_support'] = test_support

            #Export the results.
            train_results = pd.DataFrame(train_dict, index=[0])
            train_results.to_csv(path+'/Results/'+model+'/Top'+str(top)+'/train_classification.csv', sep="\t")

            test_results = pd.DataFrame(test_dict, index=[0])
            test_results.to_csv(path+'/Results/'+model+'/Top'+str(top)+'/test_classification.csv', sep="\t")


