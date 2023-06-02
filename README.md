# Exposome and Human Milk: Machine Learning for Chemical Propensity Prediction

This repository contains the code and dataset used for the manuscript titled "Exposome and Human Milk: Machine Learning for Chemical Propensity Prediction." In this research project, we aim to predict the entry of exogenous chemicals into human breast milk using machine-learning techniques. Further, we predict Milk-to-Plasma ratio of these models as well.

# Repository Organization
```
Dataset - This folder contains the train, test, and External dataset used in this study
Models 
  ├── Classification - Codes used for classification models
  ├── Regression - Codes used for Regression models
Trained_Models - This folder contains the models trained on our dataset
ReadMe.md - Contains project and dataset description, along with steps to run the codes.
```
  
## Dataset
We utilized a comprehensive collection of Milk-to-Plasma (M/P) ratio data to build and evaluate our predictive model. The data set comprises M/P ratios of 375 different chemical compounds. These ratios represent the concentration of chemicals in human milk relative to their concentration in maternal plasma. To generate features for these compounds, we used PaDEL, which is open-source software that calculates various molecular descriptors.
To evaluate the generalizability of our model, we used an external test set of 206 compounds collected from ExHuMId, ExHuMUS, and ExHuM Explorer. These compounds were not included in the training data set, and we used them to assess the performance of our model on new and unseen data.
* train.csv - Data used for training
* test.csv - Data used for testing
* descriptors_1d2d3d_exhumid.csv - External set 

## Models
The code in this repository provides a complete workflow for predicting the entry of exogenous chemicals into human breast milk using machine learning. The workflow includes feature pre-processing, feature selection, hyperparameter tuning, training, and evaluation.

Classification-  <br /><br />
* lda.py - Python file to train and evaluate Linear Discrimination Analysis model for predicting entry of chemicals into breast milk.<br />
* mlp.py - Python file to train and evaluate Multi Layer Perceptron model for predicting entry of chemicals into breast milk.<br />
* randomforest.py - Python file to train and evaluate Random Forest model for predicting entry of chemicals into breast milk.<br />
* svm.py - Python file to train and evaluate Support Vector Machine model for predicting entry of chemicals into breast milk.<br />
* xg_boost.py - Python file to train and evaluate Xgboost model for predicting entry of chemicals into breast milk.<br />
* external_set - Python file that contains code for model evaluation in External set after applying Domain of applicability.<br />

Regression- <br /><br />
* randomforest.py - Python file to train and evaluate Random Forest model for prediction of milk-to-plasma ratio of compounds.<br /> 
* svm.py - Python file to train and evaluate Support Vector Machine model for prediction of milk-to-plasma ratio of compounds. <br />
* xgboost.py - Python file to train and evaluate Xgboost models for prediction of milk-to-plasma ratio of compounds.<br />


Codes for these are included in the Models folder.  Further, external_set_Classification.py and external_set_Regression.py include code for testing machine learning models on ExHumid Dataset.


## Steps to run the code-
1.	Use the following command to download all the required dependencies. 
   ```
   pip3 install -r requirements.txt
   ```

2.	Compute structures of molecules using PubChem website.

  PubChem:https://pubchem.ncbi.nlm.nih.gov/
  
3. Verify chemical structures based on Tanimoto coefficient using chemical.py file.
   ```
   python3 chemical.py <path of dataset>
   ```

3.	Use the PaDEL software to compute descriptors for the chemicals. PaDEL Link:  http://padel.nus.edu.sg/software/padeldescriptor

4.	Commands to run the python file for classification and regression tasks.
    ```
    Python3 <path to python file> <no. Of top features to be considered>
    ```
5. Commands to run external set and classification based on regression.
    ```
    Python3 <path to python file>  <path to result folder>
    ```

   
 





