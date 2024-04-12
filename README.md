# M/P-ratio-Pred

This repository provides the datasets and codes associated with the following manuscript:<br><br>
[<i>Machine learning models for prediction of xenobiotic chemicals with high propensity to transfer into human milk</i>](https://pubs.acs.org/doi/10.1021/acsomega.3c09392),<br>
Sudharsan Vijayaraghavan, Akshaya Lakshminarayanan, Naman Bhargava, Janani Ravichandran,  R.P. Vivek-Ananth*, Areejit Samal*, <br>
ACS Omega, 9(11):13006-13016, 2024
(*Corresponding author)

# Schematic Workflow
<figure>
  <img src="https://github.com/asamallab/M-by-P-ratio-Pred/blob/main/SchematicWorkflow.png" alt="SchematicWorkflow" style="width:100%">
  <figcaption>Schematic diagram summarizes the workflow to build the classification- and regression-based machine learning models to predict xenobiotic chemicals with high propensity to transfer from maternal plasma to human milk. The figure shows the key steps involved in data curation, feature generation, data preprocessing, feature selection, and the training and evaluation of classification- and regression-based machine learning models.</figcaption>
</figure>

# Repository Organization
```
Dataset - This folder contains the train, (internal) test, and external test dataset used in this study
Models 
  ├── Classification - Codes used for classification models
  ├── Regression - Codes used for Regression models
ReadMe.md - Contains project and dataset description, along with steps to run the codes.
```
  
## Dataset
 To build the machine learning models, we leveraged a curated dataset of 375 chemicals with experimentally determined M/P ratios compiled from Vasios et al. (PMID: [27573378](https://pubmed.ncbi.nlm.nih.gov/27573378/)) and other published literature.  For each chemical in this dataset, we obtained the 2D structure, generated the 3D structure, and computed 1875 molecular descriptors using PaDEL. We evaluated the generalizability of our best classification models by leveraging an external test dataset, comprising 202 chemicals, with high risk of transfer from maternal plasma to human milk.
* train.csv - Training data
* test.csv - (Internal) test data
* external_test_dataset.csv - External test dataset 

## Models
The codes in this repository enable the reproduction of the results present in the manuscript to predict xenobiotic chemicals with a high propensity to transfer from maternal plasma to human milk. The code provided for the five  models corresponding to the five different classification algorithms and three models corresponding to three different regression algorithms performs end-to-end processing of the data including the feature pre-processing, feature selection, hyperparameter tuning, training, and evaluation of the models.

Classification-  <br />
* svm.py - Python code to train and evaluate Support Vector Machine model.<br />
* xg_boost.py - Python code to train and evaluate XGBoost model.<br />
* lda.py - Python code to train and evaluate Linear Discriminant Analysis model.<br />
* mlp.py - Python code to train and evaluate Multi Layer Perceptron model.<br />
* randomforest.py - Python code to train and evaluate Random Forest model.<br />
* external_set.py - Python code for evaluating the model on the external test dataset after applying domain of applicability.<br />

Regression- <br />
* svm.py - Python code to train and evaluate Support Vector Machine model. <br />
* xgboost.py - Python code to train and evaluate Xgboost models.<br />
* randomforest.py - Python code to train and evaluate Random Forest model.<br />
* classification_based_regression.py - Python code to  evaluate the classification based on the regression model on the (internal) test set.
* external_set.py - Python code for evaluating the classification based on regression model on the external test dataset after applying domain of applicability.<br />

## Syntax to run the codes
1.	Use the following command to download all the required dependencies.
```
   pip3 install -r requirements.txt
```
2.	Commands to run python code for classification and regression tasks.
```
    python3 <path to python file> < # of top features to be considered>
 ```
3. Commands to run external set and classification based on regression.
```
    python3 <path to python file>  <path to result folder>
```

