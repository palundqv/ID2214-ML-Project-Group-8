import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import DescriptionFeaturesSelection as dsf
import DataPreprocessing as dprep
import HyperparamaterTuning.grid_search as hyper
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB

def test_model_on_folds(X, y, classifier, print_results = False, number_of_folds = 5):
    # Perform n-fold cross-validation 
    y_proba = cross_val_predict(classifier, X, y, cv=number_of_folds, method='predict_proba')
    y_pred = np.argmax(y_proba, axis=1)
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)
    auc_roc = roc_auc_score(y, y_proba[:, 1])
    if print_results:
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        print("\nAUC-ROC Score:", auc_roc)

    return conf_matrix,class_report,auc_roc

def balance_labels_down(dataframe, label_column="ACTIVE"):
    original_counts = dataframe[label_column].value_counts()
    print("Original Label Counts:")
    print(original_counts)

    minority_label = original_counts.idxmin()
    majority_label = original_counts.idxmax()

    label_difference = original_counts[majority_label] - original_counts[minority_label]

    if label_difference == 0:
        print("Labels are already balanced.")
        return dataframe

    majority_indices = dataframe[dataframe[label_column] == majority_label].index
    indices_to_remove = np.random.choice(majority_indices, size=label_difference, replace=False)

    balanced_dataframe = dataframe.drop(indices_to_remove)

    new_counts = balanced_dataframe[label_column].value_counts()
    print("\nNew Label Counts:")
    print(new_counts)

    return balanced_dataframe

def test_for_hyperparameters(training, training_lables):
    n_kbest_features_to_test = [10,30,50,80,100]
    results = {}
    for number_of_features in n_kbest_features_to_test:
        print("number_of_features")
        ## test using KBest
        n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=number_of_features)
        feature_results = hyper.test_hyperparameter_all_models_grid_search(training[list(n_best_features.keys())], training_lables)
        results[str(number_of_features)] = [feature_results]
    
    for key in results.keys():
        print(key, ":", results[key])

def RandomForestWithFeatures(training, training_lables,numberOfFeatures):
    classifier = RandomForestClassifier(bootstrap=True, max_depth=40, min_samples_leaf=2,min_samples_split=2,n_estimators=200,class_weight="balanced")
    n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=numberOfFeatures)
    conf_matrix,class_report,auc_roc = test_model_on_folds(training[list(n_best_features.keys())], training_lables, classifier,False)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print("\nAUC-ROC Score:", auc_roc)
    




    



if __name__ == "__main__":
    training_data_path = 'csvData\\training_data_207_features.csv'
    training_data = pd.read_csv(training_data_path)

    training_data = balance_labels_down(training_data)

    training_data, column_filter = dprep.create_column_filter(training_data)
    training_data, imputation = dprep.create_imputation(training_data)

    training = training_data.drop(columns = ["INDEX","SMILES","ACTIVE"])
    training_lables = training_data["ACTIVE"]

    #RandomForestWithFeatures(training, training_lables, 50)
    test_for_hyperparameters(training, training_lables)
    '''
    ## Loop here (for classifier in ect ect)
    n_kbest_features_to_test = [5,10,30,50,80,100,150,200]
    results = {}
    for number_of_features in n_kbest_features_to_test:
        classifier = GaussianNB()
        n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=number_of_features)
        #print(n_best_features.keys())
        conf_matrix, class_report, auc_roc = test_model_on_folds(training[list(n_best_features.keys())],training_lables,classifier,False,5)
        results[str(number_of_features)] = [auc_roc]
    for key in results.keys():
        print(key, ":", results[key])
    '''
