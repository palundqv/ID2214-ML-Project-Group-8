import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import DescriptionFeaturesSelection as dsf
import DataPreprocessing as dprep
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

if __name__ == "__main__":
    training_data_path = 'training_data_207_features.csv'
    training_data = pd.read_csv(training_data_path)

    #training_data = balance_labels_down(training_data)

    training_data, column_filter = dprep.create_column_filter(training_data)
    training_data, imputation = dprep.create_imputation(training_data)

    training = training_data.drop(columns = ["INDEX","SMILES","ACTIVE"])
    training_lables = training_data["ACTIVE"]

    print(training_lables.value_counts())

    ## Loop here (for classifier in ect ect)
    n_kbest_features_to_test = [3,5,10,30,50,80,100,150,200]
    results = {}
    for number_of_features in n_kbest_features_to_test:
        classifier = GaussianNB()
        n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=number_of_features)
        #print(n_best_features.keys())
        conf_matrix, class_report, auc_roc = test_model_on_folds(training[list(n_best_features.keys())],training_lables,classifier,True,5)
        results[str(number_of_features)] = [auc_roc]
    for key in results.keys():
        print(key, ":", results[key])
