import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import DescriptionFeaturesSelection as dsf
import DataPreprocessing as dprep
import HyperparamaterTuning.grid_search as hyperGrid
import HyperparamaterTuning.random_search as hyperRandom

from imblearn.over_sampling import SMOTE

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
    #print("Original Label Counts:")
    #print(original_counts)

    minority_label = original_counts.idxmin()
    majority_label = original_counts.idxmax()

    label_difference = original_counts[majority_label] - original_counts[minority_label]

    if label_difference == 0:
        #print("Labels are already balanced.")
        return dataframe

    majority_indices = dataframe[dataframe[label_column] == majority_label].index
    indices_to_remove = np.random.choice(majority_indices, size=label_difference, replace=False)

    balanced_dataframe = dataframe.drop(indices_to_remove)

    new_counts = balanced_dataframe[label_column].value_counts()
    #print("\nNew Label Counts:")
    #print(new_counts)
    return balanced_dataframe

def new_balance_labels_down(X,y):
    df = pd.concat([pd.DataFrame(X), pd.Series(y, name='label')], axis=1)

    original_counts = df['label'].value_counts()
    print("Original Label Counts:")
    print(original_counts)

    minority_label = original_counts.idxmin()
    majority_label = original_counts.idxmax()

    label_difference = original_counts[majority_label] - original_counts[minority_label]

    if label_difference == 0:
        print("Labels are already balanced.")
        return X, y

    majority_indices = df[df['label'] == majority_label].index
    indices_to_remove = np.random.choice(majority_indices, size=label_difference, replace=False)

    balanced_df = df.drop(indices_to_remove)

    new_counts = balanced_df['label'].value_counts()
    print("\nNew Label Counts:")
    print(new_counts)
    return balanced_df.drop('label', axis=1).values, balanced_df['label'].values

def test_for_hyperparameters_grid(training, training_lables):
    n_kbest_features_to_test = [10,30,50,80,100]
    results = {}
    for number_of_features in n_kbest_features_to_test:
        print("number_of_features")
        ## test using KBest
        n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=number_of_features)
        feature_results = hyperGrid.test_hyperparameter_all_models_grid_search(training[list(n_best_features.keys())], training_lables)
        results[str(number_of_features)] = [feature_results]
    
    for key in results.keys():
        print(key, ":", results[key])

def ClassifierWithFeatures(training, training_lables,numberOfFeatures, print_results=False, use_classifier="randomForest"):
    if use_classifier == "randomForest":
        classifier = RandomForestClassifier(bootstrap=True, max_depth=40, min_samples_leaf=2,min_samples_split=2,n_estimators=200,class_weight="balanced",random_state=42)
    elif use_classifier == "MLP":
        classifier = MLPClassifier(activation = 'relu',alpha= 0.0001, hidden_layer_sizes=(50,))
    else:
        print("Invalid classifier specified:")
        return
    n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=numberOfFeatures)
    conf_matrix,class_report,auc_roc = test_model_on_folds(training[list(n_best_features.keys())], training_lables, classifier,False)
    if(print_results):
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        print("\nAUC-ROC Score:", auc_roc)
    return conf_matrix,class_report,auc_roc
    
def manual_crossValidation(training, training_lables, number_of_features = 50, use_classifier="randomForest", sampling="smote"):
    if use_classifier == "randomForest":
        classifier = RandomForestClassifier(bootstrap=True, max_depth=40, min_samples_leaf=2,min_samples_split=2,n_estimators=200,class_weight="balanced",random_state=42)
    elif use_classifier == "MLP":
        classifier = MLPClassifier(activation = 'relu',alpha= 0.0001, hidden_layer_sizes=(50,))
    else:
        print("Invalid classifier specified:")
        return
    
    X = training.copy()
    y = training_lables.copy()

    # Set up cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    auc_scores = []
    confusion_matrices = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        n_best_features = dsf.select_n_best_features_selectKBest(X_train, y_train, n_features=number_of_features) ## should be moved into the fold calculation loop
        X_train = X_train[list(n_best_features.keys())]
        X_test = X_test[list(n_best_features.keys())]


        if sampling=="smote":
            # Apply SMOTE to the training set
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        elif sampling=="under":
            X_train_resampled, y_train_resampled = new_balance_labels_down(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        classifier.fit(X_train_resampled, y_train_resampled)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
        cm = confusion_matrix(y_test, y_pred)

        accuracies.append(accuracy)
        auc_scores.append(auc_score)
        confusion_matrices.append(cm)

    avg_auc_score = np.mean(auc_scores)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    class_report = classification_report(y_test, y_pred)

    return {
        'average_auc_score': avg_auc_score,
        'confusion_matrix': avg_confusion_matrix,
        'classification_report': class_report
    }


def test_random_forest_smote(training_data):
    classifier = RandomForestClassifier(bootstrap=True, max_depth=40, min_samples_leaf=2,min_samples_split=2,n_estimators=200,class_weight="balanced",random_state=42)
    X = training_data.drop(columns = ["INDEX","ACTIVE"])
    y = training_data["ACTIVE"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    n_best_features = dsf.select_n_best_features_selectKBest(X_train_resampled, y_train_resampled, n_features=50)

    classifier.fit(X_train_resampled[list(n_best_features.keys())], y_train_resampled)

    y_pred = classifier.predict(X_test[list(n_best_features.keys())])

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy: ",accuracy)
    print("auc_score: ",auc_score)







    



if __name__ == "__main__":
    training_data_path = 'csvData\\training_merged_fingerprints207.csv'
    training_data = pd.read_csv(training_data_path)


    training_data, column_filter = dprep.create_column_filter(training_data)
    training_data, imputation = dprep.create_imputation(training_data)

    training = training_data.drop(columns = ["INDEX","ACTIVE"])
    training_lables = training_data["ACTIVE"]

    #smote = SMOTE(sampling_strategy='auto', random_state=42)
    #training_resampled, training_lables_resampled = smote.fit_resample(training, training_lables)
    
    ## Testing Random forest with and without smote
    results_oversampled = manual_crossValidation(training, training_lables, 50,use_classifier='randomForest',sampling="smote")
    results_undersampled = manual_crossValidation(training, training_lables, 50,use_classifier='randomForest',sampling="under")

    print("Random-Forest with SMOTE: ", results_oversampled["average_auc_score"])
    print("Random-Forest no SMOTE: ", results_undersampled["average_auc_score"])

    ## Testing MLP with and without smote
    MLP_results_oversampled = manual_crossValidation(training, training_lables, 50,use_classifier='MLP', sampling="smote")
    MLP_results_undersampled = manual_crossValidation(training, training_lables, 50, use_classifier='MLP',sampling="under")
    print("MLP with SMOTE: ", MLP_results_oversampled["average_auc_score"])
    print("MLP no SMOTE: ", MLP_results_undersampled["average_auc_score"])


    #test_for_hyperparameters(training, training_lables)

