from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import DataPreprocessing as dprep
import DescriptionFeaturesSelection as dsf
from imblearn.over_sampling import SMOTE

import pandas as pd

# returns a dict with models names and their ROC-AUC score, change models to test by uncomment the model in the models list.
def test_hyperparameter_all_models_random_search(X, y, test_size=0.2, use_smote=True):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    if use_smote:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Define a list of models to test
    models = [
        ("Random Forest", RandomForestClassifier()),
        #("Support Vector Machine", SVC()), # Says it only works with grid search?
        #("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-layer Perceptron", MLPClassifier(max_iter=1000)),
    ]

    # Results storage
    results = []

    # Hyperparameter search for each model
    for model_name, model in models:
        if model_name == "Random Forest":
            param_grid = {
                'n_estimators': [10, 50, 100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'bootstrap': [True, False],
                'max_features': [None, 'sqrt', 'log2'],
            }
        elif model_name == "Support Vector Machine":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
            }
        elif model_name == "K-Nearest Neighbors":
            param_grid = {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 50],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
            }
        elif model_name == "Multi-layer Perceptron":
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant','adaptive'],
            }
        else:
            print(f"Unsupported model: {model_name}")
            continue

        # Randomized search for hyperparameter tuning using training and validation sets
        random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1, scoring='roc_auc', verbose=3)

        # Train the model on the combined training and validation sets
        random_search.fit(X_train_resampled, y_train_resampled)

        # Get the best model from the search
        best_model = random_search.best_estimator_

        # Make predictions on the test set
        y_scores = best_model.predict_proba(X_test)[:, 1]  # Probability estimates for positive class

        # Calculate ROC-AUC
        roc_auc = roc_auc_score(y_test, y_scores)

        # Print the parameter grid and results
        print("----------------------------------------------------------------------------------------------------------------------")
        print(f"{model_name} Best Parameters: {random_search.best_params_}")
        print(f"{model_name} ROC-AUC: {roc_auc:.4f}")

        # Store results
        results.append({
            'Model': model_name,
            'ROC-AUC': roc_auc,
        })
    return results

if __name__ == "__main__":
    print("running: test_hyperparameter_all_models_random_search")
    x = pd.read_csv('..csvData\\training_merged_fingerprints207.csv')

    training_data, column_filter = dprep.create_column_filter(x)
    training_data, imputation = dprep.create_imputation(training_data)

    training = training_data.drop(columns = ["INDEX","ACTIVE"])
    training_lables = training_data["ACTIVE"]

    n_best_features = dsf.select_n_best_features_selectKBest(training, training_lables, n_features=50)

    test_hyperparameter_all_models_random_search(training[list(n_best_features.keys())], training_lables)
