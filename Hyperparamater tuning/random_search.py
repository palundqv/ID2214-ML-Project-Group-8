# Import necessary libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_hyperparameter_all_models_random_search(X, y, test_size=0.3):
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Define a list of models to test
    models = [
        ("Random Forest", RandomForestClassifier()),
        # ("Support Vector Machine", SVC()), # Can't get this shit to work
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Multi-layer Perceptron", MLPClassifier(max_iter=1000)),
    ]

    # Results storage
    results = []

    # Hyperparameter search for each model
    for model_name, model in models:
        if model_name == "Random Forest":
            param_grid = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
            }
        elif model_name == "Support Vector Machine":
            param_grid = {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
            }
        elif model_name == "K-Nearest Neighbors":
            param_grid = {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
            }
        elif model_name == "Multi-layer Perceptron":
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
            }
        else:
            print(f"Unsupported model: {model_name}")
            continue

        # Randomized search for hyperparameter tuning using training and validation sets
        random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)

        # Train the model on the combined training and validation sets
        random_search.fit(X_train, y_train)

        # Get the best model from the search
        best_model = random_search.best_estimator_

        # Make predictions on the test set
        y_pred = best_model.predict(X_test)

        # Evaluate the model on the test set and store accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Print the results
        print(f"\n{model_name} Best Parameters: {random_search.best_params_}")
        print(f"{model_name} Accuracy: {accuracy:.4f}")

        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
        })
    return results


if __name__ == "__main__":
    print("running: test_hyperparameter_all_models_random_search")
    x = pd.read_csv("training_with_207_features.csv")
    x.drop(columns="SMILES", inplace=True)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(x)
    training_data = pd.read_csv("training_smiles.csv")
    y = training_data["ACTIVE"].astype("category")

    test_hyperparameter_all_models_random_search(pd.DataFrame(X_imputed).head(100), y.head(100))