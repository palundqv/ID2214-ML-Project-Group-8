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

from skopt import BayesSearchCV
import pandas as pd
from sklearn.impute import SimpleImputer

def test_hyperparameter_all_models_bayes_optimization(X, y, test_size=0.3):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Define a list of models to test
    models = [
        # ("Random Forest", RandomForestClassifier()),
        # ("Support Vector Machine", SVC()), # Uncomment when SVC is working
        # ("K-Nearest Neighbors", KNeighborsClassifier()),
        # ("Gaussian Naive Bayes", GaussianNB()),
         ("Multi-layer Perceptron", MLPClassifier(max_iter=1000)),
    ]

    # Results storage
    results = []

    # Hyperparameter search for each model using Bayesian optimization
    for model_name, model in models:
        if model_name == "Random Forest":
            search_space = {
                'n_estimators': (10, 200),
                'max_depth': (10, 30),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 4),
                'bootstrap': [True, False],
            }
        elif model_name == "Support Vector Machine":
            search_space = {
                'C': (0.1, 1),
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
            }
        elif model_name == "K-Nearest Neighbors":
            search_space = {
                'n_neighbors': (3, 10),
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
            }
        elif model_name == "Multi-layer Perceptron":
            search_space = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (10, 10, 10)],
                'activation': ['relu', 'tanh'],
                 'alpha': (1e-4, 1e-2, 'log-uniform'),
    }

        else:
            print(f"Unsupported model: {model_name}")
            continue

        # Bayesian optimization for hyperparameter tuning using training and validation sets
        opt = BayesSearchCV(model, search_space, n_iter=10, cv=5, n_jobs=-1)

        # looking for bugs
        print(f"{model_name} Search Space: {search_space}")
        
        # Train the model on the combined training and validation sets
        opt.fit(X_train, y_train)

        # Get the best model from the search
        best_model = opt.best_estimator_

        # Make predictions on the test set
        y_pred = best_model.predict(X_test)

        # Evaluate the model on the test set and store accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Print the results
        print(f"\n{model_name} Best Parameters: {opt.best_params_}")
        print(f"{model_name} Accuracy: {accuracy:.4f}")

        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
        })

    return results

if __name__ == "__main__":
    print("Running: find_hyperparameters_bayes_optimization")

    x = pd.read_csv("training_with_207_features.csv")
    x.drop(columns="SMILES", inplace=True)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(x)
    training_data = pd.read_csv("training_smiles.csv")
    y = training_data["ACTIVE"].astype("category") # .cat.codes
    # print(pd.DataFrame(X_imputed).head(10).apply(pd.to_numeric()))
    test_hyperparameter_all_models_bayes_optimization(pd.DataFrame(X_imputed).head(1000), y.head(1000))
