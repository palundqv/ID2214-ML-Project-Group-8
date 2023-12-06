import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# returns dict with the best features. The keys are the feature names
def select_n_best_features_selectKBest(X, y, n_features=5):
    k_best = SelectKBest(score_func=f_classif, k=n_features)
    k_best.fit(X, y)

    selected_indices = k_best.get_support(indices=True)
    feature_names = X.columns
    feature_scores = k_best.scores_

    selected_features_dict = {}
    for index in selected_indices: # Should this perhaps be returned as an array instead? Would maybe be easier to work with
        feature_name = feature_names[index]
        feature_score = feature_scores[index]
        selected_features_dict[feature_name] = feature_score

    return selected_features_dict

# returns dict with the best features. The keys are the feature names
def select_n_best_features_randomForestImportance(X, y, n_features=5, random_state=None):
    rf_classifier = RandomForestClassifier(random_state=random_state)
    rf_classifier.fit(X, y)
    feature_importances = rf_classifier.feature_importances_
    top_indices = feature_importances.argsort()[-n_features:][::-1]

    feature_names = X.columns
    selected_features_dict = {}
    for index in top_indices:
        feature_name = feature_names[index]
        feature_importance = feature_importances[index]
        selected_features_dict[feature_name] = feature_importance

    return selected_features_dict

def apply_pca(df, n_components=2):
    df_work = df.copy()
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_work)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA{i+1}' for i in range(n_components)])
    return pca_df


if __name__ == "__main__":
    print("You are in main")