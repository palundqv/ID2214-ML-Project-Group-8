import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import DescriptionFeaturesSelection as dsf
import DataPreprocessing as dprep


def pca_plot_2features(feature_dataframe):

    pca_result = dsf.apply_pca(feature_dataframe)
    plt.scatter(pca_result['PCA1'], pca_result['PCA2'])
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    training_data_path = 'training_data_207_features.csv'
    training_data = pd.read_csv(training_data_path)
    training_data.drop(columns = ["INDEX","SMILES","ACTIVE"], inplace=True)
    training_data, column_filter = dprep.create_column_filter(training_data)
    training_data, imputation = dprep.create_imputation(training_data)

    

    pca_plot_2features(training_data)