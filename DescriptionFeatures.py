import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def extract_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    features = {}
    descriptorCalculator = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    descriptorNames = descriptorCalculator.GetDescriptorNames()
    descriptors = descriptorCalculator.CalcDescriptors(mol)
    for description, desc_Name in zip(descriptors, descriptorNames):
        features[desc_Name] = description
    return features

def convert_smiles_to_features(dataframe_with_smiles_column):
    dataframe = dataframe_with_smiles_column.copy()
    dataframe['features'] = dataframe['SMILES'].apply(extract_molecular_features)
    features_df = pd.DataFrame(dataframe['features'].tolist())
    dataframe = pd.concat([dataframe, features_df], axis=1)
    dataframe = dataframe.drop('features', axis=1)
    return dataframe

def save_dataframe_to_csv(dataframe, dataframeName):
    dataframe.to_csv(dataframeName, index=False)

def load_project_default_csv():
    training_data_path = 'training_smiles.csv'
    test_data_path = 'test_smiles.csv'
    training_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)
    return training_data, test_data

def create_csv_with_207features():
    training_data, test_data = load_project_default_csv()

    training_data = convert_smiles_to_features(training_data)
    test_data = convert_smiles_to_features(test_data) 

    save_dataframe_to_csv(training_data,"training_data_207_features.csv")
    save_dataframe_to_csv(test_data,"test_data_207_features.csv")




if __name__ == "__main__":
    create_csv_with_207features()

    

