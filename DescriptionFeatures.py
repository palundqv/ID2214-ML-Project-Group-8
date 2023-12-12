import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
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

def extract_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    features = {}
    # Try nBits 2048, 1024, 512, 256
    # Morgan Fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    for i in range(512):
        features[f'fp_{i}'] = morgan_fp[i]

    return features

def convert_smiles_to_features(dataframe_with_smiles_column):
    dataframe = dataframe_with_smiles_column.copy()
    dataframe['features'] = dataframe['SMILES'].apply(extract_molecular_features)
    features_df = pd.DataFrame(dataframe['features'].tolist())
    dataframe = pd.concat([dataframe, features_df], axis=1)
    dataframe = dataframe.drop('features', axis=1)
    return dataframe

def convert_smiles_to_fingerprints(dataframe_with_smiles_column):
    dataframe = dataframe_with_smiles_column.copy()
    dataframe['features'] = dataframe['SMILES'].apply(extract_fingerprints)
    features_df = pd.DataFrame(dataframe['features'].tolist())
    dataframe = pd.concat([dataframe, features_df], axis=1)
    dataframe = dataframe.drop('features', axis=1)
    return dataframe

def save_dataframe_to_csv(dataframe, dataframeName):
    dataframe.to_csv(dataframeName, index=False)

def load_project_default_csv():
    training_data_path = 'csvData\\training_smiles.csv'
    test_data_path = 'csvData\\test_smiles.csv'
    training_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)
    return training_data, test_data

def create_csv_with_207features():
    training_data, test_data = load_project_default_csv()
    
    training_data = convert_smiles_to_features(training_data)
    test_data = convert_smiles_to_features(test_data) 

    save_dataframe_to_csv(training_data,"csvData\\training_data_207_features.csv")
    save_dataframe_to_csv(test_data,"csvData\\test_data_207_features.csv")

def create_csv_with_fingerprint_features():
    training_data, test_data = load_project_default_csv()

    training_data = convert_smiles_to_fingerprints(training_data)
    test_data = convert_smiles_to_fingerprints(test_data) 

    save_dataframe_to_csv(training_data,"csvData\\training_data_fingerprints.csv")
    save_dataframe_to_csv(test_data,"csvData\\test_data_fingerprints.csv")

def create_csv_with_fingerprints_and_207Features():
    training_data_finger_path = 'csvData\\training_data_fingerprints.csv'
    test_data_finger_path = 'csvData\\test_data_fingerprints.csv'
    training_data_207_path = 'csvData\\training_data_207_features.csv'
    test_data_207_path = 'csvData\\test_data_207_features.csv'

    training_data_finger = pd.read_csv(training_data_finger_path)
    test_data_finger = pd.read_csv(test_data_finger_path)
    training_data_207 = pd.read_csv(training_data_207_path)
    test_data_207 = pd.read_csv(test_data_207_path)

    merged_training_finger207_df = pd.merge(training_data_finger, training_data_207, on='INDEX', how='inner')
    merged_test_finger207_df = pd.merge(test_data_finger, test_data_207, on='INDEX', how='inner')

    merged_training_finger207_df.drop(['SMILES_x','SMILES_y','ACTIVE_y'],inplace=True, axis=1)
    merged_training_finger207_df.rename(columns={'ACTIVE_x': 'ACTIVE'}, inplace=True)
    merged_test_finger207_df.drop(['SMILES_x','SMILES_y'],inplace=True,axis=1)

    save_dataframe_to_csv(merged_training_finger207_df,"csvData\\training_merged_fingerprints207.csv")
    save_dataframe_to_csv(merged_test_finger207_df,"csvData\\test_merged_fingerprints207.csv")




if __name__ == "__main__":
    create_csv_with_fingerprints_and_207Features()
    



    

