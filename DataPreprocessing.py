import numpy as np
import pandas as pd
import sklearn



def create_column_filter(dataframe):
    workDF = dataframe.copy()
    columns_to_drop = []
    for columnName, columnData in workDF.items():
        if columnName != "SMILES" and columnName != "ACTIVE" and columnName != "INDEX":
            if columnData.dropna().nunique() < 2:
                columns_to_drop.append(columnName)
    workDF.drop(columns=columns_to_drop,inplace=True)
    return workDF, workDF.columns.tolist()

def apply_column_filter(dataframe, column_filter):
    workDF = dataframe.copy()
    workDF.drop(columns=[col for col in workDF.columns if col not in column_filter],inplace=True)
    return workDF


def create_imputation(dataframe):
    workDF = dataframe.copy()
    imputation = {}
    # int and float
    for columnName in workDF.select_dtypes(include=['int64', 'float64']).columns:
        if columnName != "SMILES" and columnName != "ACTIVE" and columnName != "INDEX":
            ## start to check empty case
            if workDF[columnName].isnull().all():
                workDF[columnName].fillna(0,inplace=True)
            else:
                mean = workDF[columnName].mean()
                workDF[columnName].fillna(mean,inplace=True)
                imputation[columnName] =  mean

    # object and series
    for columnName in workDF.select_dtypes(include=['object', 'category']).columns:
        if columnName != "SMILES" and columnName != "ACTIVE" and columnName != "INDEX":
            if workDF[columnName].isnull().all():
                dtype = workDF[columnName].dtype
                if dtype == 'object':
                    workDF[columnName].fillna("",inplace=True)
                    imputation[columnName] = ""
                elif dtype == 'category':  
                    cat = workDF[columnName].cat.categories[0]                 
                    workDF[columnName].fillna(cat,inplace=True)
                    imputation[columnName] = cat 
            else:
                mostCommon = workDF[columnName].mode()[0]
                workDF[columnName].fillna(mostCommon, inplace=True)
                imputation[columnName] = mostCommon

    return workDF, imputation


def apply_imputation(dataframe, imputation):
    workDF = dataframe.copy()
    for columnName, impVal in imputation.items():
        if columnName in workDF.columns:
            workDF[columnName].fillna(impVal, inplace=True)
    return workDF