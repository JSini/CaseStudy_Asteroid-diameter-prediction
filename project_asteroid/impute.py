import numpy as np
import pandas as pd
import pickle

def categorical_value_imputer(df):
    # imputation for class feature
    # taking feature 'semi-major axis - a', 'perihelion distance - q', and 'aphelion distance -ad' forimputing the orbit class
    if np.any(df['class'].isnull().values):
        if (df['a'].values.item() > 1.0) and (df['q'].values.item() > 1.017 and df['q'].values.item() < 1.3):
            df['class'] = 'AMO'
        elif df['a'].values.item() > 1.0 and df['q'].values.item() < 1.017:
            df['class'] = 'APO'
        elif df['a'].values.item() < 1.0 and df['ad'].values.item() > 0.983:
            df['class'] = 'ATE'
        elif df['a'].values.item() > 5.5 and df['a'].values.item() < 30.1:
            df['class'] = 'CEN'
        elif df['a'].values.item() < 2.0 and df['q'].values.item() > 1.666:
            df['class'] = 'IMB'
        elif (df['a'].values.item() > 2.0 and df['a'].values.item() < 3.2) and df['q'].values.item() > 1.666:
            df['class'] = 'MBA'
        elif df['a'].values.item() < 3.2 and (df['q'].values.item() > 1.3 and df['q'].values.item() < 1.666):
            df['class'] = 'MCA'
        elif df['a'].values.item() > 3.2 and df['a'].values.item() < 4.6:
            df['class'] = 'OMB'
        elif (df['a'].values.item() > 4.6 and df['a'].values.item() < 5.5) and df['e'].values.item() < 0.3:
            df['class'] = 'TJN'
        elif df['a'].values.item() > 30.1:
            df['class'] = 'TNO'
        else:
            df['class'] = 'AST'

    # imputation for 'neo' feature based on orbit class
    if np.any(df['neo'].isnull().values):
        if df['class'].values.item() in ['AST', 'CEN', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO']:
            df['neo'] = 'N'
        elif df['class'].values.item() in ['AMO', 'APO', 'ATE']:
            df['neo'] = 'Y'
        # imputing with mode values 'N'
        else:
            df['neo'] = 'N'

    # imputation for 'pha' feature based on orbit class
    # asteroids with minimum orbit intersection distance (MOID) of 0.05 au or less
    # and an absolute magnitude (H) of 22.0 or less are considered PHAs.
    if np.any(df['pha'].isnull().values):
        if df['moid'].values.item() <= 0.05 and df['H'].values.item()<=22:
            df['pha'] = 'Y'
        else:
            df['pha'] = 'N'

    return df

def numerical_value_imputer(df):
    '''function to impute missingvalues in numerical column
    parameter: dataframe one hot encoded for categorical features
    returns: dataframe impute for  missing numerical feature value'''

    #median based imputation for any numerical feature except 'H' and 'albedo'
    with open('median_values.pkl', 'rb') as f:
        median_values = pickle.load(f)

    num_cols = ['H', 'albedo', 'epoch', 'e', 'a', 'q', 'i', 'ad', 'n', 'tp', 'per', 'moid', 'moid_jup', 'data_arc',
                'condition_code', 'rms']

    med_num_cols = [col for col in num_cols if col not in ['H', 'albedo']]
    for col in med_num_cols:
            if np.any(df[col].isnull().values):
                    df[col]=median_values[col]
    return df

def knn_imputer(df):
    # KNN imputation for 'H' and 'albedo'
    num_cols = ['H', 'albedo', 'epoch', 'e', 'a', 'q', 'i', 'ad', 'n', 'tp', 'per', 'moid', 'moid_jup', 'data_arc',
                'condition_code', 'rms']

    ## scaling the data
    with open('scaler', 'rb') as f:
        scaler = pickle.load(f)
    df.loc[:, num_cols] = scaler.transform(df.loc[:, num_cols])

    ##getting knn imputer
    with open('knn_imp', 'rb') as f:
        knn_imp = pickle.load(f)
    knn_imputed = knn_imp.transform(df) #returns numpy array
    #converting to dataframe
    df_imputed = pd.DataFrame(knn_imputed, columns=df.columns)

    #unscale result
    df_imputed.loc[:, num_cols] = scaler.inverse_transform(df_imputed.loc[:, num_cols])
    return df_imputed