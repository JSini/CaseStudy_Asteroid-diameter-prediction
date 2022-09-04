import pandas as pd
import numpy as np
import pickle


def validate_range(df):

    num_cols = [f for f in df.columns if f not in ['neo', 'pha', 'class']]

    with open('min_max_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    # validation for out-of-range numerical values
    for f in num_cols:
        if (df[f].min() < loaded_dict[f][0]) or (df[f].max() > loaded_dict[f][1]):
            error1 = f"feature '{f}' value out_of_range. Currently supported range: {loaded_dict[f][0]} - {loaded_dict[f][1]}"
            return error1

    #validation for out-of-range categorical values
    for f in ['neo', 'pha', 'class']:
        if f in ['neo', 'pha']:
            if np.any(df[f].notnull()):
                if df[f].values.item() not in ['N', 'Y', np.nan]:
                    error1 = f"Warning: feature '{f}' value out_of_range, Currently support only 'N' or 'Y'!!!"
                    return error1
        if f == 'class':
            if np.any(df[f].notnull()):
                if df[f].values.item() not in ['MBA', 'OMB', 'TJN', 'APO', 'MCA', 'IMB', 'AMO', 'ATE', 'CEN', 'TNO',
                                                   'AST'] and df[f].values.item() not in ['HYA', 'PAA', 'IEO']:
                    error3 = f"Warning: Please check the Asteroid Orbit Class specified: https://pdssbn.astro.umd.edu/data_other/objclass.shtml"
                    return error3

                #model doesn't support value range for classes -'HYA', 'PAA', 'IEO' (labelled data available is not in this range)
                #this type of input would filter out in value range validation step
                #if at all feature values falls in model range, then we classify this as 'AST', and make predictions
                elif df[f].values.item() not in ['MBA', 'OMB', 'TJN', 'APO', 'MCA', 'IMB', 'AMO', 'ATE', 'CEN', 'TNO', 'AST']:
                        df[f] = 'AST'

