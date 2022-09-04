import pandas as pd

def process(filepath):
    """function takes numoy array of raw_data and returns model prediction"""
    cols = ['full_name', 'name', 'neo', 'pha', 'H', 'G', 'diameter', 'extent', 'albedo', 'rot_per', 'GM', 'BV', 'UB',
            'IR', 'spec_B', 'spec_T', 'H_sigma', 'diameter_sigma', 'epoch', 'e', 'a', 'q', 'i', 'om', 'w', 'ma', 'ad',
            'n', 'tp', 'per', 'per_y', 'moid', 'moid_jup', 'class', 'data_arc', 'condition_code', 'rms']

    df = pd.read_csv(filepath, names=cols)

    # dropping features as per missing value analysis study and EDA study
    to_drop_columns = ['full_name', 'name', 'extent', 'diameter', 'rot_per', 'G', 'GM', 'BV', 'UB', 'IR', 'spec_B',
                       'spec_T', 'H_sigma', 'diameter_sigma', 'per_y', 'w', 'ma', 'om']
    df.drop(to_drop_columns, axis=1, inplace=True)

    return df