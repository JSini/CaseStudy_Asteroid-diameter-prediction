from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from preprocess import process
from validate import validate_range
from impute import categorical_value_imputer, numerical_value_imputer, knn_imputer

ohe = pickle.load(open('ohe', 'rb'))
model = pickle.load(open('gbdt_model.sav', 'rb'))

app = Flask(__name__)

# Uploader folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    #get the uploaded file
    if request.method == 'POST':
        #getting file
        file = request.files['file']
        if file and allowed_file(file.filename):
            # defining filepath to save
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            # saving file
            file.save(file_path)

            msg = f"Input file uploaded successfully..."
            return render_template('index.html', message=msg)
        else:
            msg = f"Warning: Invalid input file"
            return render_template('index.html', message=msg)


@app.route('/predict', methods=['POST'])
def predict():

    #preprocess input file
    file = os.listdir('static/files/')[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
    #using process script generating dataframe
    df = process(file_path)

    #dataframe generate, now remove file from the directory
    for f in os.listdir('static/files/'):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

    # preprocess
    cols = ['neo', 'pha', 'H', 'albedo', 'epoch', 'e', 'a', 'q', 'i', 'ad', 'n', 'tp', 'per', 'moid', 'moid_jup',
            'class', 'data_arc', 'condition_code', 'rms']

    #validate missing values and out-of-range values
    error = validate_range(df)
    if error:
        return render_template('index.html', error=error)

    #create a copy of dataframe
    data_df = df.copy()

    # data categorical and numerical columns list
    cat_cols = ['neo', 'pha', 'class']
    num_cols = ['H', 'albedo', 'epoch', 'e', 'a', 'q', 'i', 'ad', 'n', 'tp', 'per', 'moid', 'moid_jup', 'data_arc',
                'condition_code', 'rms']

    # convert datatypes
    data_df[num_cols] = data_df[num_cols].astype('float64')

    #missing value imputation
    data_df = numerical_value_imputer(data_df)
    data_df = categorical_value_imputer(data_df)

    # onehotencoding
    data_df_cat_cols = ohe.transform(data_df[cat_cols]).toarray()
    stack_num_cat = np.hstack((data_df[num_cols].values, data_df_cat_cols))
    ccols = list(ohe.get_feature_names_out())
    num_cols.extend(ccols)
    data_encoded_df = pd.DataFrame(stack_num_cat, columns=num_cols)

    # knn-based imputation for 'H' and 'albedo'
    data_imputed_df = knn_imputer(data_encoded_df)


    # predict using the model
    prediction = model.predict(data_imputed_df)
    result = f"Asteroid Diameter(Km) predicted is: {prediction.item()}"

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(debug=True)
