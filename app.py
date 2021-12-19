import numpy as np
from flask import Flask, request, jsonify, render_template,send_file
import pickle
import pandas as pd
import json
from io import BytesIO
import csv
import io
from process import estimate
import warnings
from test_preprocess import test_preprocessing
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
app = Flask(__name__)
# this will clena the url data and return as clean DataFrame
def col_reset(df):
    column_name=df.iloc[0]
    col_lis=[]
    for i in column_name:
        col_lis.append(i)
    drop_df=df.drop(index=df.index[0], axis=0,inplace=True)
    my_df=df.set_axis(col_lis, axis=1)
    final_df=my_df.reset_index(drop=True)
    return final_df
@app.route("/training",methods=['POST'])
def training():
    request_data = request.get_json()
    model_df=pd.DataFrame(request_data["data"],columns=request_data["columns"])

    print(model_df)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    # return send_file(BytesIO(model_data),attachment_filename='model',as_attachment=True)
    return jsonify({"predict":"ok"})

@app.route('/testing',methods=['POST'])
def testing():
    # upload_model=request.files['upload_model']
    # main_df=request.files['datasets']
    test_dict={'age':19,'sex':'female','bmi':27.9 ,'children':0,'smoker':'yes','region':'northwest'}
    with open('models/insurance_model', 'rb') as file:
        my_mdl = pickle.load(file)
    main_df=pd.read_csv('datasets/insurance.csv')
    clean_df=main_df.dropna(axis=0)
    merged_df=clean_df.append(test_dict,ignore_index=True)
    test_data=test_preprocessing(merged_df)
    pre_val=my_mdl.predict([test_data])
    # predict_val=my_mdl.predict([[label_col]])
    return jsonify({"predict":pre_val[0]})

if __name__ == "__main__":
    app.run(debug=True)