import numpy as np
from flask import Flask, request, jsonify, render_template,send_file
import pickle
import pandas as pd
import json
import codecs
import re
from io import BytesIO
import csv
import io
import pickle as pkl
from sklearn.externals._packaging.version import parse
from werkzeug.wrappers import response
from process import estimate
import warnings
from test_preprocess import test_preprocessing
from flask_cors import CORS
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
app = Flask(__name__)
CORS(app)
# this will clena the url data and return as clean DataFrame
# def col_reset(df):
#     column_name=df.iloc[0]
#     col_lis=[]
#     for i in column_name:
#         col_lis.append(i)
#     drop_df=df.drop(index=df.index[0], axis=0,inplace=True)
#     my_df=df.set_axis(col_lis, axis=1)
#     final_df=my_df.reset_index(drop=True)
#     return final_df

@app.route("/training",methods=['POST'])
def training():
    request_data = request.get_json()
    model_df=pd.DataFrame(request_data["data"],columns=request_data["columns"])

    select_col=request_data['selected_column']
    target_col=request_data['target_col']
    df=model_df[select_col]
    df.dropna(inplace=True)

    # print(select_col)
    mdl=estimate(df,str(target_col))
    # # print(mdl_df)
    pkl_model=pickle.dumps(mdl)
    # # print(target_col)

    # with open(pkl_model, 'wb') as file:
    #     pickle.dump(response, file)

    response=pkl_model

    # print(model_df)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    # response = send_file(BytesIO(pkl_model),attachment_filename='model',as_attachment=True)

    # response = jsonify({"predict":"ok"})
    # response.headers.add('Access-Control-Allow-Origin', '*')



    # response = jsonify({'hello': 'world'})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/testing',methods=['POST'])
def testing():
    request_data = request.get_json()
    # model_df=pd.DataFrame(request_data["data"],columns=request_data["columns"])
    # test_df=pd.DataFrame(data)
    dfdf=request_data["data"]
    # df.dropna(inplace=True)
    my_mdl=request_data['upload_mdl']
    new_mdl=re.sub('[\0]', '', my_mdl)
    # # cal_c=pickle.loads(new_mdl)
    # with open(new_mdl, 'wb') as file:
    #     w_mdl=pickle.dump(file)
    # # print(new_mdl)
    with open(new_mdl,'rb') as f:
        response = pickle.loads(f)

    # my_new_mdl=pickle.loads(new_mdl)
    # my_new_mdl.predict()
    return response


@app.route('/testroute', methods=['GET'])
def testroute():
    # response  = "hello world"
    response = jsonify({'hello': 'world'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(debug=True)