from flask import Flask, jsonify, request
import pandas as pd
import joblib

import flask
app = Flask(__name__)



@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('finalized_model.sav') 
    final_dic = request.form.to_dict()
    for key,value in final_dic.items():
        final_dic[key]=float(value)
    x1,x2,x3,x4,x5,x6,x7,x8,x9 = final_dic['fare_class'],final_dic['passenger_count'],final_dic['Year'],final_dic['Month'],final_dic['Day'],final_dic['Hours'],final_dic['Minutes'],final_dic['mornight'],final_dic['Total distance']
    input_Data=pd.DataFrame([[x1,x2,x3,x4,x5,x6,x7,x8,x9]],columns=['fare_class', 'passenger_count', 'Year', 'Month', 'Day', 'Hours', 'Minutes', 'mornight', 'Total distance'])
    pred = clf.predict(input_Data)
    final_ans="Your Taxi Fare will be $"+str(pred[0])
    return flask.render_template('result.html',fare=final_ans)


if __name__ == '__main__':
    app.run(host='localhost', port=80,debug=True)

