import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)



model = pickle.load(open('decision_tree.pkl','rb'))
model1 = pickle.load(open('linearregression.pkl','rb'))



@app.route('/')
def check():
    return render_template("check.html")

@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')

@app.route('/new')
def new():
    return render_template("new.html")

  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp1 = float(request.args.get('exp1'))
    exp2 = float(request.args.get('exp2'))

    #prediction = model.predict(input_data)

    #input_pred = input_pred.astype(int)
    Model = (request.args.get('Model'))

    if Model=="Naive Bayes Classifier":
      prediction = model.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="Decision Tree Classifer":
      prediction = model.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))
      
    elif Model=="KNN Classifer":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="SVM Classifer":
      prediction = model.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="Kernel SVM CLassifer":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="RANdom Forest Classifer":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="Linear Regression":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))




    
   


if __name__ == "__main__":
    app.run(debug=True)