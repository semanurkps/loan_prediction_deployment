import numpy as np
import pandas as pd
import imblearn
import sklearn
from flask import Flask, request, jsonify, render_template
import pickle
import os

#app name
app = Flask(__name__)

#load the saved model
def load_model():
    return pickle.load(open('loan_pred.pkl', 'rb'))

#home page
@app.route('/')
def home():
    return render_template('index.html')

#predict the result and return it
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    labels = ['Approved', 'Rejected']

    features = [str(x) for x in request.form.values()]

    for i in range(len(features)):
        if i in [0,1,2,3,4,11]:
            pass
        else:
            features[i]=float(features[i])

    values = [np.array(features)]
    
    model = load_model()
    prediction = model.predict(values)

    result = labels[prediction[0]]

    return render_template('index.html', output='Your loan application is: {}'.format(result))


if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
