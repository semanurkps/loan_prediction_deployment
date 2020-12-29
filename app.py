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
        if i in [0,1,2,3,4,9,10]:
            pass
        else:
            features[i]=float(features[i])

    x=np.zeros(14)

    x[0]=features[5]                                        #ApplicantIncome
    x[1]=features[6]                                        #CoapplicantIncome
    x[2]=features[7]                                        #LoanAmount
    x[3]=features[8]                                        #Loan_Amount_Term
    x[4]= 1 if features[9]=="All debts paid" else 0         #Credit_History
    x[5]= 1 if features[9]=="Male"else 0                     #Gender_Male
    x[6]= 1 if features[1]=="Yes" else 0                    #Married_Yes
    x[7]= 1 if features[2]== "1"  else 0                     #Dependents_1
    x[8]= 1 if features[2]== "2"  else 0                    #Dependents_2
    x[9]= 1 if features[2]== "3+" else 0                    #Dependents_3+
    x[10]= 1 if features[3]== "No" else 0                   #Education_Not Graduate
    x[11]=1 if features[4]== "Yes" else 0                   #Self_Employed_Yes
    x[12]=1 if features[4]== "Semiurban" else 0             #Property_Area_Semiurban
    x[13]=1 if features[10]== "Urban" else 0                #Property_Area_Urban
    
    #values = [np.array(features)]   
    model = load_model()
    prediction = model.predict([x])
    result = labels[prediction[0]]
    return render_template('index.html', output='Your loan application is: {}'.format(result))

if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
