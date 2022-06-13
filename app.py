import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
from sklearn.decomposition import PCA

app = Flask(__name__) #Initialize the flask App

model = pickle.load(open('sk.pkl', 'rb'))
pca = pickle.load(open('kdd.pkl', 'rb'))

@app.route('/')
@app.route('/')
@app.route('/index') 
def index():
	return render_template('index.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
@app.route('/chart') 
def chart():
    df = pd.read_csv(dataset,encoding = 'unicode_escape')
    plt.rcParams["figure.figsize"] = (30,5)

    zfig,(ax1)= plt.subplots(ncols=1, figsize=(25, 7.5), dpi=100)

    fig.suptitle(f'Counts of Observation Labels', fontsize=25)

    sns.countplot(x="labels",palette="OrRd_r",data=train,order=train['labels'].value_counts().index,ax=ax1)
    ax1.set_title('Train Set', fontsize=20)
    ax1.set_xlabel('label', fontsize=15)
    ax1.set_ylabel('count', fontsize=15)
    ax1.tick_params(labelrotation=90)
    return render_template('chart.html')    
@app.route('/abstract') 
def abstract():
	return render_template('abstract.html')    
@app.route('/performance') 
def  performance():
	return render_template('performance.html')   
@app.route('/future') 
def future():
	return render_template('future.html')  
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

@app.route('/home')

def home():
    return render_template('test.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = pca.transform(final_features)    
    prediction = model.predict(final_features)
    output = prediction
    print(output)
    return render_template('test.html', prediction_text= output)
  
    
if __name__ == "__main__":
    app.run(debug=True)
