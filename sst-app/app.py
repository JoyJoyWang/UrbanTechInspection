from flask import Flask, request, url_for, redirect, render_template
import torch
import pandas as pd
app1 = Flask(__name__)

# model = torch.load("")
# this is a decorator that binds the function below to the 
# URL path '/'
# so that when user access this path, the decorated function
# will be called.
@app1.route('/')
def use_template():
    return render_template("index.html")

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     return render_template("result.html",pred="result!!")


if __name__ == '__main__':
    app1.run(host='0.0.0.0',port='80',debug=True)