from flask import Flask,request, jsonify
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import os
from pathlib import Path
# import sklearn
from keras.models import load_model
app = Flask(__name__)
# api=Api(app)

@app.route('/getProbability',methods=['GET'])
def hello():
  content = request.get_json(silent=True)
  age=int(content['age'])
  monthlyIncome=int(content['monthlyIncome'])
  numberOfOpenCreditLinesAndLoans = int(content['numberOfOpenCreditLinesAndLoans'])
  numberOfTimes90DaysLate = int(content['numberOfTimes90DaysLate'])
  inputInArray=np.array([age,monthlyIncome,numberOfOpenCreditLinesAndLoans,numberOfTimes90DaysLate]).reshape(1,-1)
  print(inputInArray)
  print("*************************** request is ",content)
  pred_prob=predict(inputInArray)
  print("=============prob recieved=======   ",pred_prob)
  return jsonify({'probability':pred_prob})


def predict(input2model):
  print("          os.path.firnamr :::::::  ",os.path.dirname(os.path.abspath(__file__)))
  curr_dir = Path(__file__).parent
  print("current directoty is : ",curr_dir)
  file_path = curr_dir.joinpath('final_rfc_credit_Model.pkl')
  # filepath = os.path.join(dirname, 'final_rfc_credit_model.pkl')
  print('filepath: ',file_path)
  # path_rel=os.path.dirname(os.path.realpath(filename))+'/'+filename
  with open(file_path, 'rb') as pickle_file:
    try:
      print('in try')
      loaded_model = pickle.load(pickle_file)
      predicted_val=loaded_model.predict_proba(input2model)
      print("Probability of both class :::::::   ",predicted_val)
      return predicted_val[0][0]
    except pickle.UnpicklingError :
      print('error occured')
      raise
      # return "An error occurred..."

if __name__ == "__main__":
  app.run()
