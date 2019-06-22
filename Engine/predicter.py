import numpy as np
import cv2 as cv
import warnings
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
import tensorflow
import pandas as pd
from features import *
from preprocessing import *
from training import *


def read_in():
  image = sys.argv[1]
  return image

def loadNNModel():
  with open('../Data/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
  model.load_weights('../Data/model_weights.h5')
  return model

def predictDiagnosis(image, imagePath):
  image = [image]
  maxGray = 150 
  minGray = 210 
  meanGrayValue = meanGray(image, maxGray, minGray)
  stdGrayValue = stdGray(image, maxGray, minGray)
  modalGrayValue = modalGray(image, maxGray, minGray)
  aspectRatioValue = aspectRatio(imagePath)
  areaFractionValue = areaFraction(image)
  [circularityValue, roundnessValue, areadValues, solidityValue] = detectBrain(imagePath)
  densityValue = integratedDensity(meanGrayValue, areadValues)
  imageInfo = {
    'meanGray': meanGrayValue[0],
    'stdGray': stdGrayValue[0],
    'modalGray': modalGrayValue[0],
    'circularity': circularityValue[0],
    'roundness': roundnessValue[0],
    'solidity': solidityValue[0],
    'density': densityValue[0],
    'aspectRatio': aspectRatioValue[0],
    'areaFraction': areaFractionValue[0]
  };
  inputData = pd.DataFrame([imageInfo])
  model = naivebayes()
  result = model.predict(inputData)
  return result[0]

def main():
  warnings.filterwarnings("ignore")
  imagePath = '../Data/CT/12.jpg';
  image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
  result = predictDiagnosis(image, imagePath)
  print (result)
  

if __name__ == '__main__':
  main()
