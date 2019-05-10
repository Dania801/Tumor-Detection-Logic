import numpy as np
import cv2 as cv
import glob
import pickle
from keras.models import Sequential
from keras.layers import Dense
import tensorflow
from features import *
from preprocessing import *

np.random.seed(7)

def prepareTrainingData():
  """
  Retrieves all extracted features and organize it in dictionary.
  @rtype {List}: list of features dictionary
  """
  [meanGrayValues, stdGrayValues, modalGrayValues, 
   circularityValues, roundnessValues, solidityValues, 
   densityValues, aspectRatioValues, areaFractionValues] = featureExtractionScript()
  diagnosisList = getDiagnosis()
  imagesInfo = []
  for i, info in enumerate(meanGrayValues):
    imagesInfo.append({
      'diagnosis': diagnosisList[i],
      'meanGray': meanGrayValues[i],
      'stdGray': stdGrayValues[i],
      'modalGray': modalGrayValues[i],
      'circularity': circularityValues[i],
      'roundness': roundnessValues[i],
      'solidity': solidityValues[i],
      'density': densityValues[i],
      'aspectRatio': aspectRatioValues[i],
      'areaFraction': areaFractionValues[i]
    });
  return imagesInfo

def saveTrainingData():
  """
  Stores features in a binary file.
  """
  imagesInfo = prepareTrainingData()
  dataFile = '../Data/dataset_info.dat'
  openDataFile = open(dataFile, 'wb')
  pickle.dump(imagesInfo, openDataFile)
  openDataFile.close()

def loadTrainingData():
  """
  Retrieves dataset features from desk.
  """
  dataFile = '../Data/dataset_info.dat'
  openedSumFile = open(dataFile, 'rb')
  data = pickle.load(openedSumFile)
  return data

def neuralNetwork():
  """
  Neural network with 5 layers, the first layer has 12 neurons and expects 9 inputs
  the rest of layers have 9 neurons. The resulted accuracy with rectifier activation function = 68.75%
  while with sigmoid function = 58.30%
  """
  model = Sequential()
  model.add(Dense(12, input_dim=9, activation='relu'))
  model.add(Dense(9, activation='relu'))
  model.add(Dense(9, activation='relu'))
  model.add(Dense(9, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  dataset = loadTrainingData()
  inputList = []
  outputList = []
  for row in dataset:
    values = [value for value in row.values()]
    outputList.append(values[0])
    del values[0]
    inputList.append(values)
  inputList = np.array(inputList)
  outputList = np.array(outputList)
  model.fit(inputList, outputList, epochs=150, batch_size=10)
  scores = model.evaluate(inputList, outputList)
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# loadTrainingData()
# prepareTrainingData()
# saveTrainingData()
neuralNetwork()