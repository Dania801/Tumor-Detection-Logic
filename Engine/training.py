import numpy as np
import cv2 as cv
import glob
import pickle
from keras.models import Sequential
from keras.layers import Dense
import tensorflow
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from keras.utils import plot_model
import graphviz
from features import *
from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
from ann_visualizer.visualize import ann_viz;

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
  the rest of layers have 9 neurons. The resulted accuracy with rectifier activation function = 70.59%
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
  scaler = StandardScaler()
  # inputList = scaler.fit_transform(inputList[7][:, np.newaxis])
  # inputList = np.array([inputList.flatten()])
  # print (inputList)
  inputList = scaler.fit_transform(inputList)
  outputList = np.array(outputList)
  model.fit(inputList, outputList, epochs=150, batch_size=16, shuffle=False)
  scores = model.evaluate(inputList, outputList)
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  model_json = model.to_json()
  with open("../Data/nn_architecture.json", "w") as json_file:
      json_file.write(model_json)
  model.save_weights("../Data/nn_weights.h5")
  plot_model(model, to_file='../Data/nn_model.png')
  ann_viz(model, title='Tumor predictor')

def decisionTree():
  """
  Decision tree model. Firstly data got split into training and testing set, then training set is fit into the model
  and in order to calculate the accuracy, we predict the result of training set and provide the result to evaluator.
  The resulted accuracy of decision tree = 44.44%
  """
  dataset = loadTrainingData()
  inputList = []
  outputList = []
  for row in dataset:
    values = [value for value in row.values()]
    outputList.append(values[0])
    del values[0]
    inputList.append(values)
  inputList = np.array(inputList)
  scaler = StandardScaler()
  inputList = scaler.fit_transform(inputList)
  xTrainingList = inputList[0:int(len(inputList)/2):1]
  xTestingList = inputList[int(len(inputList)/2):len(inputList):1]
  outputList = np.array(outputList)
  yTrainingList = outputList[0:int(len(outputList)/2):1]
  yTestingList = outputList[int(len(outputList)/2):len(outputList):1]
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(xTrainingList, yTrainingList)
  dot_data = tree.export_graphviz(clf, out_file=None) 
  graph = graphviz.Source(dot_data) 
  graph.render("iris")
  predictedList = clf.predict(xTestingList)
  accuracy = accuracy_score(yTestingList, predictedList)
  print ('acc: {0}'.format(accuracy))

def naivebayes():
  """
  Accuracy: 42.3%
  """
  data = loadTrainingData()
  data = pd.DataFrame(data)
  X = data[['areaFraction', 'aspectRatio', 'circularity', 'density','meanGray', 'modalGray', 'roundness', 'solidity', 'stdGray']]
  y = data['diagnosis']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
  model = GaussianNB()
  model.fit(X_train, y_train)
  GaussianNB(priors=None)
  y_pred = model.predict(X_test)
  # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
  return model

# loadTrainingData()
# prepareTrainingData()
# saveTrainingData()
# neuralNetwork()
# decisionTree()
# naivebayes()