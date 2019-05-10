import numpy as np
import cv2 as cv
import glob
import pickle
from features import *
from preprocessing import *


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
  print (data)
  return data

# loadTrainingData()
# prepareTrainingData()
saveTrainingData()