import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt


def readImage(path):
  """
  Read an image from a given path and display it
  @params path
  """
  img = cv.imread(path, cv.IMREAD_COLOR);
  imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
  cv.imshow('image',imgGray)
  cv.waitKey(0)
  cv.destroyAllWindows()

def convertDatasetToGray(path):
  """
  Read dataset from a given path, convert it to gray and save it. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    img = readImage(fileName)
    cv.imwrite('../Data/CT_gray/{0}.jpg'.format(count+1),img)

def histogramEqualization(path):
  """
  Apply histogram equalization on a given dataset and display the result in a stacked format
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    img = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    imageEnhanced = cv.equalizeHist(img)
    stackedImages = np.hstack((img,imageEnhanced))
    cv.imshow('image', stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def Clahe(path):
  """
  Apply Contrast Limited Adaptive Histogram Equalization on a given dataset and display the result in a stacked format
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    img = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    stackedImages = np.hstack((img,cl1))
    cv.imshow('image', stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preprocessingScript():
  """
  Apply all of dataset preprocessing function and return the raw images. 
  """
  datasetPath = '../Data/CT/*.jpg'
  grayDatasetPath = '../Data/CT_gray/*.jpg'
  convertDatasetToGray(datasetPath)
  # the following two methods can be used for increasing contrast (we need to choose which one is better)
  histogramEqualization(datasetPath)
  Clahe(datasetPath)

preprocessingScript()

