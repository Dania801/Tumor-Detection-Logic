import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import imutils


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

def smoothDataset(path):
  """
  Apply Gaussian filter for smoothing (noise removal), then apply sharpening filter for highlighting edges
  then save the processed images. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    img = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    resized = imutils.resize(img, width=300)
    ratio = img.shape[0] / float(resized.shape[0])
    blurred = cv.GaussianBlur(resized, (5, 5), 0)
    kernelSharpening = np.array([[-1,-1,-1], 
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
    sharpened = cv.filter2D(img, -1, kernelSharpening)
    cv.imwrite('../Data/CT_smoothed/{0}.jpg'.format(count+1),sharpened)
    cv.imshow('img', sharpened)
    cv.waitKey(0)
    cv.destroyAllWindows()

def erodeAndDilate(path):
  """
  Erode and dilate the dataset for later shape detection. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    img = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    resized = imutils.resize(img, width=300)
    ratio = img.shape[0] / float(resized.shape[0])
    blurred = cv.GaussianBlur(resized, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    imgErosion = cv.erode(thresh, kernel, iterations=1)
    imgDilation = cv.dilate(imgErosion, kernel, iterations=1)
    cv.imshow('img', thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preprocessingScript():
  """
  Apply all of dataset preprocessing function and return the raw images. 
  """
  datasetPath = '../Data/CT/*.jpg'
  grayDatasetPath = '../Data/CT_gray/*.jpg'
  smoothedDataPath = '../Data/CT_smoothed/*.jpg'
  #convertDatasetToGray(datasetPath)
  # the following two methods can be used for increasing contrast (we need to choose which one is better)
  #histogramEqualization(datasetPath)
  #Clahe(datasetPath)
  #smoothDataset(datasetPath)
  erodeAndDilate(smoothedDataPath)

preprocessingScript()