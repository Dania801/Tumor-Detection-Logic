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
  image = cv.imread(path, cv.IMREAD_COLOR);
  imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);
  cv.imshow('image',imageGray)
  cv.waitKey(0)
  cv.destroyAllWindows()

def convertDatasetToGray(path):
  """
  Read dataset from a given path, convert it to gray and save it. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = readImage(fileName)
    cv.imwrite('../Data/CT_gray/{0}.jpg'.format(count+1),image)

def histogramEqualization(path):
  """
  Apply histogram equalization on a given dataset and display the result in a stacked format
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    imageEnhanced = cv.equalizeHist(image)
    stackedImages = np.hstack((image,imageEnhanced))
    cv.imshow('image {0}'.format(count+1), stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def Clahe(path):
  """
  Apply Contrast Limited Adaptive Histogram Equalization on a given dataset and display the result in a stacked format
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    appliedClahe = clahe.apply(image)
    stackedImages = np.hstack((image,appliedClahe))
    cv.imshow('image {0}'.format(count+1), stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def smoothDataset(path):
  """
  Apply Gaussian filter for smoothing (noise removal), then apply sharpening filter for highlighting edges
  then save the processed images. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    resizedImage = imutils.resize(image, width=300)
    blurredImage = cv.GaussianBlur(resizedImage, (5,5), cv.BORDER_DEFAULT)
    kernelSharpening = np.array([[-1,-1,-1], 
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
    sharpened = cv.filter2D(resizedImage, -1, kernelSharpening)
    cv.imwrite('../Data/CT_smoothed/{0}.jpg'.format(count+1),blurredImage)
    cv.imwrite('../Data/CT_sharpened/{0}.jpg'.format(count+1), sharpened)
    cv.imshow('image {0}'.format(count+1), blurredImage)
    cv.waitKey(0)
    cv.destroyAllWindows()

def erodeAndDilate(path):
  """
  Erode and dilate the dataset for later shape detection. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE);
    (thresh, imageBW) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #thresh = cv.threshold(resized, 60, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    imageErosion = cv.erode(imageBW, kernel, iterations=1)
    imageDilation = cv.dilate(imageErosion, kernel, iterations=1)
    stackedImages = np.hstack((imageDilation, image))
    cv.imwrite('../Data/CT_raw/{0}.jpg'.format(count+1), imageDilation)
    cv.imshow('image {0}'.format(count+1), stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detectBrain(path):
  """
  Detect brain boundries by detecting all contours of image and selecting the largest contour to be the brain.
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_COLOR);
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    (thresh, imageBW) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(imageBW, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
      largestContour = max(contours, key=cv.contourArea)
      (x,y),radius = cv.minEnclosingCircle(largestContour)
      center = (int(x),int(y))
      radius = int(radius)
      image = cv.circle(image,center,radius,(150,70,50),3)
      cv.imwrite('../Data/CT_detected/{0}.jpg'.format(count+1), image)
      cv.imshow('image',image)
      cv.waitKey(0)
      cv.destroyAllWindows()


def preprocessingScript():
  """
  Apply all of dataset preprocessing function and return the raw images. 
  """
  datasetPath = '../Data/CT/*.jpg'
  grayDatasetPath = '../Data/CT_gray/*.jpg'
  smoothedDataPath = '../Data/CT_smoothed/*.jpg'
  rawDatasetPath = '../Data/CT_raw/*.jpg'
  convertDatasetToGray(datasetPath)
  the following two methods can be used for increasing contrast (we need to choose which one is better)
  histogramEqualization(datasetPath)
  Clahe(datasetPath)
  smoothDataset(datasetPath)
  erodeAndDilate(smoothedDataPath)
  detectBrain(rawDatasetPath)

preprocessingScript()