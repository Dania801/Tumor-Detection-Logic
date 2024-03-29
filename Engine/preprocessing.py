import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import imutils
import math
import ntpath
from diagnosis import *
ntpath.basename("a/b/c")


def readImage(path):
  """
  Read an image from a given path and display it
  @params path
  """
  image = cv.imread(path, cv.IMREAD_COLOR)
  imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  # cv.imshow('image',imageGray)
  # cv.waitKey(0)
  # cv.destroyAllWindows()
  return imageGray

def rotateDatasetLeft(path):
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    image = readImage(fileName)
    rotated = imutils.rotate_bound(image, 330)
    fileName = "../Data/CT_rotatedLeft/rotatedLeft_%d.jpg"%int(name)
    cv.imwrite(fileName, rotated)

def rotateDatasetRight(path):
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    image = readImage(fileName)
    rotated = imutils.rotate_bound(image, 30)
    fileName = "../Data/CT_rotatedRight/rotatedRight_%d.jpg"%int(name)
    cv.imwrite(fileName, rotated)

def flipDatasetVertical(path):
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    image = readImage(fileName)
    rotated = cv.flip(image, 1)
    fileName = "../Data/CT_flippedVertical/flippedVertical_%d.jpg"%int(name)
    cv.imwrite(fileName, rotated)

def rotateFlippedDatasetRight(path):
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    image = readImage(fileName)
    rotated = imutils.rotate_bound(image, 30)
    fileName = "../Data/CT_rotatedFlippedRight/flipped_{0}.jpg".format(name)
    cv.imwrite(fileName, rotated)

def rotateFlippedDatasetLeft(path):
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    image = readImage(fileName)
    rotated = imutils.rotate_bound(image, 330)
    fileName = "../Data/CT_rotatedFlippedLeft/flipped_{0}.jpg".format(name)
    cv.imwrite(fileName, rotated)

def gatherDataset(paths):
  index = 0
  for path in paths:
    for count, fileName in enumerate(glob.glob(path)):
      head, tail = ntpath.split(fileName)
      name = tail.replace('.jpg', '')
      image = readImage(fileName)
      cv.imwrite("../Data/Dataset/{0}.jpg".format(index), image)
      index += 1

def convertDatasetToGray(path):
  """
  Read dataset from a given path, convert it to gray and save it. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    image = readImage(fileName)
    cv.imwrite('../Data/CT_gray/{0}.jpg'.format(name),image)

def histogramEqualization(path):
  """
  Apply histogram equalization on a given dataset and display the result in a stacked format
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    imageEnhanced = cv.equalizeHist(image)
    stackedImages = np.hstack((image,imageEnhanced))
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    # # cv.imwrite('../Data/CT_enhanced/{0}.jpg'.format(name), imageEnhanced)
    # cv.imshow('image {0}'.format(name), stackedImages)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def Clahe(path):
  """
  Apply Contrast Limited Adaptive Histogram Equalization on a given dataset and display the result in a stacked format
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    appliedClahe = clahe.apply(image)
    stackedImages = np.hstack((image,appliedClahe))
    # cv.imshow('image {0}'.format(count+1), stackedImages)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def smoothDataset(path):
  """
  Apply Gaussian filter for smoothing (noise removal), then apply sharpening filter for highlighting edges
  then save the processed images. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    resizedImage = imutils.resize(image, width=300)
    blurredImage = cv.GaussianBlur(resizedImage, (5,5), cv.BORDER_DEFAULT)
    kernelSharpening = np.array([[-1,-1,-1], 
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
    sharpened = cv.filter2D(resizedImage, -1, kernelSharpening)
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    # cv.imwrite('../Data/CT_smoothed/{0}.jpg'.format(name),blurredImage)
    # cv.imwrite('../Data/CT_sharpened/{0}.jpg'.format(name), sharpened)
    stackedImages = np.hstack((resizedImage,sharpened))
    cv.imshow('image {0}'.format(name), stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def erodeAndDilate(path):
  """
  Erode and dilate the dataset for later shape detection. 
  @params path
  """
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    (thresh, imageBW) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #thresh = cv.threshold(resized, 60, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    imageErosion = cv.erode(imageBW, kernel, iterations=1)
    imageDilation = cv.dilate(imageErosion, kernel, iterations=1)
    stackedImages = np.hstack((image, imageErosion))
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    # cv.imwrite('../Data/CT_raw/{0}.jpg'.format(name), imageDilation)
    cv.imshow('image {0}'.format(name), stackedImages)
    cv.waitKey(0)
    cv.destroyAllWindows()

def cropImage(image, contour, index):
  """
  Crops an image according to a defined contour, then store the image and return it. 
  @params image
  @params contour: eliptic shaped figure that represents the brain
  @params index: image index in the dataset
  @rtype image  
  """
  xRect, yRect, wRect, hRect = cv.boundingRect(contour)
  croppedImage = image[yRect: yRect+hRect,
               xRect: xRect+wRect]
  # cv.imwrite('../Data/CT_cropped/{0}.jpg'.format(index), croppedImage)
  # cv.imshow('image {0}'.format(index+1),croppedImage)
  # cv.waitKey(0)
  # cv.destroyAllWindows()
  return croppedImage 

def calculateCircularity(contour):
  """
  Takes a contour (brain shaped) and find its circularity value
  @params contour
  @rtype {double}: a value between 0 and 1
  """
  area = cv.contourArea(contour)
  perimeter = cv.arcLength(contour, True)
  circularity = area/perimeter**2
  return circularity

def calculateRoundness(contour):
  """
  Takes a contour (brain shaped) and find its roundness value
  @params contour
  @rtype {double}: a value between 0 and 1
  """
  area = cv.contourArea(contour)
  perimeter = cv.arcLength(contour, True)
  roundness = 4*math.pi*area/(perimeter**2)
  return roundness

def calculateSolidity(contour):
  area = cv.contourArea(contour)
  hull = cv.convexHull(contour)
  hullArea = cv.contourArea(hull)
  solidity = float(area)/hullArea
  return solidity

def detectBrain(path):
  """
  Detect brain boundries by detecting all contours of image and selecting the largest contour to be the brain.
  This function also find stack circularity values of each detected brain.
  @params path
  """
  circularityValues = []
  roundnessValues = []
  area = []
  convexHullValues=[]
  solidity=[]
  solidityValues=[]
  for count, fileName in enumerate(glob.glob(path)):
    head, tail = ntpath.split(fileName)
    name = tail.replace('.jpg', '')
    actualImage = cv.imread('../Data/CT_sharpened/{0}.jpg'.format(name), cv.IMREAD_COLOR)
    # actualImage = imutils.resize(actualImage, width=200)
    image = cv.imread(fileName, cv.IMREAD_COLOR);
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    (thresh, imageBW) = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(imageBW, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
      largestContour = max(contours, key=cv.contourArea)
      circularity = calculateCircularity(largestContour)
      area.append(getArea(largestContour))
      convexHullValues.append(getConvexHull(largestContour))
      circularityValues.append(circularity)
      roundness = calculateRoundness(largestContour)
      roundnessValues.append(roundness)
      solidity = calculateSolidity(largestContour)
      solidityValues.append(solidity)
      (x,y),radius = cv.minEnclosingCircle(largestContour)
      center = (int(x),int(y))
      radius = int(radius)
      croppedImage = cropImage(image, largestContour, name)
      croppedImageSmoothed = cv.GaussianBlur(croppedImage, (5,5), cv.BORDER_DEFAULT)
      head, tail = ntpath.split(fileName)
      name = tail.replace('.jpg', '')
      # cv.imwrite('../Data/CT_cropped/{0}.jpg'.format(name), croppedImageSmoothed)
      imageCircled = cv.circle(image,center,radius,(150,70,50),3)
      # stackedImages = np.hstack((croppedImage, imageCircled))
      # cv.imwrite('../Data/CT_detected/{0}.jpg'.format(name), imageCircled)
      # cv.imshow('image {0}'.format(name), imageCircled)
      # cv.waitKey(0)
      # cv.destroyAllWindows()
  return circularityValues, roundnessValues, area, solidityValues


def preprocessingScript():
  """
  Apply all of dataset preprocessing function and return the raw images. 
  """
  datasetPath = '../Data/Dataset/*.jpg'
  grayDatasetPath = '../Data/CT_gray/*.jpg'
  enhancedDatasetPath = '../Data/CT_enhanced/*.jpg'
  smoothedDataPath = '../Data/CT_smoothed/*.jpg'
  rawDatasetPath = '../Data/CT_raw/*.jpg'
  convertDatasetToGray(datasetPath)
  # the following two methods can be used for increasing contrast (we need to choose which one is better)
  histogramEqualization(grayDatasetPath)
  Clahe(datasetPath)
  smoothDataset(enhancedDatasetPath)
  erodeAndDilate(smoothedDataPath)
  detectBrain(grayDatasetPath)

def getCircularityValues():
  [circularityValues, roundness, area, convexHullValues] = detectBrain('../Data/CT_gray/*.jpg')
  return circularityValues

def getRoundnessValues():
  [circularity, roundnessValues, area, convexHullValues] = detectBrain('../Data/CT_gray/*.jpg')
  return roundnessValues

def getArea(contour):
  return cv.contourArea(contour)

def getAreaValues():
  [circularity, roundnessValues, areaValues, solidityValues] = detectBrain('../Data/CT_gray/*.jpg')
  return areaValues

# def getConvexHullValues():
#   [circularity, roundnessValues ,areaValues, convexHullValues] = detectBrain('../Data/CT_gray/*.jpg')
#   return convexHullValues

def getSolidityValues():
  [circularity, roundnessValues, areadValues, solidityValues] = detectBrain('../Data/CT_gray/*.jpg')
  return solidityValues

def getConvexHull(contour):
  return cv.convexHull(contour ,False)

def getDiagnosis():
  return diagnosisList

# getRoundnessValues()
# preprocessingScript()
# rotateFlippedDatasetLeft('../Data/CT_rotatedLeft/*.jpg')
# gatherDataset(['../Data/CT/*.jpg', '../Data/CT_rotatedRight/*.jpg', '../Data/CT_rotatedLeft/*.jpg',
#                '../Data/CT_flippedVertical/*.jpg', '../Data/CT_rotatedFlippedLeft/*.jpg', 
#                '../Data/CT_rotatedFlippedRight/*.jpg'])