import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import imutils
from collections import Counter
from preprocessing import *

def readPreprocessedDataset(path):
  """
  Reads preprocessed images and append it to array reprensents the entire dataset.
  @params path
  @rtype {list}
  """
  images = []
  print(enumerate(glob.glob(path)))
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    images.append(image)
  return images

def meanGray(images, maxGray, minGray):
  """
  Finds mean gray value for the area of interest as a sum of gray pixels between [150-210] 
  and divided by number of pixels.
  @params images: as an array of 2D arrays
  @rtpe {list}: mean values for each image
  """
  grayCount = 0
  imagePixels = 0
  meanGrayValues = []
  for image in images: 
    numpyImage = np.array(image)
    flattenImage = numpyImage.flatten()
    for pixel in flattenImage: 
      if pixel <= minGray and pixel >= maxGray:
          grayCount += 1
    meanGrayValues.append(grayCount/len(flattenImage))
    grayCount = 0
    imagePixels = 0
  return meanGrayValues

def stdGray(images, maxGray, minGray):
  """
  Finds standard deviation of gray values.
  @params images: list of dataset images
  @params maxGray: max gray degree
  @params minGray: min gray degree
  @rtype {list}: std values for each image.
  """
  stdValues = []
  for image in images: 
    numpyImage = np.array(image)
    flattenImage = numpyImage.flatten()
    grayValues = [pixel for pixel in flattenImage if pixel >= maxGray and pixel <= minGray]
    imageDeviation = np.std(grayValues)
    stdValues.append(imageDeviation)
  return stdValues

def modalGray(images, maxGray, minGray):
  """
  Find modal of gray values. 
  @params images: list of dataset images
  @params maxGray: max gray degree
  @params minGray: min gray degree
  @rtype {list}: modal values for each image.
  """
  modalValues = []
  for image in images: 
    numpyImage = np.array(image)
    flattenImage = numpyImage.flatten()
    grayValues = [pixel for pixel in flattenImage if pixel >= maxGray and pixel <= minGray]
    modeValue = Counter(grayValues).most_common(1)
    modalValues.append(modeValue[0][0])
  return modalValues

def aspectRatio(path):
  """
  Find aspect ratio of the dataset.
  @params path
  @rtype {list}: aspect ratio values for each image
  """
  aspectRatios = []
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_COLOR)
    imageDimentions = image.shape
    imageHeight = imageDimentions[0]
    imageWidth = imageDimentions[1]  
    imageAspectRatio = imageWidth/imageHeight
    aspectRatios.append(imageAspectRatio)
  return aspectRatios

def integratedDensity(meanGrayValues, areaValues):
  return [a * b for a, b in zip(meanGrayValues, areaValues)]

def solidity(areaValues , convexHullValues):
  return [a / b for a, b in zip(areaValues, convexHullValues)]

def areaFraction(images):
  """
  Find area fraction of the dataset.
  @params image
  @rtype {list}: area fraction values for each image
  """
  percentages = []
  for image in images:
    numpyImage = np.array(image)
    flattenImage = numpyImage.flatten()
    nonZeroPixels = [pixel for pixel in flattenImage if pixel != 0]
    percentage = len(nonZeroPixels)/len(flattenImage)
    percentages.append(percentage)
  return percentages

def featureExtractionScript():
  maxGray = 150
  minGray = 210 
  images = readPreprocessedDataset('/Users/omar/dania/Tumor-Detection-Logic/Data/CT_cropped/*.jpg')

  meanGrayValues = meanGray(images, maxGray, minGray)
  print ('Done calculating mean gray values.')
  stdGrayValues = stdGray(images, maxGray, minGray)
  print ('Done calculating standard gray values.')
  modalGrayValues = modalGray(images, maxGray, minGray)
  print ('Done calculating modal gray values.')
  images = readPreprocessedDataset('/Users/omar/dania/Tumor-Detection-Logic/Data/CT_cropped/*.jpg')
  circularityValues = getCircularityValues()
  print ('Done calculating circularity values.')
  roundnessValues = getRoundnessValues()
  print ('Done calculating roundness values.')
  areaValues = getAreaValues()
  # convexHullValues = getConvexHullValues()
  # solidityValues = solidity(areaValues ,convexHullValues)
  solidityValues2 = getSolidityValues()
  print ('Done calculating solidity values.')
  densityValues = integratedDensity(meanGrayValues, areaValues)
  print ('Done calculating density values.')
  datasetPath = '../Data/CT/*.jpg'
  aspectRatioValues = aspectRatio(datasetPath)
  print ('Done calculating aspect ratio values.')
  areaFractionValues = areaFraction(images)
  print ('Done calculating area fraction values.')
  return meanGrayValues, stdGrayValues, modalGrayValues, circularityValues, roundnessValues, solidityValues2, densityValues, aspectRatioValues, areaFractionValues
featureExtractionScript()