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

    print(image)
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
  for count, image in enumerate(images):
    for line in image: 
      imagePixels += len(line)
      for pixel in line: 
        if pixel <= minGray and pixel >= maxGray:
          grayCount += 1
    meanGrayValues.append(grayCount/imagePixels)
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
  return [a * b for a, b in zip(meanGrayValue, areaValues)]


def featureExtractionScript():
  maxGray = 150 
  minGray = 210 
  # images = readPreprocessedDataset('/Users/omar/Tumor-Detection-Logic/Data/CT_cropped/*.jpg')
  # print(images)
  # meanGrayValues = meanGray(images, maxGray, minGray)
  # stdGrayValues = stdGray(images, maxGray, minGray)
  # modalGrayValues = modalGray(images, maxGray, minGray)
  images = readPreprocessedDataset('../Data/CT_cropped/*.jpg')
  # cv.contourArea()
  meanGrayValues = meanGray(images, maxGray, minGray)
  stdGrayValues = stdGray(images, maxGray, minGray)
  modalGrayValues = modalGray(images, maxGray, minGray)
  circularityValues = getCircularityValues()
  roundnessValues = getRoundnessValues()
  areaValues = getAreaValues()
  print(integratedDensity(meanGrayValues, areaValues))
  # print(areaValues)
  datasetPath = '../Data/CT/*.jpg'
  aspectRatioValues = aspectRatio(datasetPath)

# >>>>>>> 4142001690f3983927ae7cc6f1cbca1ad3473866
featureExtractionScript()