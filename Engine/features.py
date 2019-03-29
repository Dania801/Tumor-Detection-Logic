import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import imutils

def readPreprocessedDataset(path):
  """
  Reads preprocessed images and append it to array reprensents the entire dataset.
  @params path
  @rtype {list}
  """
  images = []
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

def StdGray(images, maxGray, minGray):
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
  

def featureExtractionScript():
  maxGray = 150 
  minGray = 210 
  images = readPreprocessedDataset('../Data/CT_cropped/*.jpg')
  meanGrayValues = meanGray(images, maxGray, minGray)
  stdGrayValues = StdGray(images, maxGray, minGray)

featureExtractionScript()