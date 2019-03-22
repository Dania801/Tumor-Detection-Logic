import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import imutils

def readPreprocessesDataset(path):
  images = []
  for count, fileName in enumerate(glob.glob(path)):
    image = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
    images.append(image)
  return images

def featureExtractionScript():
  images = readPreprocessesDataset('../Data/CT_cropped/*.jpg')
  print (len(images))

featureExtractionScript()