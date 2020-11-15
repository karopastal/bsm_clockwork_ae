##################
# Test statistic #
##################

### Explanations ###
# - This file contains some functions to calculate the test statistic.

#Import standard
import math, numpy
from bisect  import bisect_left
import matplotlib.pyplot as plt

#Import custom
import Basic, LSign, WT



#Test statistic 2 (Sum of negative log likelihood for bins most likely to be a signal in each column with machine learning)
def testStatCalc2(cw, MatIn, NminValue, wMinGrid, wMaxGrid, pvalueGrid, massList, mergeY):

  #Initialize variables
  testStatList = []

  #Calculate p-value map
  pvalueMap = LSign.pvalueCalc(wMinGrid, wMaxGrid, pvalueGrid, cw)

  #Different values of upper limit
  minValueList = numpy.linspace(0.0, 20.0, NminValue)

  #Loop over columns
  for minValue in minValueList:

    testStat = 0

    for i in range(0, cw.shape[0]):

      MaxColumn = 0

      #Loop over bins in column
      for j in range(0, cw.shape[1]):

        #Check if this corresponds to the maximum value
        if MatIn[i, j] > minValue and MatIn[i, j] > MaxColumn:
          MaxColumn = MatIn[i, j]
          iMax      = i
          jMax      = j

      #Add value to statistical significance
      if i < cw.shape[0] - 1:
        binWidth  = massList[i + 1] - massList[i]
      else:
        binWidth  = massList[i] - massList[i - 1]

      if MaxColumn > 0:
        scale     = 2**(jMax*mergeY/WT.NBins)*binWidth
        testStat += -math.log(pvalueMap[iMax, jMax])/scale

    testStatList.append(testStat)

  #Return test statistic
  return testStatList



#Test statistic for the autoencoder
def testStatCalc5(rIn, rOut):

  #Initalize variable
  sumR = 0

  #Loop over results
  for i in range(0, len(rIn)):
    sumR += (rIn[i] - rOut[i])**2

  #Return loss function
  return sumR/len(rIn)




