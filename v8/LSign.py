######################
# Local significance #
######################

### Explanations ###
# - This file contains some functions to calculate the local significance.
# - pvalueConverter is a rather complicated piece of code, but it is O(N).

#Import standard
import math, numpy, pywt
from scipy import signal

#Import custom
import v8.SBLD as SBLD
import v8.WT as WT


#Function that generates many toy experiments and orders the results for each grid point
def ToyExperimentGridMaker(listBFix, mergeX, mergeY, nTrials):

  #Initialize variables
  listGridNorm = []

  #Generate toy experiments
  for i in range(0, nTrials):
    t1       = numpy.array(SBLD.SBmain(listBFix, listBFix, 1, 0))
    t2, _, _ = WT.WaveletTransform(t1, mergeX, mergeY)
    listGridNorm.append(t2)

    #Make report
    if i%100 == 0:
      print(i)

  #Compile statistics for each bin
  orderedGrid = numpy.zeros((t2.shape[0], t2.shape[1], nTrials))
  listAvNorm  = numpy.zeros((t2.shape[0], t2.shape[1]))

  for i in range(0, t2.shape[0]):
    for j in range(0, t2.shape[1]):

      #Initialize variable
      orderedList = []

      #Loop over toy experiements
      for k in range(0, nTrials):
        orderedList.append(listGridNorm[k][i, j])

      #Order the results
      orderedList = numpy.sort(orderedList)

      #Replace in list
      for k in range(0, nTrials):
        orderedGrid[i, j, k] = orderedList[k]
        listAvNorm[i, j]    += listGridNorm[k][i, j]/nTrials

  #Return final result
  return orderedGrid, listAvNorm



#Function to convert grid from steps in constant p-value to steps in constant wavelet coefficients
def pvalueConverter(toyGrid, i, j):

  #Read preliminary values
  wmin  = toyGrid[i, j,  0]
  wmax  = toyGrid[i, j, -1]
  wstep = (wmax - wmin)/toyGrid.shape[2]

  #Initialize variables
  w           = wmin - wstep/2
  listpvalues = []

  #Safety for no signal in any case
  if wmax == 0:
    for k in range(0, toyGrid.shape[2]):
      listpvalues.append(1)

    return wmin, wmax + 0.01, listpvalues

  #Loop over toy experiments
  for k in range(0, toyGrid.shape[2]):

    #Assign p-values if not inside the bin
    while toyGrid[i, j, k] > w + wstep:
      listpvalues.append(1 - k/toyGrid.shape[2])
      w += wstep

  #Return final values
  return wmin, wmax, listpvalues



#Function that creates a grid of p-values
def pvalueGridMaker(toyGrid):

  #Initialize variables
  wMinGrid   = numpy.zeros((toyGrid.shape[0], toyGrid.shape[1]))
  wMaxGrid   = numpy.zeros((toyGrid.shape[0], toyGrid.shape[1]))
  pvalueGrid = numpy.zeros((toyGrid.shape[0], toyGrid.shape[1], toyGrid.shape[2]))

  #Loop over bins
  for i in range(0, toyGrid.shape[0]):
    for j in range(0, toyGrid.shape[1]):

      #Read info
      wmintemp, wmaxtemp, listpvaluestemp = pvalueConverter(toyGrid, i, j)

      #Write info wmin and wmax
      wMinGrid[i, j] = wmintemp
      wMaxGrid[i, j] = wmaxtemp

      #Write info pvalues
      for k in range(0, toyGrid.shape[2]):
        pvalueGrid[i, j, k] = listpvaluestemp[k]

  #Return final results
  return wMinGrid, wMaxGrid, pvalueGrid



#Function that returns the p-value grid for a given wavelet transform grid
def pvalueCalc(wMinGrid, wMaxGrid, pvalueGrid, cw):

  #Initialize variables
  pvalueEvent = numpy.zeros((wMinGrid.shape[0], wMinGrid.shape[1]))

  #Loop over bins
  for i in range(0, wMinGrid.shape[0]):
    for j in range(0, wMinGrid.shape[1]):

      #Find the coefficient
      coeff    = numpy.floor((cw[i, j] - wMinGrid[i, j])/(wMaxGrid[i, j] - wMinGrid[i, j])*pvalueGrid.shape[2])
      coeffint = int(coeff)

      #Correct if necessary
      if coeffint < 0:
        coeffint = 0

      if coeffint > pvalueGrid.shape[2] - 1:
        coeffint = pvalueGrid.shape[2] - 1

      #Assign correct p-value
      pvalueEvent[i, j] = pvalueGrid[i, j, coeffint]

  #Return pvalue of the wavelet coefficient
  return pvalueEvent




