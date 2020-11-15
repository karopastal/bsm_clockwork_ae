##########
# Merger #
##########

### Explanations ###
# - This file contains a function to merge together bins to form a less fine 2D grid.

#Import standard
import math, numpy



#Function that returns a less fine grid
def merger(gridIn, sizeBinx, sizeBiny):

  #Calculate size of new bins
  nBinxt = numpy.floor(gridIn.shape[0]/sizeBinx)
  nBinyt = numpy.floor(gridIn.shape[1]/sizeBiny)

  nBinx = int(nBinxt)
  nBiny = int(nBinyt)

  #Initialize variables
  gridOut = numpy.zeros([nBinx, nBiny])

  #Loop over new bins
  for i in range(0, nBinx):
    for j in range(0, nBiny):

      #Initialize variables
      sumBins = 0

      #Loop over old bins in new bin
      for k1 in range(0, sizeBinx):
        for k2 in range(0, sizeBiny):
          sumBins += gridIn[i*sizeBinx + k1, j*sizeBiny + k2]

      #Assign the average to the new grid
      gridOut[i, j] = sumBins/(sizeBinx*sizeBiny)

  #Return merged grid
  return gridOut




