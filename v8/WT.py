#########################
# Signal and background #
#########################

### Explanations ###
# - Returns the wavelet transform
# - A complex Morlet transform is used.
# - pywt is a fairly new library to do wavelet transforms.
# - The cone of influence cut is set to off by default.

#Import standard
import math, numpy, pywt
from scipy import signal

#Import custom
import v8.Merger as Merger



#Number of bins
NBins = 12



#Wavelet transform general
def WaveletTransform(SignalIn, mergeX, mergeY, cone=False, Ncone=1):

  #Define widths
  temp   = numpy.arange(0, NBins*math.log(len(SignalIn))/math.log(2) - NBins)
  widths = 2**(temp/NBins)

  #Wavelet transform
  cwtmatr, _ = pywt.cwt(SignalIn, widths, 'cmor2.0-1.0')

  #Merger
  cwmatrMergedNorm = Merger.merger(abs(cwtmatr), mergeY, mergeX)
  cwmatrMergedReal = Merger.merger(numpy.real(cwtmatr), mergeY, mergeX)
  cwmatrMergedImag = Merger.merger(numpy.imag(cwtmatr), mergeY, mergeX)

  #Cone of influence
  if cone:
    for i in range(0, cwmatrMergedNorm.shape[0]):
      for j in range(0, cwmatrMergedNorm.shape[1]):

        #Calculate scale
        scale = widths[mergeY*i]

        #Calculate position
        m = mergeX*j

        #Replace by zero if in cone of influence
        if m < Ncone*2**0.5*scale or m > cwtmatr.shape[1] - Ncone*2**0.5*scale:
          cwmatrMergedNorm[i, j] = 0
          cwmatrMergedReal[i, j] = 0
          cwmatrMergedImag[i, j] = 0

  #Return result
  return numpy.transpose(cwmatrMergedNorm), numpy.transpose(cwmatrMergedReal), numpy.transpose(cwmatrMergedImag)




