##########
#   FT   #
##########

### Explanations ###
# - This program assigns a test statistic using Fourier transforms.

#Import generic
import cmath, math, numpy
from   bisect import bisect_left

#Import custom
import Basic



#Fourier transform
def FourierP(Signal, lum, lumDiv, T, k, M5, massList, mpnk):

  #Initialize variables
  Psq = 0

  #Bins to scan over
  binMin = bisect_left(massList,  max(k,            150   ))
  binMax = bisect_left(massList,  max((1 + mpnk)*k, binMin))

  #Loop over signal
  for b in range(binMin, binMax - 1):
    if lumDiv:
      lumT = lum[b]
    else:
      lumT = 1

    Psq += 1/(2*math.pi)**0.5*Signal[b]/lumT*cmath.exp(1j*2*math.pi*Basic.R(M5, k)*(massList[b]**2 - k**2)**0.5/T)*(massList[b + 1] - massList[b])

  #Return P(T)
  return (numpy.absolute(Psq))**2



#List of test statistics
def FourierTS(Signal, Background, backgroundSub, lum, lumDiv, T, k, M5, massList, stepFT):

  #Different values of upper limit
  mpnkList = numpy.linspace(5/stepFT, 5, stepFT)

  #Initialize variable
  listTS = []

  #Signal internal (if background is subtracted)
  if backgroundSub:
    SignalInt = [x1 - x2 for (x1, x2) in zip(Signal, Background)]
  else:
    SignalInt = Signal

  #Loop over possible values of mpnk
  for mpnk in mpnkList:
    listTS.append(FourierP(SignalInt, lum, lumDiv, T, k, M5, massList, mpnk))

  #Return the list of test statistics
  return listTS




