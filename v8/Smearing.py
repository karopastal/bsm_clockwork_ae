################
#   Smearing   #
################

### Explanations ###
# - Apply smearing
# - The smearing is based on 1711.08437.

#Import generic
import math



#Energy resolution ATLAS
def EresATLAS(E):
  a = 0.12
  c = 0.01
  return (a**2/E + c**2)**0.5

#Resolution
def resATLAS(m):
  return EresATLAS(m/2)/2**0.5

#Shape
def shape(M, m, Gamma):
  sigma = (Gamma**2 + (resATLAS(m)*m)**2)**0.5
  return 1/(2*math.pi*sigma**2)**0.5*math.exp(-(M - m)**2/(2*sigma**2))




