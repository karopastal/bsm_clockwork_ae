##################
#   Luminosity   #
##################

### Explanations ###
# - Calculate the luminosities associated to each process

#Comments:
# - S is the center of mass energy (13 TeV at LHC).
# - M is the mass of the resonance.
# - q is the renormalization scale.
# - fi are the quark flavors.

#Import generic
import scipy.integrate

#Import custom
import PDFcalc



#Custom integrator
def integrator(functionToIntegate, M, S):
  return scipy.integrate.quad(functionToIntegate, M**2/S, 1, epsabs=10**(-2))[0]



#Gluon-gluon luminosity
def ggLum(S, M, q, pdf):

  #Function to integrate
  def functionToIntegate(x):
    return 1/(S*x)*PDFcalc.PDFxq(x, q, 0, pdf)*PDFcalc.PDFxq(M**2/(x*S), q, 0, pdf)

  #Return integrated function
  return integrator(functionToIntegate, M, S)



#Quark-antiquark luminosity
def qqbarLum(S, M, q, f1, f2, pdf):

  #Symmetry factor
  if f1 == f2:
    sym = 2
  else:
    sym = 1

  #Function to integrate
  def functionToIntegate(x):
    return 1/(sym*S*x)*(PDFcalc.PDFxq(x, q, f1, pdf)*PDFcalc.PDFxq(M**2/(x*S), q, -f2, pdf) + PDFcalc.PDFxq(x, q, f2, pdf)*PDFcalc.PDFxq(M**2/(x*S), q, -f1, pdf))

  #Return integrated function
  return integrator(functionToIntegate, M, S)




