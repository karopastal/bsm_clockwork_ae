################
#   Graviton   #
################

### Explanations ###
# - This file contains the information for the cross sections and branching ratios of a given graviton.
# - Some branching ratios that we did not use are included here for convenience but are never called.
# - To speed up the code, some functions are designed to reuse the results of previous computations. These must of course be performed first.

#Import generic
import math, numpy

#Import custom
import v8.Basic as Basic
import v8.Couplings as Couplings
import v8.KKcascade as KKcascade
import v8.Luminosity as Luminosity


#Decay width of a graviton n
def Gamma(n, M5, k):

  #Partial widths
  total  = Basic.PS2SM(Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)
  total += KKcascade.GammaG2GG(n, M5, k)
  total += KKcascade.GammaG2SG(n, M5, k)  
  total += KKcascade.GammaG2S0Gfast(n, M5, k) 

  #Return total
  return total



#Branching ratio to gauge bosons and Higgs
#Photons
def BRgammagamma(n, M5, k, listGamma):
  return Basic.PSgammagamma()*Basic.Gamma0(n, M5, k)/listGamma[n]

#Gluons (not used)
def BRgg(n, M5, k, listGamma):
  return Basic.PSgg(Couplings.mg/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Z bosons (not used)
def BRZZ(n, M5, k, listGamma):
  return Basic.PSZZ(Couplings.mZ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#W bosons (not used)
def BRWW(n, M5, k, listGamma):
  return Basic.PSWW(Couplings.mW/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#h bosons (not used)
def BRhh(n, M5, k, listGamma):
  return Basic.PShh(Couplings.mh/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]



#Branching ratios to fermions
#Electrons (not used)
def BRee(n, M5, k, listGamma):
  return Basic.PSffbar(Couplings.me/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Muons (not used)
def BRmumu(n, M5, k, listGamma):
  return Basic.PSffbar(Couplings.mmu/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Taus (not used)
def BRtautau(n, M5, k, listGamma):
  return Basic.PSffbar(Couplings.mtau/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Neutrinos 1 (not used)
def BRnunu1(n, M5, k, listGamma):
  return 1/2*Basic.PSffbar(Couplings.mnu1/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Neutrinos 2 (not used)
def BRnunu2(n, M5, k, listGamma):
  return 1/2*Basic.PSffbar(Couplings.mnu2/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Neutrinos 3 (not used)
def BRnunu3(n, M5, k, listGamma):
  return 1/2*Basic.PSffbar(Couplings.mnu3/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Ups (not used)
def BRuu(n, M5, k, listGamma):
  return 3*Basic.PSffbar(Couplings.muQ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Downs (not used)
def BRdd(n, M5, k, listGamma):
  return 3*Basic.PSffbar(Couplings.mdQ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Stranges (not used)
def BRss(n, M5, k, listGamma):
  return 3*Basic.PSffbar(Couplings.msQ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Charms (not used)
def BRcc(n, M5, k, listGamma):
  return 3*Basic.PSffbar(Couplings.mcQ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Bottoms (not used)
def BRbb(n, M5, k, listGamma):
  return 3*Basic.PSffbar(Couplings.mbQ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]

#Tops (not used)
def Btt(n, M5, k, listGamma):
  return 3*Basic.PSffbar(Couplings.mtQ/Basic.m(n, M5, k))*Basic.Gamma0(n, M5, k)/listGamma[n]



#Cross section of a given resonance
def CrossSectionKKn(n, M5, k, listMass, pdf, S):

  #Read constants
  M = listMass[n]
  q = M

  #Calculate the luminosities
  lumGG = Luminosity.ggLum(S, M, q, pdf)
  lumUU = Luminosity.qqbarLum(S, M, q, 1, 1, pdf)
  lumDD = Luminosity.qqbarLum(S, M, q, 2, 2, pdf)

  #Return cross section (Eq. (2.57))
  return M**2*math.pi/(48*Basic.Lambda2(n, M5, k))*(3*lumGG + 4*lumUU + 4*lumDD)*Couplings.Gev2fb 




