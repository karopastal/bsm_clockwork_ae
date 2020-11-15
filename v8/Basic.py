#############
#   Basic   #
#############

### Explanations ###
# - This file contains basic information on the linear dilaton.
# - All equations are taken from 1711.08437.

#Import generic
import math

#Import custom
import Couplings



#Radius (Eq. (2.10))
def R(M5, k):
  return math.log(Couplings.MP**2*k/M5**3 + 1)/(2*math.pi*k)



#Mass of the nth KK dilatons for rigid boundary conditions (Eq. (2.40))
#Comment: For n != 0, this also corresponds to the mass of the gravitons (Eq. (2.34))
def m(n, M5, k):
  if n > 0:
    return (k**2 + n**2/R(M5, k)**2)**0.5
  else:
    return (8/9)**0.5*k

#Coupling of gravitons to stress-energy tensor (Eq. (2.36))
def Lambda2(n, M5, k):
  return M5**3*math.pi*R(M5, k)*(1 + k**2*R(M5, k)**2/n**2)

#Decay constant for graviton decay (Eq. (C.6))
def Gamma0(n, M5, k):
  return m(n, M5, k)**3/(80*math.pi*Lambda2(n, M5, k))



#Phase-space to different Standard Model channels (Eqs. (C.1 to C.5))
#Two photons
def PSgammagamma():
  return 1

#Two gluons
def PSgg(R):
  if 4*R**2 < 1:
    return 8*(1 - 4*R**2)**0.5
  else:
    return 0

#Two Z bosons
def PSZZ(R):
  if 4*R**2 < 1:
    return (13/12 + 14/3*R**2 + 4*R**4)*(1 - 4*R**2)**0.5
  else:
    return 0

#Two W bosons
def PSWW(R):
  return 2*PSZZ(R)

#Two fermions
def PSffbar(R):
  if 4*R**2 < 1:
    return 1/2*(1 + 8/3*R**2)*(1 - 4*R**2)**1.5
  else:
    return 0

#Two Higgs bosons
def PShh(R):
  if 4*R**2 < 1:
    return 1/12*(1 - 4*R**2)**2.5
  else:
    return 0

#Total phase-space
def PS2SM(mn):

  #Gauge bosons and Higgs
  total  = PSgammagamma()
  total += PSgg(Couplings.mg/mn)
  total += PSZZ(Couplings.mZ/mn)
  total += PSWW(Couplings.mW/mn)
  total += PShh(Couplings.mh/mn)

  #Neutrinos
  for m in [Couplings.mnu1, Couplings.mnu2, Couplings.mnu3]:
    total += 1/2*PSffbar(m/mn)

  #Charged leptons
  for m in [Couplings.me, Couplings.mmu, Couplings.mtau]:
    total += PSffbar(m/mn)

  #Quarks
  for m in [Couplings.muQ, Couplings.mdQ, Couplings.msQ, Couplings.mcQ, Couplings.mbQ, Couplings.mtQ]:
    total += 3*PSffbar(m/mn)

  #Return total
  return total




