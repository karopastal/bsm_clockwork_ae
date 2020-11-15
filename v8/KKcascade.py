##################
#   KK cascade   #
##################

### Explanations ###
# - This file contains the information on Kaluza-Klein graviton decay cascade.
# - All equations are taken from 1711.08437.

#Import generic
import math

#Import custom
import Basic



#Approximate decay width of gravitons to BSM particles
#Graviton to two other gravitons (Eq. (D.13))
def GammaG2GGapprox(n, M5, k):
  return (1 - k/Basic.m(n, M5, k))**9*(5*7*17)/(3*2**14*math.pi**2)*(k*Basic.m(n, M5, k))**0.5*Basic.m(n, M5, k)**3/(k*Basic.R(M5, k)*M5**3)

#Graviton to two scalars (Eq. (G.9))
def GammaG2SSapprox(n, M5, k):
  return (1 - k/Basic.m(n, M5, k))**16/(2**13*math.pi**2)*(k*Basic.m(n, M5, k))**0.5*k**2*Basic.m(n, M5, k)/(k*Basic.R(M5, k)*M5**3)

#Graviton to another graviton and a scalar (Eq. (G.15))
def GammaG2SGapprox(n, M5, k):
  return (1 - k/Basic.m(n, M5, k))**16/(2**5*math.pi**2)*k**2*Basic.m(n, M5, k)**2/(k*Basic.R(M5, k)*M5**3)

#Same as previous but with NLO corrections
def GammaG2SGapproxNLO(n, M5, k):
  return (1 + k/Basic.m(n, M5, k))**3*(1 - 5/2*(k/Basic.m(n, M5, k))**0.5)/(2**5*math.pi**2)*k**2*Basic.m(n, M5, k)**2/(k*Basic.R(M5, k)*M5**3)



#Kinematic functions necessary for the exact calculations of graviton decay widths
#A2 function (Eqs. (D.6-D.9))
def A2(n, x, y):
  if   n == 1:
    return 2*(x*(1 + y**2) - 2)**2/(3*(x**2 - y**2))
  elif n == 2:
    return 2*(y**2 - 2*y*(1 + x) + 3)**2/(x - y)
  elif n == 3:
    return 8*(2*x**2 + 2*x*y + (1 + y)**2)**2/(3*(x + y)**2)
  elif n == 4:
    return ((6 - 4*x + x**2 + 2*x**3 + y**4 + y**2*(x*(5*x - 6) - 5))/(9*(x**2 - y**2)))**2

#m4A2 function (Eq. (D.10))
def m4A2(a, b, n, M5, k):
  total = 0
  for j in range(1, 5):
    total += A2(j, (Basic.m(a, M5, k)**2 + Basic.m(b, M5, k)**2)/Basic.m(n, M5, k)**2, (Basic.m(a, M5, k)**2 - Basic.m(b, M5, k)**2)/Basic.m(n, M5, k)**2)
  return Basic.m(n, M5, k)**4*total

#mm function
def mm(z, M5, k):
  return k**2 + z**2/Basic.R(M5, k)**2

#IMMC function
def IMMC(a, b, n, M5, k):
  return (32/Basic.R(M5, k)**3)*k*n*Basic.m(n, M5, k)*a*Basic.m(a, M5, k)*b*Basic.m(b, M5, k)/(mm(a + b - n, M5, k)*mm(a - b + n, M5, k)*mm(b + n - a, M5, k)*mm(a + b + n, M5, k))

#AT2 function
def AT2(a, b, n, M5, k):
  return 1/5*IMMC(a, b, n, M5, k)**2/(M5*math.pi*Basic.R(M5, k))**3*(m4A2(a, b, n, M5, k) + m4A2(b, n, a, M5, k) + m4A2(n, a, b, M5, k))

#PS function
def PS(a, b, n, M5, k):
  return 1/(2*Basic.m(n, M5, k))*(Basic.m(n, M5, k)**2 - 2*(Basic.m(a, M5, k)**2 + Basic.m(b, M5, k)**2) + (Basic.m(a, M5, k)**2 - Basic.m(b, M5, k)**2)**2/Basic.m(n, M5, k)**2)**0.5



#Exact calculation of graviton decay width to a specific set of BSM particles
#Decay of graviton n to graviton a and b
def GammaG2GGInt(n, a, b, M5, k):
  if Basic.m(n, M5, k) < Basic.m(a, M5, k) + Basic.m(b, M5, k):
    return 0
  else:
    return PS(a, b, n, M5, k)/(16*math.pi*Basic.m(n, M5, k))*AT2(a, b, n, M5, k)

#Decay of graviton n to scalars a and b
def GammaG2SSInt(n, a, b, M5, k, kR):
  if Basic.m(n, M5, k) < Basic.m(a, M5, k) + Basic.m(b, M5, k):
    return 0
  else:
    return (8*a**2*b**2*k**4*n**2*(-2*a**2 - 2*b**2 - 3*kR**2 + n**2 + (a**2 - b**2)**2/(kR**2 + n**2))**(5/2)*(a**4*(27*b**2 + 11*kR**2) + a**2*(27*b**4 + 14*kR**4 - 19*kR**2*n**2 + b**2*(65*kR**2 - 27*n**2)) + kR**2*(11*b**4 + 3*kR**4 + 5*kR**2*n**2 + 8*n**4 + b**2*(14*kR**2 - 19*n**2)))**2)/(135*kR**2*(a**2 + kR**2)*(9*a**2 + kR**2)*(b**2 + kR**2)*(9*b**2 + kR**2)*M5**3*(kR**2 + (a + b - n)**2)**2*(kR**2 + (a - b + n)**2)**2*(kR**2 + (-a + b + n)**2)**2*(kR**2 + (a + b + n)**2)**2*math.pi**4)

#Decay of graviton n to scalar a and scalar 0
def GammaG2S0SInt(n, a, M5, k, kR):
  if Basic.m(n, M5, k) < Basic.m(a, M5, k) + Basic.m(0, M5, k):
    return 0
  else:
    return (64*a**2*k**4*kR*n**2*(-9*(18*a**2 + 25*kR**2) + 81*n**2 + (9*a**2 + kR**2)**2/(kR**2 + n**2))**(5/2))/(3645*(a**2 + kR**2)*(9*a**2 + kR**2)*M5**3*(81*a**4 + 18*a**2*(16*kR**2 - 9*n**2) + (16*kR**2 + 9*n**2)**2)**2*math.pi**3)

#Decay of graviton n to two scalars 0
def GammaG2S0S0Int(n, M5, k, kR):
  if Basic.m(n, M5, k) < Basic.m(0, M5, k) + Basic.m(0, M5, k):
    return 0
  else:
    return (8*k**4*n**2*(-23*kR**2 + 9*n**2)**(5/2))/(3645*M5**3*(kR**2 + n**2)**2*(25*kR**2 + 9*n**2)**2*math.pi**2)

#Decay of graviton n to scalar a and graviton b
def GammaG2SGInt(n, a, b, M5, k, kR):
  if Basic.m(n, M5, k) < Basic.m(a, M5, k) + Basic.m(b, M5, k):
    return 0
  elif a == 0:
    return (8*b**2*k**4*n**2*(955*kR**4 + 171*kR**2*n**2 + 9*b**2*(19*kR**2 + 3*n**2))**2*math.sqrt(81*b**4 - 224*kR**4 - 144*kR**2*n**2 + 81*n**4 - 18*b**2*(8*kR**2 + 9*n**2))*(6561*b**8 + 686836*kR**8 + 1532232*kR**6*n**2 + 1012338*kR**4*n**4 + 173502*kR**2*n**6 + 6561*n**8 + 1458*b**6*(119*kR**2 + 117*n**2) + 486*b**4*(2083*kR**4 + 3783*kR**2*n**2 + 1701*n**4) + 18*b**2*(85124*kR**6 + 177786*kR**4*n**2 + 102141*kR**2*n**4 + 9477*n**6)))/(2657205*kR*(b**2 + kR**2)**3*M5**3*(kR**2 + n**2)**(9/2)*(81*b**4 + 18*b**2*(16*kR**2 - 9*n**2) + (16*kR**2 + 9*n**2)**2)**2*math.pi**3)
  else:
    return (64*a**2*b**2*k**4*n**2*(kR**2*(-16*a**6 + a**4*(42*b**2 - 86*kR**2) + a**2*(-45*b**4 + 157*b**2*kR**2 - 54*kR**4) + (b**2 + kR**2)*(19*b**4 + 99*b**2*kR**2 + 16*kR**4)) + (3*b**6 - 16*b**4*kR**2 - 160*b**2*kR**4 + 115*kR**6 - 6*a**4*(b**2 - 7*kR**2) + a**2*(b**2 + kR**2)*(3*b**2 + 157*kR**2))*n**2 + (3*a**2*(b**2 - 15*kR**2) - 2*(3*b**4 + 8*b**2*kR**2 - 59*kR**4))*n**4 + (3*b**2 + 19*kR**2)*n**6)**2*math.sqrt(a**4 + b**4 - 3*kR**4 - 2*kR**2*n**2 + n**4 - 2*b**2*(kR**2 + n**2) - 2*a**2*(b**2 + kR**2 + n**2))*(a**8 + b**8 + 26*b**6*(kR**2 + n**2) - 4*a**6*(b**2 + kR**2 + n**2) + 2*b**2*(kR**2 + n**2)**2*(111*kR**2 + 13*n**2) + 2*b**4*(kR**2 + n**2)*(74*kR**2 + 63*n**2) + (kR**2 + n**2)**2*(99*kR**4 + 24*kR**2*n**2 + n**4) + a**4*(6*b**4 + 34*b**2*kR**2 + 28*kR**4 + 34*(b**2 + kR**2)*n**2 + 6*n**4) - 4*a**2*(b**2 + kR**2 + n**2)*(b**4 + 13*b**2*(kR**2 + n**2) + (kR**2 + n**2)*(12*kR**2 + n**2))))/(1215*(a**2 + kR**2)*(9*a**2 + kR**2)*(b**2 + kR**2)**3*M5**3*(kR**2 + (a + b - n)**2)**2*(kR**2 + n**2)**(9/2)*(kR**2 + (a - b + n)**2)**2*(kR**2 + (-a + b + n)**2)**2*(kR**2 + (a + b + n)**2)**2*math.pi**4)



#Exact calculation of graviton decay to any BSM particles
#Decay of graviton n to two gravitons
def GammaG2GGexact(n, M5, k):
  total = 0
  for a in range(1, n):
    for b in range(a, n):
      total += GammaG2GGInt(n, a, b, M5, k)
  return total

#Decay of graviton n to two non-zero scalars
def GammaG2SSexact(n, M5, k):
  total = 0
  kR    = k*Basic.R(M5, k)
  for a in range(1, n):
    for b in range(a, n):
      total += GammaG2SSInt(n, a, b, M5, k, kR)
  return total

#Decay of graviton n to a non-zero and zero scalar
def GammaG2S0Sexact(n, M5, k):
  total = 0
  kR    = k*Basic.R(M5, k)
  for a in range(1, n):
    total += GammaG2S0SInt(n, a, M5, k, kR)
  return total

#Decay of graviton n to graviton and non-zero scalar
def GammaG2SGexact(n, M5, k):
  total = 0
  kR    = k*Basic.R(M5, k)
  for a in range(1, n):
    for b in range(1, n):
      total += GammaG2SGInt(n, a, b, M5, k, kR)
  return total

#Decay of graviton n to graviton and zero scalar
def GammaG2S0Gexact(n, M5, k):
  total = 0
  kR    = k*Basic.R(M5, k)
  for b in range(1, n):
    total += GammaG2SGInt(n, 0, b, M5, k, kR)
  return total



#Complete decay widths of the gravitons
#Decay of graviton n to two gravitons
def GammaG2GG(n, M5, k):
  if n < 70:
    return GammaG2GGexact(math.floor(n), M5, k)
  else:
    return GammaG2GGapprox(n, M5, k)

#Decay of graviton n to two scalars
def GammaG2SS(n, M5, k):
  if n < 1000:
    return GammaG2SSexact(math.floor(n), M5, k)
  else:
    return GammaG2SSapprox(n, M5, k)

#Decay of graviton n to a non-zero scalar and another scalar
def GammaG2S0S(n, M5, k):
  if n < 400:
    return GammaG2S0Sexact(math.floor(n), M5, k)
  else:
    return GammaG2S0Sapprox(n, M5, k)

#Decay of graviton n to two zero scalars
def GammaG2S0S0(n, M5, k):
  return GammaG2S0S0Int(n, M5, k, k*Basic.R(M5, k))

#Decay of graviton n to a graviton and a non-zero scalar
def GammaG2SG(n, M5, k):
  if n < 300:
    return GammaG2SGexact(math.floor(n), M5, k)
  else:
    return GammaG2SGapprox(n, M5, k)

#Decay of graviton n to a graviton and a zero scalar
def GammaG2S0G(n, M5, k):
  if n < 400:
    return GammaG2S0Gexact(math.floor(n), M5, k)
  else:
    return GammaG2S0Gapprox(n, M5, k)

#Decay of graviton n to a graviton and a zero scalar fast
def GammaG2S0Gfast(n, M5, k):
  if n < 100:
    return GammaG2S0Gexact(math.floor(n), M5, k)
  else:
    return 0




