############
#   SBLD   #
############

### Explanations ###
# - Calculate the signal associated to the linear dilaton

#Import standard
import math, numpy, os, scipy
from parton import mkPDF
import matplotlib.pyplot as plt
import scipy.optimize as optimize

#Import custom
import v8.Basic as Basic
import v8.Background as Background
import v8.Couplings as Couplings
import v8.Graviton as Graviton
import v8.KKcascade as KKcascade
import v8.Luminosity as Luminosity
import v8.PDFcalc as PDFcalc
import v8.Smearing as Smearing


#Generate signal with no random fluctuations
def Sfix(M5, k, massList):

  #Parameter
  SLHC   = Couplings.SLHC
  Eff    = Couplings.Eff
  Intlum = Couplings.Intlum

  #Load pdf information
  print(os.getcwd())
  pdf = mkPDF('MSTW2008lo68cl', 0, pdfdir=os.getcwd() + '/v8')

  #Calculate masses of KK gravitons
  listMassKK = [0]
  mtemp      = 0
  nint       = 1

  while mtemp < massList[-1]:
    mtemp = Basic.m(nint, M5, k)
    listMassKK.append(mtemp)
    nint += 1

  #Calculate decay widths, branching ratios to photons and cross sections
  listGamma     = [0]
  listBRPhotons = [0]
  listCS        = [0]

  for n in range(1, len(listMassKK)):
    listGamma.append(Graviton.Gamma(n, M5, k))
    listBRPhotons.append(Graviton.BRgammagamma(n, M5, k, listGamma))
    listCS.append(Graviton.CrossSectionKKn(n, M5, k, listMassKK, pdf, SLHC))

  #Write properties
  if Couplings.writeProp:

    ftemp = open("GravitonProperties.txt", "w")

    ftemp.write("Summary of the graviton properties \n")
    ftemp.write("n     mass     Decay width     Cross section [fb]    BR to photons \n")

    for n in range(0, len(listMassKK)):
      ftemp.write(str(n) + " " + str(listMassKK[n]) + " " + str(listGamma[n]) + " " + str(listCS[n]) + " " + str(listBRPhotons[n]) + " " + "\n")

    ftemp.close()

  #Apply smearing
  listS      = []
  listCSplot = []

  for i in range(0, len(massList) - 1):

    #Initialize variables
    total      = 0
    mass       = massList[i]
    deltamass  = massList[i + 1] - massList[i]

    #Sum over particles
    for particle in range(1, len(listMassKK)):
      total += listCS[particle]*Smearing.shape(mass, listMassKK[particle], listGamma[particle])*listBRPhotons[particle]*Eff*Intlum
    
    #Append to lists
    listS.append(total*deltamass)
    listCSplot.append(total/Intlum)

  #Calculate parton luminosity
  lum = []

  for mass in massList:
    lum.append(Luminosity.ggLum(SLHC, mass, mass, pdf) + 4/3*Luminosity.qqbarLum(SLHC, mass, mass, 1, 1, pdf) + 4/3*Luminosity.qqbarLum(SLHC, mass, mass, 2, 2, pdf))

  #Make plot
  if Couplings.makePlot:
    massListR = numpy.delete(massList, -1)
    plt.plot(massListR, listCSplot)
    axes = plt.gca()
    plt.xlabel('$m_{\gamma\gamma}$ [GeV]')
    plt.ylabel(' d$\sigma$/dm x $\epsilon$ x BR[fb/GeV]')
    plt.savefig('Signal.pdf')
    plt.clf()

  #Return
  return listS, lum



#Generate background with no random fluctuations
def Bfix(massList):

  massBGlist = []
  BGlist     = []
  listB      = []

  with open('v8/Background.txt') as f:
    for line in f:
      data = line.split()
      massBGlist.append(float(data[0]))
      BGlist.append(float(data[1]))

  for i in range(0, len(massList) - 1):
    mass      = massList[i]
    deltamass = massList[i + 1] - massList[i]
    listB.append(Background.BackgroundCal(massBGlist, BGlist, mass)/20*deltamass)

  return listB


#Generate random fluctuations
def SBmain(listBFix, listSFix, aB, aS):

  listBSrandom = []

  for i in range(0, len(listSFix)):
    listBSrandom.append(numpy.random.poisson(lam = aB*listBFix[i] + aS*listSFix[i]))

  return listBSrandom




