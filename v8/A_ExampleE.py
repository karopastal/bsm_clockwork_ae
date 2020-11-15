#####################
#   Classifier FT   #
#####################

#Explanations
# - This code uses Fourier transforms to determine whether a signal is present or not.

#Import
print("Starting program...")
print("")
print("Importing files...")

#Standard
import math, numpy, scipy, random, os, sys

#Custom
import Basic, FT
import SBLD  as SB

print("Files imported.")
print("")



#Settings
print("Reading settings...")

#Binning
minMass = 150
maxMass = 2700
nBins   = 1275

#Fourier transform steps
stepFT = 100

#Number of trials for background test statistic and signal + background test statistic
nTrials1 = 1000
nTrials2 = 1000

#Background subtraction
backgroundSub = False

#Division by luminosity
lumDiv = True

#Read value for this line
M5 = 6000
k  = 750 

print("Setting read")
print("")



#Start loop
print("Starting new point...")
print("Evaluating signal...")

#Mass and scale Lists
massList = numpy.linspace(minMass, maxMass, nBins)

#Generate fix signal
SFix, lum = SB.Sfix(M5, k, massList)
BFix      = SB.Bfix(massList)
print("Signal evaluated.")
print("")



#Generate trial background + signal
#Initialize variables
print("Beginning toy experiment of background + signal")
statTestFT1 = []

for i in range(0, nTrials1):
    
  #Generate random binned event
  eventsTrial1 = SB.SBmain(BFix, SFix, 1, 1)

  #Calculate test statistic FT
  statTestFT1.append(FT.FourierTS(eventsTrial1, BFix, backgroundSub, lum, lumDiv, 1, k, M5, massList, stepFT))

print("Toy experiments of background + signal done.")
print("")



#Generate trial background
#Initialize variables
print("Beginnning toy experiment of background only")
statTestFT2 = []

for i in range(0, nTrials2):
    
  #Generate random binned events
  eventsTrial2 = SB.SBmain(BFix, SFix, 1, 0)
    
  #Calculate test statistic FT
  statTestFT2.append(FT.FourierTS(eventsTrial2, BFix, backgroundSub, lum, lumDiv, 1, k, M5, massList, stepFT))

print("Toy experiments of background only.")
print("")



#Distribution of statistical significance Fourier transform
print("Computing test statistics...")

significanceDistributionFTTotal = []

for n in range(0, stepFT):

  significanceDistributionFT = []
    
  for i in range(0, nTrials1):
    
    #Initialize variable
    temp = 0
    
    #Loop over background events
    for j in range(0, nTrials2):
      if statTestFT2[j][n] >= statTestFT1[i][n]:
        temp += 1/nTrials2
    
    #Append to list
    significanceDistributionFT.append(temp)
        
  #Append to list of significances
  significanceDistributionFTTotal.append(significanceDistributionFT)



#Average
templist1 = []

for i in range(0, stepFT):
  temp = sum(significanceDistributionFTTotal[i])/len(significanceDistributionFTTotal[i])
  print(i, temp)
  templist1.append(temp)

print("Min value is:", min(templist1))



#Median
templist2 = []

for i in range(0, stepFT):
  t1 = numpy.sort(significanceDistributionFTTotal[i])
  temp = t1[int(numpy.floor(0.5*len(significanceDistributionFTTotal[i])))]
  print(i, temp)
  templist2.append(temp)

print("Min value is:", min(templist2))

print("Print statistics computations done.")
print("")



#Saving results
fout = open("ExampleE_out.txt","w")
fout.write("Results\n\n")
fout.write("M5:\n")
fout.write(str(M5) + "\n")
fout.write("k:\n")
fout.write(str(k) + "\n")
fout.write("Expected average significance:\n")
fout.write(" ".join(map(str, templist1)) + "\n")
fout.write("Best value of expected average significance:\n")
fout.write(str(min(templist1)) + "\n")
fout.write("Expected median significance:\n")
fout.write(" ".join(map(str, templist2)) + "\n")
fout.write("Best value of expected median significance:\n")
fout.write(str(min(templist2)))
fout.close()




