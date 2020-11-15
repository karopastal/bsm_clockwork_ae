#########################
#   Signal generators   #
#########################

#Explanations
# - Generate a series of signals

#Import
print("Starting program...")
print("")
print("Importing files...")

#Import standard
import math, numpy, random , os, sys

#Import custom
import Basic
import SBLD as SB

print("Files imported.")
print("")



#Settings
print("Reading settings...")

#Binning
minMass = 150
maxMass = 2700
nBins   = 1275

#Number of signals to create
nTrials = 50

#Range to scan
M5min = 1000
M5max = 6500
kmin  = 200
kmax  = 2700

print("Setting read")
print("")



#Create file to write
nameOutput = 'ZExampleD_Signals.txt'
ftemp      = open(nameOutput,"+w")
ftemp.close()



#Start loop
massList = numpy.linspace(minMass, maxMass, nBins)

print("Starting generation")

for i in range(0, nTrials):

  #Read value for this line
  M5  = random.uniform(M5min, M5max)
  lnk = random.uniform(math.log(kmin), math.log(kmax))
  k   = math.exp(lnk)

  #Generate fix signal
  SFix, _ = SB.Sfix(M5, k, massList)

  #Saving results
  Awrite = [M5] + [k] + SFix
  fout = open(nameOutput, "a")
  fout.write(" ".join(map(str, Awrite)) + "\n")
  fout.close()

  #Make report
  if i%10 == 0:
    print(i)




