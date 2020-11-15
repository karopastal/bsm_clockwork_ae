########################
#   Signal generator   #
########################

# Explanations
# - Generate a series of signals

# Import
print("Starting program...")
print("")
print("Importing files...")

# Import standard
import math, numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Import custom
import Merger, WT
import SBLD as SB

print("Files imported.")
print("")

#  Settings
print("Reading settings...")

#  Binning
minMass = 150
maxMass = 2700
nBins = 2550

massList = numpy.linspace(minMass, maxMass, nBins + 1)

#  Parameters of the signal
M5 = 3000
k = 750

#  Files to write
nameOutputS = 'Signal.txt'
nameOutputSB = 'SBRandom.txt'

print("Setting read.")
print("")

#  Start generation of signal
print("Starting generation...")

#  Generate fix signal
SFix, _ = SB.Sfix(M5, k, massList)

#  Generate fix background
BFix = SB.Bfix(massList)

# Generate random signal + background
SBrandom = SB.SBmain(BFix, SFix, 1, 1)

# Saving results for fixed signal
Awrite = [M5] + [k] + SFix
fout = open(nameOutputS, "w")
fout.write(" ".join(map(str, Awrite)) + "\n")
fout.close()

# Saving results for signal + background with statistical fluctuations
Awrite = [M5] + [k] + SBrandom
fout = open(nameOutputSB, "w")
fout.write(" ".join(map(str, Awrite)) + "\n")
fout.close()

# Do wavelet transform of the signal + background with statistical fluctuations
mergeX = 1
mergeY = 1

yticks1 = [math.log(1), math.log(3), math.log(10), math.log(30), math.log(100), math.log(300), math.log(1000)]
yticks2 = (1, 3, 10, 30, 100, 300, 1000)
binwidth = (maxMass - minMass)/nBins

plt.gcf().subplots_adjust(left=0.135, bottom=0.14)
plt.rcParams.update({'font.size': 14})

cwtmatr1, _, _ = WT.WaveletTransform(SBrandom, mergeX, mergeY)
plt.imshow(numpy.flipud(numpy.transpose(cwtmatr1)), extent=[minMass, maxMass, math.log(binwidth), math.log(binwidth*2**(cwtmatr1.shape[1]*mergeY/WT.NBins))], cmap='bwr', aspect='auto', vmax=2.5, vmin=0, interpolation='gaussian')
plt.yticks(yticks1, yticks2)
plt.xlabel('$m_{\gamma\gamma}$ [GeV]')
plt.ylabel('$a$ [GeV]')
axes = plt.gca()
plt.savefig('SBrandom.pdf')
plt.clf()

#Do wavelet transform of the exact signal
cwtmatr2, _, _ = WT.WaveletTransform(SFix, mergeX, mergeY)
plt.imshow(numpy.flipud(numpy.transpose(cwtmatr2)), extent=[minMass, maxMass, math.log(binwidth), math.log(binwidth*2**(cwtmatr2.shape[1]*mergeY/WT.NBins))], cmap='bwr', aspect='auto', vmax=2.5, vmin=0, interpolation='gaussian')
plt.yticks(yticks1, yticks2)
plt.xlabel('$m_{\gamma\gamma}$ [GeV]')
plt.ylabel('$a$ [GeV]')
axes = plt.gca()
plt.savefig('SFix.pdf')
plt.clf()
0
#Final report
print("Done.")




