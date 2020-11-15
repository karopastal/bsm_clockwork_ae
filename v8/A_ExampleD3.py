#############
#   RF TS   #
#############

#Explanations
# - This code uses a anomaly finder and a test statistic to set limits.

#Import
print("Starting program...")
print("")
print("Importing files...")

#Import standard
import math, numpy, scipy, random, os, sys

#Import custom
import Merger, LSign, TestS, WT
import SBLD as SB

#Machine learning
import keras, sklearn
from keras.layers            import Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers            import Convolution2D, Conv1D, Conv2D, Conv2DTranspose
from keras.layers            import AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.merge      import add, concatenate
from keras.models            import load_model, Model, Sequential
from sklearn.model_selection import train_test_split

print("Files imported.")
print("")



#Settings
print("Reading settings...")

#Binning
minMass = 150
maxMass = 2700
nBins   = 1275
mergeX  = 20
mergeY  = 2

#Number of possible maximum values of the minimum value of the NN checked
NminV = 200

#Cone of influence
Cone   = False
NconeR = 2

#Number of trials for p-value map, background test statistic and signal + background test statistic
nTrials  = 500
nTrials1 = 100
nTrials2 = 100

#Read value for this line
M5 = 7000
k  = 750 



#Mass and scale Lists
massList = numpy.linspace(minMass, maxMass, nBins)

#Significance grid
print("Beginning evaluation of p-value grid...")
print("")

#Grid of toy experiments
BFix                     = SB.Bfix(massList)
toyExpGrid1, listavNorm1 = LSign.ToyExperimentGridMaker(BFix, mergeX, mergeY, nTrials)

#Grid of pvalues
wMinGrid, wMaxGrid, pvalueGrid = LSign.pvalueGridMaker(toyExpGrid1)

print("Evaluation of p-value grid done.")
print("")



#Define model for region finder
print('Loading model...')
model2 = Sequential()

#add model layers
model2.add(Conv2D(32,  kernel_size=5, activation='softplus', input_shape=(pvalueGrid.shape[0], pvalueGrid.shape[1], 1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64,  kernel_size=5, activation='softplus'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128,  kernel_size=5, activation='softplus'))
model2.add(Flatten())
model2.add(Dense(5000, activation='softplus'))
model2.add(Dense(pvalueGrid.shape[0]*pvalueGrid.shape[1], activation='softplus'))

#Compile
model2.compile(optimizer='adam',loss='mean_squared_error')

#Load best weights
model2.load_weights('ZExampleD_WandB.txt')
print('Model loaded.')



#Generate fix signal
SFix, lum = SB.Sfix(M5, k, massList)
BFix      = SB.Bfix(massList)
print("Signal evaluated.")
print("")

#Generate trial background + signal
print("Beginnning toy experiment of background + signal")
statTest1 = []

for i in range(0, nTrials1):
    
  #Generate random binned event
  eventsTrial1 = SB.SBmain(BFix, SFix, 1, 1)

  #Do wavelet transform
  cwtmatBSTrial1Norm, _, _ = WT.WaveletTransform(eventsTrial1, mergeX, mergeY, Cone, NconeR)
  cwtmatBSTrial1NormPerAvg = cwtmatBSTrial1Norm/listavNorm1

  #Format for predictions
  cwtmatBSTrial1Flat      = numpy.ravel(cwtmatBSTrial1NormPerAvg)
  cwtmatBSTrial1Formatted = numpy.reshape(cwtmatBSTrial1Flat,(1, pvalueGrid.shape[0], pvalueGrid.shape[1], 1))

  #Make prediction
  temp1 = model2.predict(cwtmatBSTrial1Formatted)[0]
  temp2 = numpy.reshape(temp1, (cwtmatBSTrial1Norm.shape[0], cwtmatBSTrial1Norm.shape[1]))

  #Calculate test statistic FT
  statTest1.append(TestS.testStatCalc2(cwtmatBSTrial1Norm, temp2, NminV, wMinGrid, wMaxGrid, pvalueGrid, massList, mergeY))

print("Toy experiments of background + signal done.")
print("")



#Generate trial background only
print("Beginnning toy experiment of background only")
statTest2 = []

for i in range(0, nTrials2):
    
  #Generate random binned event
  eventsTrial2 = SB.SBmain(BFix, SFix, 1, 0)

  #Do wavelet transform
  cwtmatBTrial2Norm, _, _ = WT.WaveletTransform(eventsTrial2, mergeX, mergeY, Cone, NconeR)
  cwtmatBTrial2NormPerAvg = cwtmatBTrial2Norm/listavNorm1

  #Format for predictions
  cwtmatBTrial2Flat      = numpy.ravel(cwtmatBTrial2NormPerAvg)
  cwtmatBTrial2Formatted = numpy.reshape(cwtmatBTrial2Flat,(1, pvalueGrid.shape[0], pvalueGrid.shape[1], 1))

  #Make prediction
  temp1 = model2.predict(cwtmatBTrial2Formatted)[0]
  temp2 = numpy.reshape(temp1, (cwtmatBTrial2Norm.shape[0], cwtmatBTrial2Norm.shape[1]))

  #Calculate test statistic FT
  statTest2.append(TestS.testStatCalc2(cwtmatBTrial2Norm, temp2, NminV, wMinGrid, wMaxGrid, pvalueGrid, massList, mergeY))

print("Toy experiments of background only.")
print("")



#Distribution of statistical significance wavelet transform
print("Computing test statistics...")

significanceDistributionWTTotal = []

for n in range(0, NminV):

  significanceDistributionWT = []
    
  for i in range(0, nTrials1):
    
    #Initialize variable
    temp = 0
    
    #Loop over background events
    for j in range(0, nTrials2):
      if statTest2[j][n] >= statTest1[i][n]:
        temp += 1/nTrials2
    
    #Append to list
    significanceDistributionWT.append(temp)
        
  #Append to list of significances
  significanceDistributionWTTotal.append(significanceDistributionWT)



#Average
templist1 = []

for i in range(0, NminV):
  temp = sum(significanceDistributionWTTotal[i])/len(significanceDistributionWTTotal[i])
  print(i, temp)
  templist1.append(temp)

print("Min value is:", min(templist1))



#Median
templist2 = []

for i in range(0, NminV):
  t1 = numpy.sort(significanceDistributionWTTotal[i])
  temp = t1[int(numpy.floor(0.5*len(significanceDistributionWTTotal[i])))]
  print(i, temp)
  templist2.append(temp)

print("Min value is:", min(templist2))

print("Print statistics computations done.")
print("")



#Saving results
fout = open("ExampleD3_out.txt","w")
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




