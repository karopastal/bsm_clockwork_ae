######################
#   Autoencoder TS   #
######################

#Explanations
# - This code uses a previously trained autoencoder to find signals

#Import
print("Starting program...")
print("")
print("Importing files...")

#Import standard
import math, numpy, scipy, random, os, sys

#Import custom
import Merger, LSign, TestS, WT
import SBLD  as SB

#Machine learning
import keras, sklearn
from keras.layers            import Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers            import Convolution2D, Conv1D, Conv2D, Conv2DTranspose
from keras.layers            import AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Reshape, UpSampling2D
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

#Cone of influence
Cone   = False
NconeR = 2

#File to save weights and biases
nameModelFile = "ZExampleC_WandB.txt"

#Number of trials for p-value map, background test statistic and signal + background test statistic
nTrials  = 500
nTrials1 = 500
nTrials2 = 500

#Read value for this line
M5 = 6000
k  = 750 



#Mass and scale Lists
massList = numpy.linspace(minMass, maxMass, nBins)

#Generate fix background
BFix = SB.Bfix(massList)
print("Background evaluated.")
print("")

#Grid of toy experiments
print("Generating average map...")
print("")

toyExpGrid1, _                 = LSign.ToyExperimentGridMaker(BFix, mergeX, mergeY, nTrials)
wMinGrid, wMaxGrid, pvalueGrid = LSign.pvalueGridMaker(toyExpGrid1)



#Define model for region finder
print('Loading model...')

sizeX = 60
sizeY = 56
dimWX = [sizeX, sizeY]
dimWY = [sizeX, sizeY]

model1 = Sequential()

model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same', input_shape=(dimWX[0], dimWX[1], 1)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(Flatten())
model1.add(Dense(40, activation='elu'))
model1.add(Dense(20, activation='elu')) #Encoded layer
model1.add(Dense(40, activation='elu'))
model1.add(Dense(int(dimWX[0]*dimWX[1]/16*128), activation='elu'))
model1.add(Reshape((int(dimWX[0]/4), int(dimWX[1]/4), 128)))
model1.add(Conv2D(128,  kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(UpSampling2D((2 ,2)))
model1.add(Conv2D(128,  kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(UpSampling2D((2 ,2)))
model1.add(Conv2D(1, kernel_size=(3, 3), activation='elu', padding='same'))

model1.compile(optimizer='adam',loss='mean_squared_error')
model1.load_weights(nameModelFile)



#Apply machine learning
print("Starting new point...")
print("Evaluating signal...")

#Generate fix signal
SFix, _ = SB.Sfix(M5, k, massList)
print("Signal evaluated.")
print("")



#Generate trial background + signal
#Initialize variables
print("Beginnning toy experiment of background + signal")

statTest1 = []

for i in range(0, nTrials1):
    
  #Generate random binned event
  eventsTrial1 = SB.SBmain(BFix, SFix, 1, 1)

  #Do wavelet transform
  cwtmatBSTrial1Norm, _, _ =  WT.WaveletTransform(eventsTrial1, mergeX, mergeY, Cone, NconeR)
  cwtmatBSTrial1NormdAv    = -numpy.log(LSign.pvalueCalc(wMinGrid, wMaxGrid, pvalueGrid, cwtmatBSTrial1Norm))
  cwtmatBSTrial1Norm       =  cwtmatBSTrial1NormdAv[0:sizeX, 0:sizeY]

  #Format for predictions
  cwtmatBSTrial1Flat      = numpy.ravel(cwtmatBSTrial1Norm)
  cwtmatBSTrial1Formatted = numpy.reshape(cwtmatBSTrial1Flat,(1, sizeX, sizeY, 1))

  #Make prediction
  temp1 = model1.predict(cwtmatBSTrial1Formatted)[0]

  #Apply test statistic
  rIn  = numpy.ravel(cwtmatBSTrial1Norm)
  rOut = numpy.ravel(temp1)
  ts   = TestS.testStatCalc5(rIn, rOut)

  #Calculate test statistic FT
  statTest1.append(ts)

print("Toy experiments of background + signal done.")
print("")



#Generate trial background
#Initialize variables
print("Beginnning toy experiment of background only")
statTest2 = []

for i in range(0, nTrials2):
    
  #Generate random binned event
  eventsTrial2 = SB.SBmain(BFix, SFix, 1, 0)

  #Do wavelet transform
  cwtmatBSTrial2Norm, _, _ =  WT.WaveletTransform(eventsTrial2, mergeX, mergeY, Cone, NconeR)
  cwtmatBSTrial2NormdAv    = -numpy.log(LSign.pvalueCalc(wMinGrid, wMaxGrid, pvalueGrid, cwtmatBSTrial2Norm))
  cwtmatBSTrial2Norm       =  cwtmatBSTrial2NormdAv[0:sizeX, 0:sizeY]

  #Format for predictions
  cwtmatBSTrial2Flat      = numpy.ravel(cwtmatBSTrial2Norm)
  cwtmatBSTrial2Formatted = numpy.reshape(cwtmatBSTrial2Flat,(1, sizeX, sizeY, 1))

  #Make prediction
  temp2 = model1.predict(cwtmatBSTrial2Formatted)[0]

  #Apply test statistic
  rIn  = numpy.ravel(cwtmatBSTrial2Norm)
  rOut = numpy.ravel(temp2)
  ts   = TestS.testStatCalc5(rIn, rOut)

  #Calculate test statistic FT
  statTest2.append(ts)

print("Toy experiments of background only done.")
print("")



#Distribution of statistical significance
significanceDistribution = []

for TS1 in statTest1:
    
  #Initialize variable
  temp = 0
    
  #Loop over background events
  for TS2 in statTest2:
    if TS2 >= TS1:
      temp += 1/nTrials2
    
  #Append to list
  significanceDistribution.append(temp)



#Average
average = sum(significanceDistribution)/len(significanceDistribution)

#Median
t1     = numpy.sort(significanceDistribution)
median = t1[int(numpy.floor(0.5*len(significanceDistribution)))]



#Return results
print("Final results: \n")
print("Expected average significance: ")
print(str(average) + "\n")
print("Expected median significance: ")
print(str(median) + "\n")



#Print to file
fout = open("ExampleC2_out.txt","w")
fout.write("Results\n\n")
fout.write("M5:\n")
fout.write(str(M5) + "\n")
fout.write("k:\n")
fout.write(str(k) + "\n")
fout.write("Expected average significance:\n")
fout.write(str(average) + "\n")
fout.write("Expected median significance:\n")
fout.write(str(median) + "\n")
fout.close()




