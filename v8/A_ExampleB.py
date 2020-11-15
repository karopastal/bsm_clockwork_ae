####################
#   WT classifer   #
####################

#Explanations
# - This code uses a classifier on the wavelet transform to discover a known signal.

#Import
print("Starting program...")
print("")
print("Importing files...")

#Import standard
import math, numpy, scipy, random, os, sys

#Import custom
import Basic, WT
import SBLD as SB

#Machine learning
import keras, sklearn
import tensorflow as tf
from keras.callbacks         import EarlyStopping
from keras.layers            import Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers            import Convolution2D, Conv1D, Conv2D, Conv2DTranspose
from keras.layers            import AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.merge      import add
from keras.models            import load_model, Model, Sequential
from sklearn.model_selection import train_test_split
from keras                   import backend as K

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
nameModelFile = "ZExampleB_WandB.txt"

#Simulation settings
nTraining = 500
nTrials1  = 500
nTrials2  = 500

#Parameters of the signal
M5 = 7000
k  = 750

#Number of threads to use
NumberThreads = 8

print("Setting read")
print("")



#Mass and scale Lists
massList = numpy.linspace(minMass, maxMass, nBins)

#Generate fix signal
SFix, _ = SB.Sfix(M5, k, massList)
BFix    = SB.Bfix(massList)
print("Signal evaluated.")
print("")



#Training neural network
print("Beginning generation of training events...")
print("")

#Generate events for training and testing
X_train = []
Y_train = []

for i in range(0, nTraining):
    
  #Generate random parameters for signal
  aStemp = random.randint(0, 1)

  #Generate random binned events
  eventsInt = SB.SBmain(BFix, SFix, 1, aStemp)

  #Do wavelet transform
  cwtmatBSIntNorm, _, _ = WT.WaveletTransform(eventsInt, mergeX, mergeY, Cone, NconeR)

  #Append results
  X_train.append(cwtmatBSIntNorm)
  Y_train.append([aStemp])
    
  #Make report
  if i%100 == 0:
    print(i)

#Convert to numpy array
X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)

#Dimension info
dimWX = (X_train.shape[1], X_train.shape[2])
dimWY = (Y_train.shape[1])

#Format
X_train = numpy.ravel(X_train)
Y_train = numpy.ravel(Y_train)

X_train = numpy.reshape(X_train, (nTraining, dimWX[0], dimWX[1], 1))
Y_train = numpy.reshape(Y_train, (nTraining, dimWY))

print('Generation of training events completed.')
print("")



#Define model for classifier
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NumberThreads
config.inter_op_parallelism_threads = NumberThreads
sess = tf.Session(config=config)

model1 = Sequential()

#Add model layers
model1.add(Conv2D(4,  kernel_size=3, activation='elu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(8,  kernel_size=3, activation='sigmoid'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(16, kernel_size=3, activation='sigmoid'))
model1.add(Flatten())
model1.add(Dense(200, activation='sigmoid'))
model1.add(Dense(100, activation='sigmoid'))
model1.add(Dense(Y_train.shape[1], activation='sigmoid'))

#Compile
model1.compile(optimizer='adam', loss='binary_crossentropy')

#Checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(nameModelFile, verbose=1, monitor='val_loss', save_best_only=True, mode='auto') 

#Train model
train_history = model1.fit(X_train, Y_train, batch_size=200, epochs=500, validation_split=0.2, callbacks=[checkpoint])

#Load best weights
model1.load_weights(nameModelFile)



#Generate trial background + signal
#Initialize variables
print("Beginning toy experiments of background + signal")
print("")

statTest1   = []

for i in range(0, nTrials1):
    
  #Generate random binned events
  eventsTrial1 = SB.SBmain(BFix, SFix, 1, 1)

  #Do wavelet transform
  cwtmatBTrial1Norm, _, _ = WT.WaveletTransform(eventsTrial1, mergeX, mergeY, Cone, NconeR)
 
  #Format for predictions
  cwtmatBTrial1Flat      = numpy.ravel(cwtmatBTrial1Norm)
  cwtmatBTrial1Formatted = numpy.reshape(cwtmatBTrial1Flat,(1, X_train.shape[1], X_train.shape[2], 1))
    
  #Make prediction
  predTrial1 = model1.predict(cwtmatBTrial1Formatted)[0]

  #Calculate test statistic
  statTest1.append(predTrial1[0])

print("Toy experiments of background + signal done.")
print("")



#Generate trial background + signal
#Initialize variables
print("Beginning toy experiments of background only")
print("")

statTest2   = []

for i in range(0, nTrials1):
    
  #Generate random binned events
  eventsTrial2 = SB.SBmain(BFix, SFix, 1, 0)

  #Do wavelet transform
  cwtmatBTrial2Norm, _, _ = WT.WaveletTransform(eventsTrial2, mergeX, mergeY, Cone, NconeR)
 
  #Format for predictions
  cwtmatBTrial2Flat      = numpy.ravel(cwtmatBTrial2Norm)
  cwtmatBTrial2Formatted = numpy.reshape(cwtmatBTrial2Flat,(1, X_train.shape[1], X_train.shape[2], 1))
    
  #Make prediction
  predTrial2 = model1.predict(cwtmatBTrial2Formatted)[0]

  #Calculate test statistic
  statTest2.append(predTrial2[0])

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
fout = open("ExampleB_out.txt","w")
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




