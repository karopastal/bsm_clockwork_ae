##################
#   RF trainer   #
##################

#Explanations
# - This code trains a neural network to discover anomalies.

#Import
print("Starting program...")
print("")
print("Importing files...")

#Import standard
import math, numpy, scipy, random, os

#Import custom
import LSign, WT
import SBLD as SB

#Machine learning
import keras, sklearn
import tensorflow as tf
from keras.layers            import Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers            import Convolution2D, Conv1D, Conv2D, Conv2DTranspose
from keras.layers            import AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.merge      import add, concatenate
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

#Simulation settings
nTrials            = 500
nTrainingPerPoints = 100

#Number of threads to use
NumberThreads = 8

#Fraction of background
fracBackground = 0.5



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



#Training
nTrainingPoints = 0
X_train = []
Y_train = []

print('Generating training data...')

#Generate signals
with open('ZExampleD_Signals.txt') as fin:
  for line in fin:

    #Generate fix signal
    data = line.split()
    SFix = []
    nTrainingPoints += 1

    for i in range(2, len(data)):
      SFix.append(float(data[i]))

    #Local maps
    cwtmatIntPNorm, _, _ = WT.WaveletTransform(SFix, mergeX, mergeY, Cone, NconeR)
    cwtmatIntSTemp       = cwtmatIntPNorm/listavNorm1
    localMap             = cwtmatIntSTemp

    for j in range(0, nTrainingPerPoints):

      #Generate random binned events
      temp = random.random()

      if temp > fracBackground:
        aS = 1
      else:
        aS = 0

      eventsInt = SB.SBmain(BFix, SFix, 1, aS)

      #Do wavelet transform
      cwtmatBSIntNorm, _, _ = WT.WaveletTransform(eventsInt, mergeX, mergeY, Cone, NconeR)
      cwtmatIntPTemp        = cwtmatBSIntNorm/listavNorm1
      inputX_train          = cwtmatIntPTemp

      if aS==1:
        outputY_train = localMap.copy()
      else:
        outputY_train = numpy.full((listavNorm1.shape[0], listavNorm1.shape[1]), 0)

      #Append results
      X_train.append(inputX_train)
      Y_train.append(outputY_train)

#Convert to numpy array
X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)

#Dimension info
dimWX = (X_train.shape[1], X_train.shape[2])
dimWY = (Y_train.shape[1], Y_train.shape[2])

#Format
X_train = numpy.ravel(X_train)
Y_train = numpy.ravel(Y_train)

X_train = numpy.reshape(X_train, (nTrainingPoints*nTrainingPerPoints, dimWX[0], dimWX[1], 1))
Y_train = numpy.reshape(Y_train, (nTrainingPoints*nTrainingPerPoints, dimWY[0]*dimWY[1]))

#Inform completion
print('Events generated')



#Define model for region finder
print('Starting training')

config = tf.ConfigProto()
config.intra_op_parallelism_threads = NumberThreads
config.inter_op_parallelism_threads = NumberThreads
sess = tf.Session(config=config)

model2 = Sequential()

#add model layers
model2.add(Conv2D(32,  kernel_size=5, activation='softplus', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64,  kernel_size=5, activation='softplus'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128,  kernel_size=5, activation='softplus'))
model2.add(Flatten())
model2.add(Dense(5000, activation='softplus'))
model2.add(Dense(Y_train.shape[1], activation='softplus'))

#Compile
model2.compile(optimizer='adam',loss='mean_squared_error')

#Checkpoint
checkpoint = keras.callbacks.ModelCheckpoint('ZExampleD_WandB.txt', verbose=1, monitor='val_loss', save_best_only=True, mode='auto') 

#Train model
train_history = model2.fit(X_train, Y_train, batch_size=1000, epochs=100, validation_split=0.2, callbacks=[checkpoint])

#Load best weights
model2.load_weights('ZExampleD_WandB.txt')
print('Training done')




