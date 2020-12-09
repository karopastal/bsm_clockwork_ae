import math, numpy, scipy, random, os, sys

import v8.LSign as LSign
import v8.WT as WT
import v8.SBLD as SB

import keras
import sklearn
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers import Conv2D

from keras.layers import AveragePooling2D,\
    BatchNormalization, \
    GlobalAveragePooling2D, \
    MaxPooling2D, \
    Reshape, \
    UpSampling2D

from keras.models import Sequential

# Binning
minMass = 150
maxMass = 2700
nBins = 1275
mergeX = 20
mergeY = 2

# Cone of influence
Cone = False
NconeR = 2

# File to save weights and biases
nameModelFile = "ZExampleC_WandB.txt"

# Number of trials for p-value map and number of training backgrounds for neural network
nTrials = 2000
nTraining = 5000

# Number of threads to use
NumberThreads = 8

# Number of epochs to train
NumberEpochs = 100

print("Setting read")
print("")

# Mass and scale Lists
massList = numpy.linspace(minMass, maxMass, nBins)

# Generate fix background
BFix = SB.Bfix(massList)
print("Background evaluated.")
print("")

# Grid of toy experiments
print("Generating average map...")
print("")

toyExpGrid1, _ = LSign.ToyExperimentGridMaker(BFix, mergeX, mergeY, nTrials)
wMinGrid, wMaxGrid, pvalueGrid = LSign.pvalueGridMaker(toyExpGrid1)

# Training neural network
print("Beginning generation of training events...")
print("")

# Generate events for training and testing
X_train = []
Y_train = []

for i in range(0, nTraining):

    # Generate random binned events
    eventsInt = SB.SBmain(BFix, BFix, 1, 0)

    # Do wavelet transform
    cwtmatBSIntNorm, _, _ = WT.WaveletTransform(eventsInt, mergeX, mergeY, Cone, NconeR)
    cwtmatBSIntNormdAv = -numpy.log(LSign.pvalueCalc(wMinGrid, wMaxGrid, pvalueGrid, cwtmatBSIntNorm))

    # Reshape matrix
    sizeX = 60
    sizeY = 56
    cwtmatBSIntNorm = cwtmatBSIntNormdAv[0:sizeX, 0:sizeY]

    # Append results
    X_train.append(cwtmatBSIntNorm)
    Y_train.append(cwtmatBSIntNorm)

    # Make report
    if i % 100 == 0:
        print(i)

# Convert to numpy array
X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)

# Dimension info
dimWX = (X_train.shape[1], X_train.shape[2])
dimWY = (Y_train.shape[1], Y_train.shape[2])

# Format
X_train = numpy.ravel(X_train)
Y_train = numpy.ravel(Y_train)

X_train = numpy.reshape(X_train, (nTraining, dimWX[0], dimWX[1], 1))
Y_train = numpy.reshape(Y_train, (nTraining, dimWX[0], dimWX[1], 1))

PATH_V8_DATASET = 'data/dataset/v8/training.npy'

numpy.save(PATH_V8_DATASET, X_train)

print('Generation of training events completed.')
print("")

# Define model for autoencoder
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NumberThreads
config.inter_op_parallelism_threads = NumberThreads
sess = tf.Session(config=config)

model1 = Sequential()

# Add model layers
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same', input_shape=(dimWX[0], dimWX[1], 1)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(Flatten())
model1.add(Dense(40, activation='elu'))
model1.add(Dense(20, activation='elu'))  # Encoded layer
model1.add(Dense(40, activation='elu'))
model1.add(Dense(int(dimWX[0] * dimWX[1] / 16 * 128), activation='elu'))
model1.add(Reshape((int(dimWX[0] / 4), int(dimWX[1] / 4), 128)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(UpSampling2D((2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
model1.add(UpSampling2D((2, 2)))
model1.add(Conv2D(1, kernel_size=(3, 3), activation='elu', padding='same'))

# Compile
model1.compile(optimizer='adam', loss='mean_squared_error')

# Checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(nameModelFile, verbose=1, monitor='val_loss', save_best_only=True,
                                             mode='auto')

# Train model
train_history = model1.fit(X_train, Y_train, batch_size=200, epochs=NumberEpochs, validation_split=0.2,
                           callbacks=[checkpoint])

PATH_V8_MODEL = 'data/models/v8'

os.makedirs(PATH_V8_MODEL, exist_ok=True)
model1.save(PATH_V8_MODEL + "/autoencoder.h5")
model1.load_weights(PATH_V8_MODEL + "/weights.txt")
