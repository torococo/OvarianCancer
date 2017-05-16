# ========================================================
# Proof of concept 2 layer CNN on the fluidigm data
# ========================================================

import tensorflow as tf
import time

import Utils
import numpy as np
import DataProc as dtp
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Library to save plots as pdfs

CANONICAL_DIMS = 600
N_STAINS = 37
CORE_TO_OUTCOME_MAP = np.genfromtxt('data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv', delimiter=',') # Get Array with patient outcomes

BATCH_SIZE = 20
N_TRAINING_STEPS = 250
DROPOUT_PROB = 0.5

DIRNAME = 'data/patientsWithOutcomes/npArraysLogTransformed'
MODEL_NAME = "dataAugment"
VERBOSE = True
SAVE_MODEL_INTERVAL = 1

# ========================================================
# Function to get coreID from file name via regex matching
def GetCoreId(fName):
    regex = regex = re.compile(r'\d+')
    coreId = regex.findall(fName)
    return int(coreId[0])

# Function to get patient outcome from coreId
def GetPatientOutcome(coreId, outcomeArr):
    ptSensitive = outcomeArr[np.where(outcomeArr[:,0] == coreId),1]
    if ptSensitive.size==0:
        ptSensitiveVec = np.ones(2)*(-1) # Return -1 as to indicate no outcome available
    else:
        ptSensitiveVec = np.zeros(2)
        ptSensitiveVec[int(ptSensitive[0][0])] = 1
    return ptSensitiveVec

# Function to return a lits of all the core Ids in the directory
def GetAllCoreIds(dirName):
    dataFiles = [fName for fName in os.listdir(dirName) if '.npy' in fName] # Get file names
    ret = []

    for coreFile in dataFiles:
        ret.append(GetCoreId(coreFile))

    return np.array(ret)

# Function to load all the data in the directory 'dirName' into a single numpy array
def LoadData(dirName,coreIDs,transformName=None):
    inputArr = np.zeros((len(coreIDs),CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS)) # Array to hold all images
    outcomeArr = np.zeros((len(coreIDs),2)) # Array to hold the outcome (as 1x2 row vectors)

    for i,coreId in enumerate(coreIDs):

        # Load the image
        inputArr[i,:,:,:] = np.load(dirName+"/core_"+str(coreId)+transformName+".npy")

        # Load the outcome
        outcomeArr[i,:] = GetPatientOutcome(coreId, CORE_TO_OUTCOME_MAP)

    return inputArr, outcomeArr

# Function to partition data into training, validation and testing
def PartitionData(coreIdxVec,fracs=[0.8,0,0.2],abs=None,bShuffle=False):
    if abs==None and sum(fracs) != 1: raise Exception("The relative partition sizes need to add up to 1")
    nCores = len(coreIdxVec)
    indices = np.arange(0,nCores)
    if bShuffle: np.random.shuffle(indices) # Reshuffle the images in the dataset so to assign them to the different groups randomly

    # Convert fractions to numbers
    if abs==None:
        nTraining = int(fracs[0]*nCores)
        nValid = int(fracs[1]*nCores)
        nTest = nCores - nTraining - nValid
    else:
        nTraining = abs[0]
        nValid = abs[1]
        nTest = abs[2]

    trainingSet = coreIdxVec[indices[0:nTraining]] #{'Cores': inputArr[indices[0:nTraining],:,:,:], 'Outcomes': outcomeArr[indices[0:nTraining],:,:]}
    validSet = coreIdxVec[indices[nTraining:(nTraining+nValid)]] #{'Cores': inputArr[indices[nTraining:(nTraining+nValid)],:,:,:], 'Outcomes': inputArr[indices[nTraining:(nTraining+nValid)],:,:]}
    testingSet = coreIdxVec[indices[(nTraining+nValid):]]
    return trainingSet, validSet, testingSet

# ========================================================
def prep_data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/255.0)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image

def data_augment(input_tensor):
    output_tensor = tf.map_fn(prep_data_augment, input_tensor)
    return output_tensor
# ========================================================
# Load the data
if VERBOSE:
    print("============== CNN Training ============================")
    print("Loading Data...")
cores = GetAllCoreIds(DIRNAME) # Get the cores in the directory
if VERBOSE: print("Done.")


# Split into input and output data
trainingSet, validSet, testingSet = PartitionData(cores,abs=[100,10,12],bShuffle=True) # Returns cores ids for each set

# Generate the CNN
# Convolutional layer [nFilters,dimensions,stride]
# Maxpool layer [dimensions,stride]
# Fully connected layer [nNeurons]
if VERBOSE: print("Defining CNN Architecture...")
sampleArchitecture = [[64,3,1],[3,3],[64,3,1],[3,3],[1024]]
# alexNet = [[96,11,4],[2,2],[256,5,1],[2,2],[384,3,1],[384,3,1],[384,3,1],[256,3,1],[4096],[4096]]
myNet = Utils.ConvolutionalNetwork(sampleArchitecture,[CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2)

# Set up the interface
if VERBOSE: print("Starting the Interface...")
myInterface = myNet.CreateTFInterface()

# Run it
myInterface.StartSession()

# Train the network
if VERBOSE:
    print("Start Training:")
    print("---------------------------")

nBatches = len(trainingSet)/BATCH_SIZE # Number of batches per epoch
resArr = np.zeros((N_TRAINING_STEPS,5)) # List with the mean error at each epoch (full iteration through all the training data


batchVec,_ = Utils.GenBatchSet(trainingSet,trainingSet,BATCH_SIZE)
batch = batchVec[0][:5]
inputArr, outcomeArr = LoadData(DIRNAME,batch,"_Log") # Load all images to RAM

augmentImg = data_augment(inputArr)
tf.concat(0, [inputArr, data_augment(inputArr)])

trainOut,crossEntropy,trainRes = myInterface.Train(inputArr,outcomeArr,DROPOUT_PROB)



