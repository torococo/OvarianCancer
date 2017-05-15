
import numpy as np
import cnnUtils

MAX_CONV_LAYERS = 5
FILTER_NUMBERS_ARRAY = [32,64,128,256]
FILTER_SIZES_ARRAY = [3,5,7,11]
MAXPOOL_SIZES_ARRAY = [2,3,5]
MAX_FC_LAYERS = 2
FC_SIZES_ARRAY = [512,1024,2048]

N_ARCHITECTURES = 100
DATA_DIR = 'data/patientsWithOutcomes/npArraysLogTransformed'
CORE_TO_OUTCOME_MAP = np.genfromtxt('data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv', delimiter=',') # Get Array with patient outcomes
N_EPOCHS = 50
BATCH_SIZE = 10


# Load and partition the data
print("============== CNN Training ============================")
print("Preparing Data...")
cores = cnnUtils.GetAllCoreIds(DATA_DIR) # Get the cores in the directory

# Split into input and output data
trainingSet, validSet, testingSet = cnnUtils.PartitionData(cores,abs=[100,10,12],bShuffle=True) # Returns cores ids for each set

for i in range(N_ARCHITECTURES):
    # Generate an architecture
    architecture,signature = cnnUtils.GenRandomArchitecture(MAX_CONV_LAYERS, FILTER_NUMBERS_ARRAY, FILTER_SIZES_ARRAY, MAXPOOL_SIZES_ARRAY, MAX_FC_LAYERS, FC_SIZES_ARRAY)

    # Train it
    testError,testAccuracy = cnnUtils.TrainAndTestCNN(architecture,DATA_DIR,trainingSet,validSet,testingSet,CORE_TO_OUTCOME_MAP,N_EPOCHS,BATCH_SIZE,modelName="Model "+str(i))

    # Save the results
    np.save("results",np.array([i,testError,testAccuracy]))