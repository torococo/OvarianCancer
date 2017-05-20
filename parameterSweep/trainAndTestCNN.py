# ========================================================
# Linear classifier on the fluidigm images.
# This is meant to serve as the 'simplest base case network' and to
# see how important different stains are.
# ========================================================
import numpy as np
import cnnUtils_Cluster
import Utils_Cluster
import sys
import os
import shutil

# ==================== Parameters for the CNN =============================
archFName = sys.argv[1]
archId = int(sys.argv[2])
batchSize = int(sys.argv[3])
nEpochs = int(sys.argv[4])
outDir = sys.argv[5]

Model_NAME = "architecture"
SAVE_MODEL_INTERVAL = 1
DATA_DIR = os.path.expanduser('~')+"/fluidigmProject/data/patientsWithOutcomes/npArraysLogTransformed"
CORE_TO_OUTCOME_MAP = np.genfromtxt(os.path.expanduser('~')+"/fluidigmProject/data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv", delimiter=',')  # Get Array with patient outcomes
DROPOUT_PROB = 0.5
CANONICAL_DIMS = 600
N_STAINS = 37
# ==================== Main Code =============================
# Load the architecture
archSpecList = np.load(archFName, encoding = 'latin1')
architecture = archSpecList[0]
signature = archSpecList[1]

print(archFName)
print(architecture)
print(signature)
print(str(archId))
print(str(batchSize))
print(str(nEpochs))
print(str(outDir))

# Build the network
myNet = Utils_Cluster.ConvolutionalNetwork(architecture,[CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2)

# Load the data (just the core indeces)
dataPath = os.path.expanduser('~')+"/fluidigmProject/parameterSweep/data/"+"Set"+str(1)+".npy"
data = np.load(dataPath, encoding = 'latin1')
trainingSet = data[0]
validSet = data[1]
testingSet = np.load(os.path.expanduser('~')+"/fluidigmProject/parameterSweep/data/testingSet.npy", encoding = 'latin1')

# Prepare folder to save model in
modelDir = outDir+"/model/"
if not os.path.exists(modelDir):
    os.makedirs(modelDir)

# Train and test
testError,testAccuracy = cnnUtils_Cluster.TrainAndTestNetwork(myNet,DATA_DIR,trainingSet,validSet,testingSet,CORE_TO_OUTCOME_MAP,nEpochs,batchSize,dropOutProb=DROPOUT_PROB,outDirLogs=outDir+"/trainingLogFiles",outDirModel=modelDir,modelName=Model_NAME+str(archId))

# Record the results
performanceArr = np.array([archId,testError,testAccuracy])
performanceArr = np.append(performanceArr,signature).reshape((1,performanceArr.shape[0]))

# Save the results
np.savetxt(outDir+"/performance_" + Model_NAME + "_" + str(archId) + ".csv", performanceArr, fmt='%10.16f', delimiter=',', newline='\n')
