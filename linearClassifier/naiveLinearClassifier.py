# ========================================================
# Linear classifier on the fluidigm images.
# This is meant to serve as the 'simplest base case network' and to
# see how important different stains are.
# ========================================================
import numpy as np
import cnnUtils
import Utils
import sys
import os
import shutil

iSet = sys.argv[1]
BATCH_SIZE = int(sys.argv[2])
N_EPOCHS = int(sys.argv[3])
OUT_DIR = sys.argv[4]

modelName = "naiveLinearClassifier"
SAVE_MODEL_INTERVAL = 1
DATA_DIR = os.path.expanduser('~')+"/fluidigmProject/data/patientsWithOutcomes/npArraysLogTransformed"
CORE_TO_OUTCOME_MAP = np.genfromtxt(os.path.expanduser('~')+"/fluidigmProject/data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv", delimiter=',')  # Get Array with patient outcomes
DROPOUT_PROB = 0.5
CANONICAL_DIMS = 600
N_STAINS = 37

# ========================================================
# Define the architecture
myNet = Utils.LinearClassifier([CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2,bRegularise=False,regScale=0.1)

# Print progress
print("==========================================")
print("Training on Set: "+str(iSet))

# Train and test it
testingSet = np.load(os.path.expanduser('~')+"/fluidigmProject/linearClassifier/data/testingSet.npy", encoding = 'latin1')

# Load data partitions
dataPath = os.path.expanduser('~')+"/fluidigmProject/linearClassifier/data/"+"Set"+str(iSet)+".npy"
data = np.load(dataPath, encoding = 'latin1')
trainingSet = data[0]
validSet = data[1]

# Prepare folder to save model in
modelDir = OUT_DIR+"/models/"+modelName+str(iSet)
if not os.path.exists(modelDir):
    os.makedirs(modelDir)

# Train and test
testError,testAccuracy = cnnUtils.TrainAndTestNetwork(myNet,DATA_DIR,trainingSet,validSet,testingSet,CORE_TO_OUTCOME_MAP,N_EPOCHS,BATCH_SIZE,outDirLogs=OUT_DIR+"/trainingLogFiles",outDirModel=modelDir,modelName=modelName+str(iSet))

# Record the results
performanceArr = np.array([iSet,testError,testAccuracy],dtype=np.float32).reshape((1,3))

# Save the results
np.savetxt(OUT_DIR+"/performance_" + modelName + "_Set_" + str(iSet) + ".csv", performanceArr, fmt='%10.16f', delimiter=',', newline='\n')

# Clean up
OUT_DIR = os.path.expanduser('~') + "/fluidigmProject/linearClassifier/" + OUT_DIR
# shutil.move(OUT_DIR + "/trainingError_" + modelName+str(iSet) + ".csv", OUT_DIR + "/trainingLogFiles/trainingError_" + modelName+str(iSet) + ".csv")
# subprocess.call("mv " + OUT_DIR + "/*.qsub " + OUT_DIR + "/jobFiles/")
# subprocess.call("mv " + OUT_DIR + "/*"+str(iSet)+".meta " + OUT_DIR + "*"+str(iSet)+".index " + OUT_DIR + "*"+str(iSet)+".data* " + OUT_DIR + "checkpoint " + OUT_DIR + "/models/")

# subprocess.call("mv " + OUT_DIR + "/*trainingError* " + OUT_DIR + "/trainingLogFiles/")
