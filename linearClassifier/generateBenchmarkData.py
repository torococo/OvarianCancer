# ========================================================
# Divide data for benchmarking. Have one test set to compare all
# algorithms across and partition the rest into training and validation.
# Do the partitioning for training and validation several times so to
# avoid bias from this partitioning.
# ========================================================
import numpy as np
import cnnUtils

DATA_DIR = '../data/patientsWithOutcomes/npArraysLogTransformed'
nInstances = 10

# Load and partition the data
cores = cnnUtils.GetAllCoreIds(DATA_DIR)  # Get the cores in the directory

# Extract a testing set
np.random.shuffle(cores)
testingSet = cores[:12]
np.save("testingSet",testingSet)
cores = cores[12:]

# Split into input and output data
for i in range(1,nInstances+1):
    trainingSet, validSet, _ = cnnUtils.PartitionData(cores,abs=[100,10,0],bShuffle=True)  # Returns cores ids for each set
    np.save("Set"+str(i),np.array([trainingSet,validSet]))