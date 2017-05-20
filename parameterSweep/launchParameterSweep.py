# ========================================================
# Script to perform a parameter sweep of different architectures on the cluster.
# ========================================================
import numpy as np
import os
from subprocess import Popen
import sys
# sys.path.append(os.path.expanduser('~')+"/fluidigmProject/")
# import cnnUtils_Cluster

# ==================== Parameters for the Sweep =============================
MAX_CONV_LAYERS = 3
FILTER_NUMBERS_ARRAY = [32,64,128,256]
FILTER_SIZES_ARRAY = [3,5,7,11]
MAXPOOL_SIZES_ARRAY = [2,3,5]
MAX_FC_LAYERS = 2
FC_SIZES_ARRAY = [256,512,1024]

N_ARCHITECTURES = 200
DATA_DIR = 'data/patientsWithOutcomes/npArraysLogTransformed'
CORE_TO_OUTCOME_MAP = np.genfromtxt(os.path.expanduser('~')+"/fluidigmProject/data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv", delimiter=',')  # Get Array with patient outcomes
N_EPOCHS = 25
BATCH_SIZE = 10
MEMORY_PER_NODE = 30# In Gb
WALL_TIME = "6:00:00"#
# ===================== Helper Functions ===============================
# Generate PBS file for cluster queueing system
def ClusterRun(nodes,ppn,mem,wallTime,jobName,runCommands,filePath):
  pbsStr="#!/bin/sh\n#PBS -l nodes="+str(nodes)+":ppn="+str(ppn)+",mem="+str(mem)+"gb"+",walltime="+wallTime+"\n"+runCommands
  runFile=open(filePath,'w')
  runFile.write(pbsStr)
  Popen("qsub "+"-N"+jobName+" "+filePath,shell=True)

# Generate a random Architecture
def GenRandomArchitecture(MAX_CONV_LAYERS, FILTER_NUMBERS_ARRAY, FILTER_SIZES_ARRAY, MAXPOOL_SIZES_ARRAY, MAX_FC_LAYERS, FC_SIZES_ARRAY):
    # Generate a random architecture
    architecture = []
    signature = np.zeros((1+MAX_CONV_LAYERS*3+MAX_FC_LAYERS,1),dtype=np.int16)
    sigIdx = 0

    # 1. Choose the number of convolutional layers
    nConvLayers = np.random.randint(1,MAX_CONV_LAYERS)
    signature[sigIdx] = nConvLayers
    sigIdx += 1

    # 2. Construct the convolutional layers
    for i in range(nConvLayers):
        # Choose the number of filters
        nFilters = FILTER_NUMBERS_ARRAY[np.random.randint(0,len(FILTER_NUMBERS_ARRAY))]
        signature[sigIdx] = nFilters
        sigIdx += 1

        # Choose their size
        filterSize = FILTER_SIZES_ARRAY[np.random.randint(0,len(FILTER_SIZES_ARRAY))]
        signature[sigIdx] = filterSize
        sigIdx += 1

        # Add it to the network
        architecture.append([nFilters,filterSize,1])

        # Potentially add a Maxpool layer
        addMaxpool = bool(np.random.binomial(1,0.5))

        maxpoolSize = 0
        if addMaxpool:
            # Choose its size
            maxpoolSize = MAXPOOL_SIZES_ARRAY[np.random.randint(0,len(MAXPOOL_SIZES_ARRAY))]

            # Add it
            architecture.append([maxpoolSize,maxpoolSize])

        signature[sigIdx] = maxpoolSize
        sigIdx += 1

    sigIdx = 1+MAX_CONV_LAYERS*3
    # 4. Choose the number of fully connected layers at the end
    nFCLayers = np.random.randint(1,MAX_FC_LAYERS)

    # 5. Add them
    for i in range(nFCLayers):
        # Choose their size
        fcSize = FC_SIZES_ARRAY[np.random.randint(0,len(FC_SIZES_ARRAY))]
        signature[sigIdx] = fcSize
        sigIdx += 1

        architecture.append([fcSize])

    return architecture, signature

def LaunchTrainAndTest(architecture,signature,archIdx,nEpochs,batchSize,pMemory=MEMORY_PER_NODE,wTime=WALL_TIME):
  # Define the run parameters
  scriptName = "cNNJob_"+str(archIdx)
  jobName =  "cNN"+str(archIdx)
  print("Launching architecture "+str(i)+" with "+str(signature[0][0])+" convolutional layers.")

  # Set up the environment
  outDir = "results/architecture_"+str(archIdx)
  if not os.path.exists(outDir):
    os.makedirs(outDir)
    os.makedirs(outDir+"/model")
    os.makedirs(outDir+"/trainingLogFiles")
    os.makedirs(outDir+"/jobFiles")

  # Save architecture to file so it can be passed to script
  archFName = outDir+"/architecture_archId"+str(archIdx)+".npy"
  np.save(archFName,np.array([architecture,signature]))
  np.savetxt(outDir+"/architecture_archId"+str(archIdx)+".txt",signature)

  # Prepare the pbs run file
  runCode="cd /home/80014744/fluidigmProject/parameterSweep/\n"
  runCode+="LD_LIBRARY_PATH=\"$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/\" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python` trainAndTestCNN.py "+archFName+" "+str(archIdx)+" "+str(batchSize)+" "+str(nEpochs)+" "+outDir
  ClusterRun(1,1,pMemory,wTime,jobName,runCode,"/home/80014744/fluidigmProject/parameterSweep/"+outDir+"/jobFiles/"+scriptName+".qsub")

# ==================== Main Loop ===========================
# Load and partition the data
print("============== CNN Parameter Sweeping ============================")

seen = []
architecture,signature = GenRandomArchitecture(MAX_CONV_LAYERS, FILTER_NUMBERS_ARRAY, FILTER_SIZES_ARRAY, MAXPOOL_SIZES_ARRAY, MAX_FC_LAYERS, FC_SIZES_ARRAY)
for i in range(N_ARCHITECTURES):
    while any([architecture==seenArch for seenArch in seen]):
        # Generate an architecture
        architecture,signature = GenRandomArchitecture(MAX_CONV_LAYERS, FILTER_NUMBERS_ARRAY, FILTER_SIZES_ARRAY, MAXPOOL_SIZES_ARRAY, MAX_FC_LAYERS, FC_SIZES_ARRAY)

    # Send it to a node to train and test
    LaunchTrainAndTest(architecture,signature,i,N_EPOCHS,BATCH_SIZE)

    # Add architecture to list of tested architectures
    seen.append(architecture)

print("============== Done ============================")

# testError,testAccuracy = cnnUtils.TrainAndTestCNN(architecture,DATA_DIR,trainingSet,validSet,testingSet,CORE_TO_OUTCOME_MAP,N_EPOCHS,BATCH_SIZE,modelName="Model "+str(i))
#
# # Save the results
# np.save("results",np.array([i,testError,testAccuracy]))