# ========================================================
# Linear classifier on the fluidigm images.
# This is meant to serve as the 'simplest base case network' and to
# see how important different stains are.
# ========================================================
import numpy as np
import cnnUtils
import Utils
import time

N_EPOCHS = 1
BATCH_SIZE = 25
DROPOUT_PROB = 0.5
CANONICAL_DIMS = 600
N_STAINS = 37

modelName = "naiveClassifier"
SAVE_MODEL_INTERVAL = 1
DATA_DIR = '../data/patientsWithOutcomes/npArraysLogTransformed'
OUT_DIR = modelName+"/"
CORE_TO_OUTCOME_MAP = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv', delimiter=',')  # Get Array with patient outcomes
VERBOSE = True
nTrainingPermutations = 2

# Define the architecture
myNet = Utils.LinearClassifier([CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2,bRegularise=False,regScale=0.1)

# Train and test it
performanceArr = np.zeros((nTrainingPermutations,3))
testingSet = np.load("data/testingSet.npy", encoding = 'latin1')

for iSet in range(1,nTrainingPermutations+1):

    # Print progress
    print("==========================================")
    print("Training on Set: "+str(iSet)+" of "+str(nTrainingPermutations+1))

    # Load data partitions
    dataDir = "data/"+"Set"+str(iSet)+".npy"
    data = np.load(dataDir, encoding = 'latin1')
    trainingSet = data[0]
    validSet = data[1]

    # Train and test
    testError,testAccuracy = cnnUtils.TrainAndTestNetwork(myNet,DATA_DIR,trainingSet,validSet,testingSet,CORE_TO_OUTCOME_MAP,N_EPOCHS,BATCH_SIZE,outDir=OUT_DIR,modelName=modelName+str(iSet))

    # Record the results
    performanceArr[iSet-1,:] = [iSet,testError,testAccuracy]

# Save the results
np.savetxt("performance_" + modelName + ".csv", performanceArr, fmt='%10.16f', delimiter=',', newline='\n')


#
#
#
#
# # Set up the interface
# myInterface = myNet.CreateTFInterface()
#
# # Run it
# myInterface.StartSession()
#
# # Train the network
# nBatches = len(trainingSet)/BATCH_SIZE # Number of batches per epoch
# resArr = np.zeros((N_EPOCHS,5)) # List with the mean error at each epoch (full iteration through all the training data
#
# for i in range(N_EPOCHS):
#     # Partition training data into batches
#     batchVec,_ = Utils.GenBatchSet(trainingSet,trainingSet,BATCH_SIZE)
#     startEpoch = time.time() # Record time for epoch
#     # Train for one epoch
#     meanTimePerBatch = 0
#     err = 0
#     for iBatch, batch in enumerate(batchVec):
#         startBatch = time.time() # Record time taken by batch
#         if VERBOSE: print("Epoch "+str(i)+" of "+str(N_EPOCHS)+" ------------------ Batch "+str(iBatch)) #+":"+str(batch))
#         inputArr, outcomeArr = cnnUtils.LoadData(DATA_DIR,batch,CORE_TO_OUTCOME_MAP,"_Log") # Load all images to RAM
#         trainOut,crossEntropy,trainRes = myInterface.Train(inputArr,outcomeArr,DROPOUT_PROB)
#         print("Logits: "+str(trainOut)+" - Correct Labels: " +str(outcomeArr)+" - Entropy: "+str(crossEntropy))
#         err = err+crossEntropy
#         endBatch = time.time()
#         meanTimePerBatch += (endBatch-startBatch)
#     # Estimate avg time a batch takes (for profiling)
#     meanTimePerBatch = meanTimePerBatch/nBatches
#     # Validate
#     inputArr, outcomeArr = cnnUtils.LoadData(DATA_DIR,validSet,CORE_TO_OUTCOME_MAP,"_Log") # Load all images to RAM
#     _,_,validAccuracy = myInterface.Test(inputArr,outcomeArr)
#     endEpoch = time.time()
#     err = err/nBatches
#     # Report performance for this epoch
#     if VERBOSE: print("Epoch "+str(i)+" of "+str(N_EPOCHS)+" - Error: "+str(err)+" - Validation Classification Error: "+str(validAccuracy*100)+"%"+" - Time Taken: "+str(endEpoch-startEpoch)+"s - Avg Time per Batch: "+str(meanTimePerBatch)+"s")
#     # Save the results to file
#     resArr[i,:] = [i, err, validAccuracy, endEpoch-startEpoch, meanTimePerBatch]
#     np.savetxt(OUT_DIR + "trainingError_" + modelName + ".csv", resArr, fmt='%10.16f', delimiter=',', newline='\n') # Save the training errors
#     # Save the model
#     if i%SAVE_MODEL_INTERVAL == 0: myNet.Saver.save(myInterface.sess, OUT_DIR + modelName) #myInterface.SaveGraph("trainingResults/" + MODEL_NAME) #
#
# # Test it
# if VERBOSE:
#     print("---------------------------")
#     print("Perform Test")
# inputArr, outcomeArr = cnnUtils.LoadData(DATA_DIR,testingSet, CORE_TO_OUTCOME_MAP,"_Log") # Load all images to RAM
# _,testError,testAccuracy = myInterface.Test(inputArr,outcomeArr)
# if VERBOSE: print("Achieved: "+str(testAccuracy*100)+"% Accuracy")
#
# if VERBOSE:
#     print("Done.")
#     print("==========================================")
#

# # Save the results
# np.save("results",np.array([i,testError,testAccuracy]))
# architecture = []  # No hidden layers
# modelName = "linearClassifier"

# Train it
# testError,testAccuracy = cnnUtils.TrainAndTestCNN(architecture,DATA_DIR,trainingSet,validSet,testingSet,CORE_TO_OUTCOME_MAP,N_EPOCHS,BATCH_SIZE,outDir=OUT_DIR,modelName=modelName,saveModelInterval=SAVE_MODEL_INTERVAL)
