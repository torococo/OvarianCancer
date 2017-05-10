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

BATCH_SIZE = 4#10
N_TRAINING_STEPS = 1#1000
DROPOUT_PROB = 0.5

VERBOSE = True
DIRNAME = 'data/patientsWithOutcomes/npArraysLogTransformed'
MODEL_NAME = "alexNet"

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

    return ret

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
def PartitionData(coreIdxVec,partitionSet=[0.8,0.1,0.1],bShuffle=False):
    if sum(partitionSet) != 1: raise Exception("The relative partition sizes need to add up to 1")
    nCores = len(coreIdxVec)
    indices = np.arange(0,nCores)
    if bShuffle: np.random.shuffle(indices) # Reshuffle the images in the dataset so to assign them to the different groups randomly

    # Convert fractions to numbers
    nTraining = int(partitionSet[0]*nCores)
    nValid = int(partitionSet[1]*nCores)
    nTest = nCores - nTraining - nValid

    trainingSet = indices[0:nTraining] #{'Cores': inputArr[indices[0:nTraining],:,:,:], 'Outcomes': outcomeArr[indices[0:nTraining],:,:]}
    validSet = indices[nTraining:(nTraining+nValid)] #{'Cores': inputArr[indices[nTraining:(nTraining+nValid)],:,:,:], 'Outcomes': inputArr[indices[nTraining:(nTraining+nValid)],:,:]}
    testingSet = indices[(nTraining+nValid):]
    return trainingSet, validSet, testingSet

# ========================================================
# Load the data
if VERBOSE:
    print("============== CNN Training ============================")
    print("Loading Data...")
cores = GetAllCoreIds(DIRNAME)[:10] # Get the cores in the directory
inputArr, outcomeArr = LoadData(DIRNAME,cores,"_Log") # Load all images to RAM
if VERBOSE: print("Done.")


# Split into input and output data
trainingSet, validSet, testingSet = PartitionData(cores,partitionSet=[0.8,0,0.2],bShuffle=True) # Returns indices for each set

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

errVec = [] # List with the mean error at each epoch (full iteration through all the training data
for i in range(N_TRAINING_STEPS):

    # Train for one epoch
    err=myInterface.GenAndRunBatchTraining(inputArr[trainingSet],outcomeArr[trainingSet],batchSize=BATCH_SIZE,nInputSweeps=1,dropKeepProb=DROPOUT_PROB,verbose=False)

    # The GenAndRunBatchTraining() method returns a an array with the error for each image in the training set.
    # To generate a summary statistics from this, take the mean across all images in the training set.
    errVec.append(np.mean(err))
    if VERBOSE: print("Epoch "+str(i)+" of "+str(N_TRAINING_STEPS)+" - Error: "+str(errVec[i]))

    # Save the results
    np.savetxt("trainingResults/trainingError_" + MODEL_NAME + ".csv", errVec, fmt='%10.16f', delimiter=',', newline='\n') # Save the training errors


# Test it
if VERBOSE:
    print("---------------------------")
    print("Perform Test")
_,testError,testAccuracy = myInterface.Test(inputArr[testingSet],outcomeArr[testingSet])
if VERBOSE: print("Achieved: "+str(testAccuracy*100)+"% Accuracy")

# Save the results
if VERBOSE: print("Saving Results...")
np.savetxt("trainingResults/testResults_" + MODEL_NAME + ".csv", [testError[1], testAccuracy], fmt='%10.16f', delimiter=',', header="") # Save the training errors

# Save the model
myNet.Saver.save(myInterface.sess, "trainingResults/" + MODEL_NAME)

# Plot the training error
axs=Utils.GenAxs(1,1)
pp = PdfPages("trainingResults/" + MODEL_NAME + ".pdf")
Utils.PlotLine(axs[0],np.arange(len(errVec)),errVec,"r-")
plt.title("Training Error")
plt.ylabel('Error')
plt.xlabel('Epoch')

# Save to pdf
f = plt.gcf()
pp.savefig(f)
pp.close()
plt.close(f)

if VERBOSE:
    print("Done.")
    print("==========================================")


# ========================================================
# Other stuff

#  myInterface.sess.run((myNet.AccuracyTF),feed_dict={'inputsPL'+":0":inputArr[testingSet],'outputsPL'+":0":outcomeArr[testingSet],'dropoutPL'+":0":1}) #
 #
 # # Open pdf and figure
 #    f=plt.figure(1)
 #
 #    # Plot and save
 #    markerLabels = ['88Sr-SrBCK(Sr88Di)', '101Ru-RR101(Ru101Di)', '102Ru-RR102(Ru102Di)', '115In-AvantiLipid(In115Di)', '134Xe-XeBCK(Xe134Di)', '141Pr-CD196(Pr141Di)', '142Nd-CD19(Nd142Di)', '143Nd-Vimentin(Nd143Di)', '145Nd-CD163(Nd145Di)', '147Sm-CD20(Sm147Di)', '148Nd-CD16(Nd148Di)', '149Sm-CD25(Sm149Di)', '150Nd-p53(Nd150Di)', '151Eu-CD134(Eu151Di)', '152Sm-CD45(Sm152Di)', '153Eu-CD44s(Eu153Di)', '154Gd-CD14(Gd154Di)', '155Gd-FoxP3(Gd155Di)', '156Gd-CD4(Gd156Di)', '158Gd-E-cadherin(Gd158Di)', '159Tb-p21(Tb159Di)', '161Dy-CD152(Dy161Di)', '162Dy-CD8a(Dy162Di)', '164Dy-CD11b(Dy164Di)', '165Ho-Beta-catenin(Ho165Di)', '166Er-B7-H4(Er166Di)', '168Er-Ki67(Er168Di)', '169Tm-CollagenI(Tm169Di)', '170Er-CD3(Er170Di)', '171Yb-CD68(Yb171Di)', '172Yb-PD-L2(Yb172Di)', '173Yb-B7-H3(Yb173Di)', '174Yb-HLA-DR(Yb174Di)', '175Lu-pS6(Lu175Di)', '176Yb-HistoneH3(Yb176Di)', '191Ir-DNA191(Ir191Di)', '193Ir-DNA193(Ir193Di)']
 #
 #    for stain in range(37):
 #        plt.imshow(image[:,:,stain])
 #        plt.title("Core: "+str(coreId)+", Label:"+markerLabels[stain])
 #        pp.savefig(f)
 #
 #    # Close the files and clean up
 #    pp.close()
 #    plt.close()
 #



# testIn,testOut = LoadData(dirName,cores[1:2],"_Log")


# np.save("data/patientsWithOutcomes/logTransformedAll.npy",inputArr)
# trainingSet, testingSet = Utils.SeparateData(inputArr,outcomeArr,0.8,10,True)


