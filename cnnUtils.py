# ========================================================
# Utilities for constructing the fluidigm CNN
# ========================================================

# import tensorflow as tf
# import DataProc as dtp
import time
import Utils
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Library to save plots as pdfs

CANONICAL_DIMS=600
N_STAINS=37

# ========================================================
# Function to get coreID from file name via regex matching
def GetCoreId(fName):
    regex = regex = re.compile(r'\d+')
    coreId = regex.findall(fName)
    return int(coreId[0])

# ========================================================
# Function to return a lits of all the core Ids in the directory
def GetAllCoreIds(dirName):
    dataFiles = [fName for fName in os.listdir(dirName) if '.npy' in fName] # Get file names
    ret = []

    for coreFile in dataFiles:
        ret.append(GetCoreId(coreFile))

    return np.array(ret)

# ========================================================
# Function to get patient outcome from coreId
def GetPatientOutcome(coreId, outcomeArr):
    ptSensitive = outcomeArr[np.where(outcomeArr[:,0] == coreId),1]
    if ptSensitive.size==0:
        ptSensitiveVec = np.ones(2)*(-1) # Return -1 as to indicate no outcome available
    else:
        ptSensitiveVec = np.zeros(2)
        ptSensitiveVec[int(ptSensitive[0][0])] = 1
    return ptSensitiveVec

# ========================================================
# Function to load all the data in the directory 'dirName' into a single numpy array
def LoadData(dirName,coreIDs,coreToOutcomeMap,transformName=None):
    inputArr = np.zeros((len(coreIDs),CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS)) # Array to hold all images
    outcomeArr = np.zeros((len(coreIDs),2)) # Array to hold the outcome (as 1x2 row vectors)

    for i,coreId in enumerate(coreIDs):

        # Load the image
        inputArr[i,:,:,:] = np.load(dirName+"/core_"+str(coreId)+transformName+".npy")

        # Load the outcome
        outcomeArr[i,:] = GetPatientOutcome(coreId, coreToOutcomeMap)

    return inputArr, outcomeArr

# ========================================================
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
# Function to plot core to pdf
def plotToPdf(image,fName=None,CoreId=None):

    # Generate filename
    if fName==None and CoreId != None:
        fName = "CoreId"+str(CoreId)+".pdf"
    elif fName==None and CoreId != None:
        raise Exception("No filename for pdf provided!")

    # Open pdf and figure
    pp = PdfPages(fName)
    f=plt.figure(1)

    # Plot and save
    markerLabels = ['88Sr-SrBCK(Sr88Di)', '101Ru-RR101(Ru101Di)', '102Ru-RR102(Ru102Di)', '115In-AvantiLipid(In115Di)', '134Xe-XeBCK(Xe134Di)', '141Pr-CD196(Pr141Di)', '142Nd-CD19(Nd142Di)', '143Nd-Vimentin(Nd143Di)', '145Nd-CD163(Nd145Di)', '147Sm-CD20(Sm147Di)', '148Nd-CD16(Nd148Di)', '149Sm-CD25(Sm149Di)', '150Nd-p53(Nd150Di)', '151Eu-CD134(Eu151Di)', '152Sm-CD45(Sm152Di)', '153Eu-CD44s(Eu153Di)', '154Gd-CD14(Gd154Di)', '155Gd-FoxP3(Gd155Di)', '156Gd-CD4(Gd156Di)', '158Gd-E-cadherin(Gd158Di)', '159Tb-p21(Tb159Di)', '161Dy-CD152(Dy161Di)', '162Dy-CD8a(Dy162Di)', '164Dy-CD11b(Dy164Di)', '165Ho-Beta-catenin(Ho165Di)', '166Er-B7-H4(Er166Di)', '168Er-Ki67(Er168Di)', '169Tm-CollagenI(Tm169Di)', '170Er-CD3(Er170Di)', '171Yb-CD68(Yb171Di)', '172Yb-PD-L2(Yb172Di)', '173Yb-B7-H3(Yb173Di)', '174Yb-HLA-DR(Yb174Di)', '175Lu-pS6(Lu175Di)', '176Yb-HistoneH3(Yb176Di)', '191Ir-DNA191(Ir191Di)', '193Ir-DNA193(Ir193Di)']

    for stain in range(37):
        plt.imshow(image[:,:,stain])
        plt.title("Core: "+str(CoreId)+", Label:"+markerLabels[stain])
        pp.savefig(f)

    # Close the files and clean up
    pp.close()
    plt.close()

# ========================================================
def GenRandomArchitecture(MAX_CONV_LAYERS, FILTER_NUMBERS_ARRAY, FILTER_SIZES_ARRAY, MAXPOOL_SIZES_ARRAY, MAX_FC_LAYERS, FC_SIZES_ARRAY):
    # Generate a random architecture
    architecture = []
    signature = np.zeros((1+MAX_CONV_LAYERS*3+MAX_FC_LAYERS,1),dtype=np.int16)
    sigIdx = 0

    # 1. Choose the number of convolutional layers
    nConvLayers = np.random.randint(0,MAX_CONV_LAYERS)
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
        architecture.append(np.array([nFilters,filterSize,1]))

        # Potentially add a Maxpool layer
        addMaxpool = bool(np.random.binomial(1,0.5))

        maxpoolSize = 0
        if addMaxpool:
            # Choose its size
            maxpoolSize = MAXPOOL_SIZES_ARRAY[np.random.randint(0,len(MAXPOOL_SIZES_ARRAY))]

            # Add it
            architecture.append(np.array([maxpoolSize,maxpoolSize]))

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

        architecture.append(np.array([fcSize]))

    return architecture, signature

# ========================================================
# Function to train a given CNN on given fluidigm data
def TrainAndTestNetwork(networkHandle, dataDir, trainingSet, validSet, testingSet, coreToOutcomeMap, nEpochs, batchSize, dropOutProb=0.5, outDir="trainingResults/",modelName="CNN", verbose=True, saveModelInterval=5):
    # Set up the interface
    if verbose: print("Starting the Interface...")
    myInterface = networkHandle.CreateTFInterface()

    # Run it
    myInterface.StartSession()

    # Train the network
    if verbose:
        print("Start Training:")
        print("---------------------------")

    nBatches = len(trainingSet)/batchSize # Number of batches per epoch
    resArr = np.zeros((nEpochs,5)) # List with the mean error at each epoch (full iteration through all the training data

    for i in range(nEpochs):

        # Partition training data into batches
        batchVec,_ = Utils.GenBatchSet(trainingSet,trainingSet,batchSize)

        startEpoch = time.time() # Record time for epoch

        # Train for one epoch
        meanTimePerBatch = 0
        err = 0
        for iBatch, batch in enumerate(batchVec):
            startBatch = time.time() # Record time taken by batch
            if verbose: print("Epoch "+str(i)+" of "+str(nEpochs)+" ------------------ Batch "+str(iBatch)) #+":"+str(batch))
            inputArr, outcomeArr = LoadData(dataDir,batch,coreToOutcomeMap,"_Log") # Load all images to RAM
            trainOut,crossEntropy,trainRes = myInterface.Train(inputArr,outcomeArr,dropOutProb)
            print("Logits: "+str(trainOut)+" - Correct Labels: " +str(outcomeArr)+" - Entropy: "+str(crossEntropy))
            err = err+crossEntropy
            endBatch = time.time()
            meanTimePerBatch += (endBatch-startBatch)

        # Estimate avg time a batch takes (for profiling)
        meanTimePerBatch = meanTimePerBatch/nBatches

        # Validate
        inputArr, outcomeArr = LoadData(dataDir,validSet,coreToOutcomeMap,"_Log") # Load all images to RAM
        _,_,validAccuracy = myInterface.Test(inputArr,outcomeArr)

        endEpoch = time.time()

        # The GenAndRunBatchTraining() method returns a an array with the error for each image in the training set.
        # To generate a summary statistics from this, take the mean across all images in the training set.
        err = err/nBatches

        # Report performance for this epoch
        if verbose: print("Epoch "+str(i)+" of "+str(nEpochs)+" - Error: "+str(err)+" - Validation Classification Error: "+str(validAccuracy*100)+"%"+" - Time Taken: "+str(endEpoch-startEpoch)+"s - Avg Time per Batch: "+str(meanTimePerBatch)+"s")

        # Save the results to file
        resArr[i,:] = [i, err, validAccuracy, endEpoch-startEpoch, meanTimePerBatch]
        np.savetxt(outDir + "/trainingError_" + modelName + ".csv", resArr, fmt='%10.16f', delimiter=',', newline='\n') # Save the training errors

        # Save the model
        if i%saveModelInterval == 0: networkHandle.Saver.save(myInterface.sess, outDir+ "/" + modelName) #myInterface.SaveGraph("trainingResults/" + MODEL_NAME) #



    # Test it
    if verbose:
        print("---------------------------")
        print("Perform Test")
    inputArr, outcomeArr = LoadData(dataDir,testingSet, coreToOutcomeMap,"_Log") # Load all images to RAM
    _,testError,testAccuracy = myInterface.Test(inputArr,outcomeArr)
    if verbose: print("Achieved: "+str(testAccuracy*100)+"% Accuracy")

    if verbose:
        print("Done.")
        print("==========================================")

    return testError,testAccuracy
