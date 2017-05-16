
import tensorflow as tf
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

DIRNAME = 'data/patientsWithOutcomes/npArraysLogTransformed'
MODEL_NAME = "simpleNet"
SAMPLE_ARCHITECTURE = [[64,3,1],[3,3],[64,3,1],[3,3],[1024]]

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
# Start the session
myNet = Utils.ConvolutionalNetwork(SAMPLE_ARCHITECTURE,[CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2) # Set up a new session
myInterface = myNet.CreateTFInterface()
myInterface.StartSession()

# Print results for initial, 'random', network:
# Feed some data into it
cores = GetAllCoreIds(DIRNAME) # Get the cores in the directory
inputArr, outcomeArr = LoadData(DIRNAME,cores[:10],"_Log") # Load all images to RAM
_,testError,testAccuracy = myInterface.Test(inputArr,outcomeArr)
print("Test Classification Error of Random Network: "+str(testAccuracy))

# Load the network
new_saver = tf.train.import_meta_graph("trainingResults/simpleNet/" + MODEL_NAME+".meta") # Instantiate the saver
new_saver.restore(myInterface.sess, "trainingResults/simpleNet/" + MODEL_NAME) # Loads the model


# Feed the same data into the loaded network
_,testError,testAccuracy = myInterface.Test(inputArr,outcomeArr)
print("Test Classification Error of Loaded Network: "+str(testAccuracy))

# Compute Gradient for all images
for coreId in cores:
    inputArr, outcomeArr = LoadData(DIRNAME,[coreId],"_Log") # Load all images to RAM
    # reformattedInput = np.zeros((1,CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS))
    # reformattedInput[0,:,:,:] = inputArr[0]
    # reformattedOutput = np.reshape(outcomeArr[0],(1,2))

    reformattedInput = inputArr
    reformattedOutput = outcomeArr
    gradients,netOutPut = myInterface.sess.run([myNet.GradsTF,myNet.OutputLayerTF],feed_dict={'inputsPL'+":0":reformattedInput,'outMasksPL'+":0":reformattedOutput})

    # if np.argmax(netOutPut)==np.argmax(reformattedOutput): print("Predicted Correctly with "+str(np.max(netOutPut*100))+" % Confidence")
    print("Core: " + str(coreId) + " - Prediction for the Correct Label: "+str(netOutPut[0][np.argmax(reformattedOutput)]*100)+" % Confidence")

    gradImg = gradients[0][0]

    # Normalise
    for i in range(gradImg.shape[2]):
        maxVal = np.max(gradImg[:,:,i])
        gradImg[:,:,i] = gradImg[:,:,i]/maxVal

    # gradImg[gradImg<0.5] = 0

    fName = "trainingResults/simpleNet/gradients/core_" + str(coreId) + "_Gradients.pdf"
    title = str(coreId) + " " + str(netOutPut) + " " + str(outcomeArr)
    plotToPdf(gradImg,fName,title)



# Do some deep dreaming
coreId = 12
# inputArr, outcomeArr = LoadData(DIRNAME,[coreId],"_Log") # Load all images to RAM
inputArr = np.zeros((1,CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS)) #np.random.uniform(size=(1,CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS)) + 0
outcomeArr = np.zeros((1,2))
outcomeArr[0,:] = [0,1]

# plt.imshow(inputArr[0,:,:,1])
stepSize = 10000000
for step in range(10):
    gradients,netOutPut = myInterface.sess.run([myNet.GradsTF,myNet.OutputLayerTF],feed_dict={'inputsPL'+":0":inputArr,'outMasksPL'+":0":outcomeArr})
    inputArr += gradients[0]*(stepSize / (np.abs(gradients[0]).mean()+1e-7))
    print("Step: "+str(step)+" - Network Outputs: "+str(netOutPut)+" Mean Step Size: "+str((gradients[0]*(stepSize / (np.abs(gradients[0]).mean()+1e-7))).mean()))


# plt.imshow(inputArr[0,:,:,1])
# fName = "trainingResults/simpleNet/core_" + str(coreId) + "_Dream.pdf"
fName = "trainingResults/simpleNet/ran_dream" + str(1) + "_Dream.pdf"
# title = str(coreId)
plotToPdf(inputArr[0,:,:,:],fName,title)

#
# initImg = np.random.uniform(size=(224,224,3)) + 100.0


# Look at the gradients
# reformattedInput = np.zeros((1,CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS))
# reformattedInput[0,:,:,:] = inputArr[0]
# reformattedOutput = np.reshape(outcomeArr[0],(1,2))
#
# gradients,netOutPut = myInterface.sess.run([myNet.GradsTF,myNet.OutputLayerTF],feed_dict={'inputsPL'+":0":reformattedInput,'outMasksPL'+":0":reformattedOutput})
#
# if np.argmax(netOutPut)==np.argmax(reformattedOutput): print("Predicted Correctly with "+str(np.max(netOutPut*100))+" % Confidence")
#
# gradImg = gradients[0][0]
#
# # Normalise
# for i in range(gradImg.shape[2]):
#     maxVal = np.max(gradImg[:,:,i])
#     gradImg[:,:,i] = gradImg[:,:,i]/maxVal
#
# gradImg[gradImg<0.5] = 0
#
# fName = "trainingResults/" + MODEL_NAME + "_Gradients.pdf"
# plotToPdf(gradImg,fName,cores[0])
#

#
# g = tf.gradients(myNet.OutputLayerTF,inputArr[0])
#
# outMask = np.zeros((CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS))
#
# myNet.GradsTF(myNet.OutputLayerTF,inputArr[0])
#
# myInterface.sess.run(myNet.GradsTF,feed_dict={'inputsPL'+":0":inputArr[0:1],'outputsPL'+":0":outcomeArr[0:1],'dropoutPL'+":0":1})
# ,'outputsPL'+":0":outcomeArr[0:1]

# dum = myInterface2.sess.run(myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
# print(dum[0][0,0,0,0]) # Now this holds the same value as the saved network

# myInterface2.sess.run(myNet.AccuracyTF,feed_dict={'inputsPL'+":0":inputArr[:5],'outputsPL'+":0":outcomeArr[:5],'dropoutPL'+":0":1}) # Run the saved CNN on an image
