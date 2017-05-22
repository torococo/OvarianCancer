# ========================================================
# Extract the weight matrices from the linear classifiers and generate summary statistics.
# ========================================================
import tensorflow as tf
import Utils
import cnnUtils
import numpy as np
import DataProc as dtp
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Library to save plots as pdfs
# ===================== Parameters ================================

CANONICAL_DIMS = 600
N_STAINS = 37
CORE_TO_OUTCOME_MAP = np.genfromtxt(os.path.expanduser('~')+'/Desktop/Fluidigm_Project/ovarianCancer/data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv', delimiter=',') # Get Array with patient outcomes

pathToData = os.path.expanduser('~') + '/Desktop/Fluidigm_Project/ovarianCancer/data/patientsWithOutcomes/npArraysLogTransformed'
pathToModels = os.path.expanduser('~')+"/Desktop/Fluidigm_Project/ovarianCancer/linearClassifier/naiveLinearClassifier"
model = "naiveClassifier_BatchSize_25"

# ======================== Main code ============================
# Start the session
myNet = Utils.LinearClassifier([CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2,bRegularise=False,regScale=0.1)
myInterface = myNet.CreateTFInterface()
myInterface.StartSession()

# Print results for initial, 'random', network:
# Feed some data into it
cores = cnnUtils.GetAllCoreIds(pathToData) # Get the cores in the directory
inputArr, outcomeArr = cnnUtils.LoadData(pathToData, cores[:10], CORE_TO_OUTCOME_MAP, "_Log") # Load all images to RAM
_,testError,testAccuracy = myInterface.Test(inputArr,outcomeArr)
print("Test Classification Error of Random Network: "+str(testAccuracy))

for setId in range(1,11):
    print("Extracting the weight matrix for SetId: "+str(setId))

    # Randomly initialise and check that the current network is different from the one before to makes sure this worked.
    myInterface.sess.run(myNet.InitVarsTF)
    weightMats = myInterface.sess.run(myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    print("w1[0,0] before loading:" + str(weightMats[0][0,0]))

    # Load the network
    modelPath = pathToModels + "/" + model + "/models/naiveLinearClassifier" + str(setId) + "/naiveLinearClassifier" + str(setId)
    new_saver = tf.train.import_meta_graph(modelPath+".meta") # Instantiate the saver
    new_saver.restore(myInterface.sess, modelPath) # Loads the model


    # Feed the same data into the loaded network
    _,testError,testAccuracy = myInterface.Test(inputArr,outcomeArr)
    print("Test Classification Error of Loaded Network: "+str(testAccuracy))


    # Extract the weights
    weightMats = myInterface.sess.run(myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    print("w1[0,0] after loading:" + str(weightMats[0][0,0]))

    wMatRes = weightMats[0][:,0].reshape((600,600,37))
    wMatSens = weightMats[0][:,1].reshape((600,600,37))

    # Check the size is right
    # weightMats[0].shape
    # weightMats[1].shape

    # Save them
    fName0 = model+"_"+str(setId)+"_weightMatResistant"
    cnnUtils.plotToPdf(wMatRes,fName=fName0+".pdf") # pdf
    # np.savetxt(fName0 + ".csv", wMatRes, fmt='%10.16f', delimiter=',', newline='\n') # csv

    fName1 = model+"_"+str(setId)+"_weightMatSensitive.pdf"
    cnnUtils.plotToPdf(wMatSens,fName=fName1)

    # Collect summary statistics on the different layers
    meanWeightPerStainArr_Res = np.zeros((37,3))
    meanWeightPerStainArr_Sens = np.zeros((37,3))
    for stain in range(37):
        meanWeightPerStainArr_Res[stain,:] = [stain, np.mean(wMatRes[:,:,stain]), 0]
        meanWeightPerStainArr_Sens[stain,:] = [stain, np.mean(wMatSens[:,:,stain]), 1]

    summaryArr = np.concatenate((meanWeightPerStainArr_Res, meanWeightPerStainArr_Sens), axis=0)

    # Save the results
    fNameSummary = model+"_"+str(setId)+"_summaries"
    np.savetxt(fNameSummary + ".csv", summaryArr, fmt='%.5f', delimiter=',', newline='\n') # csv

    print("-----------------------------------")