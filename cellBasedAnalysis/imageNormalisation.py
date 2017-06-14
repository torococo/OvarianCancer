# ========================================================================
# Script to standardise the images using one set of constants for each stain
# ========================================================================
# When running via ipython
# import os
# os.chdir("dataProcessing")
# ========================================================================
from sklearn import preprocessing
import numpy as np
import sys
sys.path.insert(0, '..')
import fileHandlingUtils as fhUtils
import pickle
import pandas as pd
import os

# ============================= Configure Script =========================
# Control which of the operations the script should carry out
DO_GENERATE_NORMALISER_OBJECTS = True

# ------------------------------------------------------------------------
fileDir = "../data/patientsWithOutcomes/txtFiles_reordered"
dst_dir = '../data/patientsWithOutcomes/txtFiles_Normed'
normObj_dir = '../data/patientsWithOutcomes/normaliserObjects'
# ============================= Main Code ================================
# Get all the coreIds to process
coreVec = fhUtils.GetAllCoreIds(fileDir,fType=".txt")

markerLabels = ['SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin',
                   'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3',
                   'CD4', 'E.cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta.catenin', 'B7.H4',
                   'CollagenI', 'CD3', 'CD68', 'PD.L2', 'B7.H3', 'HLA.DR', 'pS6', 'HistoneH3', 'DNA191',
                   'DNA193']

if (DO_GENERATE_NORMALISER_OBJECTS):
    # Set up environment
    allImgsArrDir = normObj_dir+"/tmp"
    # if not os.path.exists(normObj_dir):
    #     os.mkdir(normObj_dir)
    #     os.mkdir(allImgsArrDir)
    #
    # # Prepare arrays to hold cumulative values for each stain
    # for marker in markerLabels:
    #     allImgsArr = np.array([0])
    #     allImgsFName = allImgsArrDir+"/marker_"+marker
    #     np.save(allImgsFName,allImgsArr)
    #
    # # Generate the normaliser objects by accumulating the stains across all the images
    # for coreId in coreVec:
    #     # Load the data
    #     print("===============================")
    #     print("Loading core: " + str(coreId))
    #     print("===============================")
    #     coreFName = fileDir+"/core_"+str(coreId)+".txt"
    #     currIm = pd.read_table(coreFName)
    #
    #     # Extract each of the stains and add to cumulative array
    #     for marker in markerLabels:
    #         print("Processing Marker: "+marker)
    #         # Load the cumulative array
    #         allImgsFName = allImgsArrDir+"/marker_"+marker+".npy"
    #         allImgsArr = np.load(allImgsFName)
    #
    #         # Extract the desired stain
    #         currStain = currIm.loc[:, marker]
    #
    #         # Append to main array
    #         allImgsArr = np.append(allImgsArr, currStain)
    #
    #         # GIve some output to make sure it's working properly
    #         print("Mean: "+str(np.mean(currStain))+", Size of CumArr: "+str(allImgsArr.shape))
    #
    #         # Save
    #         np.save(allImgsFName, allImgsArr)

    # Generate the normalising constants as a normaliser object
    print("===============================")
    print("Generate Normalisers")
    print("===============================")
    for marker in markerLabels:
        print("Normalising Marker: "+marker)
        # Load the cumulative array
        allImgsFName = allImgsArrDir+"/marker_"+marker+".npy"
        allImgsArr = np.load(allImgsFName)

        # Normalise
        normaliser = preprocessing.Normalizer().fit(allImgsArr.reshape(1,-1))

        # Save it
        normFName = normObj_dir+"/"+marker+"_Normaliser"
        fileHandle = open(normFName, 'wb')
        pickle.dump(normaliser,fileHandle)
        fileHandle.close()

