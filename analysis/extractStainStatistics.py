# ========================================================
# Script to collect summary statistics on the stains across all cores.
# ========================================================
import numpy as np
import DataProc as dtp
import cnnUtils
import matplotlib.pyplot as plt

# ========================================================
# Define which statistics to calculate
SUMMARY_STATS = False
CORRELATION_MATRIX = True
# ========================================================
# import os
# os.chdir("analysis")
cores = cnnUtils.GetAllCoreIds("../data/patientsWithOutcomes/npArraysRaw/")
outcomeArr = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',')  # Get Array with patient outcomes
statsArr = np.zeros((len(cores)*37,5))
corrValArr = np.zeros((len(cores),int(36*37/2.+2)))
totCounter = 0
markerLabelsVec = ['SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin', 'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3', 'CD4', 'E-cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta-catenin', 'B7-H4', 'Ki67', 'CollagenI', 'CD3', 'CD68', 'PD-L2', 'B7-H3', 'HLA-DR', 'pS6', 'HistoneH3', 'DNA191', 'DNA193']


for i,coreId in enumerate(cores):
    print(str(i + 1) + " of " + str(len(cores)) + " - Computing Summary Statistics for core " + str(coreId))

    # Load the data
    image = np.load("../data/patientsWithOutcomes/npArraysRaw/core_" + str(coreId) + ".npy")
    outcome = outcomeArr[np.where(outcomeArr[:,0] == coreId),1][0][0]

    if(SUMMARY_STATS):
        # Get the summary stats
        for stain in range(37):

            meanExpr = np.mean(image[:,:,stain])
            totExpr = np.linalg.norm(image[:,:,stain],ord=1)

            statsArr[totCounter,:] = [stain,meanExpr,totExpr,outcome,coreId]
            totCounter += 1

    if(CORRELATION_MATRIX):
        # Compute the covariance matrix
        imFlattend = image.reshape((image.shape[0]*image.shape[1],image.shape[2]))

        corrMat = np.corrcoef(imFlattend,rowvar=0)

        # Reshape into a 37x38/2+2 row vector
        corrValArr[i,0:2] = [coreId,outcome]
        k = 2
        for ii in range(37):
            for jj in range(ii):
                # print(markerLabelsVec[ii],markerLabelsVec[jj])
                # print(ii,jj)
                corrValArr[i,k] = corrMat[ii,jj]
                k += 1

# Save the results
if (SUMMARY_STATS):
    np.savetxt("rawStainSummaries.csv",statsArr,delimiter=",")
if (CORRELATION_MATRIX):
    np.savetxt("pixelCorrelations.csv",corrValArr,delimiter=",")