# ========================================================
# Script to collect summary statistics on the stains across all cores.
# ========================================================
import numpy as np
import DataProc as dtp
import cnnUtils


cores = cnnUtils.GetAllCoreIds("../data/patientsWithOutcomes/npArraysRaw/")
outcomeArr = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',')  # Get Array with patient outcomes
statsArr = np.zeros((len(cores)*37,5))
totCounter = 0

for i,coreId in enumerate(cores):
    print(str(i + 1) + " of " + str(len(cores)) + " - Computing Summary Statistics for core " + str(coreId))

    # Load the data
    image = np.load("../data/patientsWithOutcomes/npArraysRaw/core_" + str(coreId) + ".npy")
    outcome = outcomeArr[np.where(outcomeArr[:,0] == coreId),1][0][0]

    # Get the summary stats
    for stain in range(37):

        meanExpr = np.mean(image[:,:,stain])
        totExpr = np.linalg.norm(image[:,:,stain],ord=1)

        statsArr[totCounter,:] = [stain,meanExpr,totExpr,outcome,coreId]
        totCounter += 1

# Save the results
np.savetxt("rawStainSummaries.csv",statsArr,delimiter=",")