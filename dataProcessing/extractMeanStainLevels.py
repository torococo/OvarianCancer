# ========================================================
# Script to collect summary statistics on the stains across all cores using panda rather than numpy structures.
# ========================================================
import numpy as np
import DataProc as dtp
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

# ========================================================
# Function to get coreID from file name via regex matching
def GetCoreId(fName):
    regex = regex = re.compile(r'\d+')
    coreId = regex.findall(fName)
    return int(coreId[0])

# ========================================================
# Function to return a lits of all the core Ids in the directory
def GetAllCoreIds(dirName):
    dataFiles = [fName for fName in os.listdir(dirName) if '.txt' in fName] # Get file names
    ret = []

    for coreFile in dataFiles:
        ret.append(GetCoreId(coreFile))

    return np.array(ret)

# ========================================================
# Define which statistics to calculate
SUMMARY_STATS = True
CORRELATION_MATRIX = False
# ========================================================
# import os
# os.chdir("analysis")
dataDir = "../data/patientsWithOutcomes/txtFiles_reordered/"
coreVec = GetAllCoreIds(dataDir)
outcomeArr = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',')  # Get Array with patient outcomes
markerLabelsVec = ['SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin',
                   'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3',
                   'CD4', 'E.cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta.catenin', 'B7.H4', 'Ki67',
                   'CollagenI', 'CD3', 'CD68', 'PD.L2', 'B7.H3', 'HLA.DR', 'pS6', 'HistoneH3', 'DNA191',
                   'DNA193']
statsArr = pd.DataFrame(index=np.arange(0, len(coreVec)), columns=('CoreId','PtSnty')+tuple(markerLabelsVec))
# corrValArr = np.zeros((len(cores),int(36*37/2.+2)))

for i,coreId in enumerate(coreVec):
    print(str(i + 1) + " of " + str(len(coreVec)) + " - Computing Summary Statistics for core " + str(coreId))

    # Load the data
    coreFName = dataDir + "core_" + str(coreId) + ".txt"
    image = pd.read_table(coreFName)
    outcome = outcomeArr[np.where(outcomeArr[:,0] == coreId),1][0][0]

    if(SUMMARY_STATS):
        # Get the summary stats
        coreMeanExpr = image.mean()
        statsArr.loc[i,0:2] = [coreId,outcome]
        statsArr.loc[i,2:] = coreMeanExpr[6:]

    # Save the results
    if (SUMMARY_STATS):
        statsArr.to_csv("rawStainSummaries.csv", sep=',', index=False)