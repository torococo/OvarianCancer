# ========================================================================
# Script to compute the mean expression levels of each stain for each cell.
# ========================================================================
import pandas as pd
import numpy as np
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
dataDir = "segmented_images/"
outName = "cellData.txt"
outcomeArr = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',')  # Get Array with patient outcomes
coreVec = GetAllCoreIds(dataDir)

# Initialise data frame to hold cell information
print("Initialise data frame")
coreFName = dataDir+"core_"+str(1)+"_labeled.txt"
data = pd.read_table(coreFName)
markerList = [f for f in list(data) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z', 'cell_id','Unnamed: 0']]
nCells = len(pd.unique(data.loc[:,"cell_id"]))
cellDF = pd.DataFrame(index=np.arange(0, nCells*len(coreVec)), columns=('CellId','PtSnty','CoreId','CellSize')+tuple(markerList))
cellNumberDF = pd.DataFrame(index=np.arange(0, len(coreVec)), columns=('CoreId','PtSnty','nCells'))

for i,coreId in enumerate(coreVec):
    print("Processing Core: " + str(coreId))

    # Load the data
    coreFName = dataDir+"core_"+str(coreId)+"_labeled.txt"
    data = pd.read_table(coreFName)
    markerList = [f for f in list(data) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z', 'cell_id', 'Unnamed: 0']]

    # Obtain the platinum sensitivity for this core
    ptSnty = outcomeArr[np.where(outcomeArr[:,0] == coreId),1][0][0]
    cellNumberDF.loc[i] = [coreId,ptSnty,len(pd.unique(data.loc[:,"cell_id"]))]

    # Compute the mean expression for each cell
    for j,cellId in enumerate(pd.unique(data.loc[:,"cell_id"])):
        # print("Processing cell: "+str(cellId))
        cellMeanExpr = data[data["cell_id"]==cellId]
        cellSize = cellMeanExpr.shape[0]
        cellMeanExpr = cellMeanExpr[lambda df: markerList]
        cellMeanExpr = cellMeanExpr.mean()

        cellDF.loc[j] = [cellId,ptSnty,coreId,cellSize]+list(cellMeanExpr.values)

    # Save the results
    cellDF.to_csv(outName, sep='\t')
    cellNumberDF.to_csv("cellNumberDF.txt", sep='\t')