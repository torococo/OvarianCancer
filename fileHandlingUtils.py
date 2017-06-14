import numpy as np
import re
import os

# ========================================================
# Function to get coreID from file name via regex matching
def GetCoreId(fName):
    regex = re.compile(r'\d+')
    coreId = regex.findall(fName)
    return int(coreId[0])

# ========================================================
# Function to return a lits of all the core Ids in the directory
def GetAllCoreIds(dirName,fType=".txt"):
    dataFiles = [fName for fName in os.listdir(dirName) if fType in fName] # Get file names
    ret = []

    for coreFile in dataFiles:
        ret.append(GetCoreId(coreFile))

    return np.array(ret)

# ========================================================