# ========================================================================
# During the segmentation step I noticed that some of the markers do no appear
# in all cores and that the columns for the different markers are switched as well.
# Here we asses the severity of this miss-match.
# ========================================================================
import pandas as pd
import numpy as np
import re
import os

from collections import defaultdict

# os.chdir("dataProcessing")
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
coreVec = GetAllCoreIds(dataDir)

d = defaultdict(int)
posDict = defaultdict(list)
for coreId in coreVec:
    print("Analysing Core: "+str(coreId))
    # Initialise data frame to hold cell information
    coreFName = dataDir+"core_"+str(coreId)+"_labeled.txt"
    data = pd.read_table(coreFName)
    markerList = [f for f in list(data) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z', 'cell_id','Unnamed: 0']]

    for marker in markerList:
        d[marker]+=1
        posDict[marker].append(markerList.index(marker))


with open('markerFreq.csv', 'w') as writer:
    for k in d.keys():
        writer.write(k+","+str(d[k])+"\n")

with open('markerColFreq.csv', 'w') as writer:
    for k in posDict.keys():
        writer.write(k+","+str(set(posDict[k]))+"\n")