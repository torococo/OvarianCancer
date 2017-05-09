# ========================================================================
# Script to process the raw original txt files. --- Most of this is historic ---
# Checks through the files, and relabels them into a standard format 'coreXXX.txt'.
# This script also has the code I used to first check the consistency of the data we've been
# provided with. So, there are sections which check if all the cores for which I have
# patient outcome are actually physically located in the files given to us, and if not all
# are provided, this script will also tell which cores are missing. In addition, there are some
# snippets I used to check for duplicates in both the original txt files and in the assignment to patients
# in the excel file.
# ========================================================================
# import pandas as pd
import numpy as np
import DataProc as dtp
import re as re
import shutil
import os
import shlex
import pipes

# ========================================================================
# Function to get coreID from file name via regex matching
def getCoreId(fName):
    regex = re.compile(r' \d+ ') # In all file names the core id is separated by blank spaces
    coreId = regex.findall(fName)
    return int(coreId[0])

# Function to get patient outcome from coreId
def getPatientOutcome(coreId, outcomeArr):
    ptSensitive = outcomeArr[np.where(outcomeArr[:,0] == coreId),1]
    if ptSensitive.size==0:
        ptSensitiveVec = np.ones(2)*(-1) # Return -1 as to indicate no outcome available
    else:
        ptSensitiveVec = np.zeros(2)
        ptSensitiveVec[int(ptSensitive[0][0])] = 1
    return ptSensitiveVec

# ========================================================================
# Find the files for the patients with outcomes and check that we have all.
# Report which ones we're missing
rawFileNames = dtp.GetFileNames('data/originalTxtFiles/') # Get file names
outcomeArr  = np.genfromtxt('data/patientsWithOutcomes/coreToSensitivityMap.csv', delimiter=',') # Get Array with patient outcomes

fetchedCoresVec = []
coresWithOutcome =0
for i,fName in enumerate(rawFileNames):

    # Extract coreID
    coreId = getCoreId(fName)

    # Find out if patient was platinum sensitive
    ptSensitiveVec = getPatientOutcome(coreId, outcomeArr)

    # Check for duplicates
    if coreId in fetchedCoresVec:print("duplicate core found, ID:"+coreId)

    # Count the number images for which we have outcomes
    if coreId in outcomeArr[:,0]:
        coresWithOutcome+=1
        fetchedCoresVec.append(coreId)

    # Rename the file
    print('Moving Core '+str(coreId)+', Original File Name: ' + str(fName))
    outFName = "data/txtFilesRenamed/core_"+str(coreId)+".txt"
    os.system ("cp %s %s" % (pipes.quote(fName), outFName))

print('========================================================================')
print('Total Number of files in directory: '+str(len(rawFileNames)))
print('Missing: '+str(131-coresWithOutcome)+' cores from our cohort')
print('The missing cores are:')
for coreId in outcomeArr[1:,0]:
    if coreId not in fetchedCoresVec:
        print(coreId)
print('========================================================================')

# ========================================================================
# Move files from our cohorts to separate directory ('PatientsWithOutcomes') and rename them into a consistent format
rawFileNames = dtp.GetFileNames('data/originalTxtFiles/') # Get file names
outcomeArr  = np.genfromtxt('data/patientsWithOutcomes/coreToSensitivityMap.csv', delimiter=',') # Get Array with patient outcomes

print('========================================================================')
print('Move Files...')

for i,fName in enumerate(rawFileNames):

    # Extract coreID
    coreId = getCoreId(fName)

    print(coreId)
    # Move and rename the file
    if coreId in outcomeArr[1:,0]:
        print('Moving Core '+str(coreId))
        outFName = "data/patientsWithOutcomes/txtFiles/core_"+str(coreId)+".txt"
        os.system ("cp %s %s" % (pipes.quote(fName), outFName))
        shutil.copy(fName, outFName)

print('Done.')
print('========================================================================')

# Check which cores from our cohort are missing
movedFileNames = dtp.GetFileNames('data/patientsWithOutcomes/txtFiles/') # Get file names
movedCores = []

regex = re.compile(r'\d+') # In all file names the core id is separated by blank spaces
for fName in movedFileNames:
    coreId = int(regex.findall(fName)[0])
    movedCores.append(coreId)
    # print(coreId)

for coreId in outcomeArr[1:,0]:
    if coreId not in movedCores:
        print(coreId)

for coreId in movedCores:
    if coreId not in outcomeArr[1:,0]:
        print(coreId)

# ========================================================================
# Code for other stuff
# Looking for duplicate cores
# m = np.zeros_like(fetchedCoresVec, dtype=bool)
# m[np.unique(fetchedCoresVec, return_index=True)[1]] = True
# print(fetchedCoresVec[~m])
#
# coresFound = 0
# for coreId in outcomeArr[1:,0]:
#     if coreId in fetchedCoresVec:
#         coresFound +=1
#         ptSensitiveVec = getPatientOutcome(coreId, outcomeArr)
#         if ptSensitiveVec[0] > -1:
#             coreInSet += 1
#         else:
#             print(ptSensitiveVec, coreId)
#
# print coresFound
# #

# Check for duplicates in outcomeArr
# seen = []
# for coreId in outcomeArr[1:,0]:
#     if int(coreId) not in seen:
#         seen.append(coreId)
#     else:
#         print('Duplicate found! Core: '+str(coreId))

# print(coreInSet)
    # print(getPatientOutcome(coreId,outcomeArr))
    # Count the number images for which we have outcomes (as an independent check)

#
# print(coresFound)