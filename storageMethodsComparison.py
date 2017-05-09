# ========================================================================
# Compare different possible ways of storing the images.
# ========================================================================
import numpy as np
import DataProc as dtp
import pickle
import time

# ========================================================================
# Function to get patient outcome from coreId
def getPatientId(coreId, outcomeArr):
    patientId = outcomeArr[np.where(outcomeArr[:,0] == coreId),2]
    if patientId.size==0:
        patientId = -1 # Return -1 as to indicate no patient id
    else:
        patientId = int(patientId[0][0])
    return patientId

# ========================================================================
# Start comparison
rawFileNames = dtp.GetFileNames('Data/PatientsWithOutcomes/TxtFiles/') # Get file names
outcomeArr  = np.genfromtxt('Data/coreToSensitivityMap.csv', delimiter=',') # Get Array with patient outcomes

nTries = 10

print("# ========================================================================")
print("Number of tries:"+str(nTries))
print("# ----------------------------------------")
print("Pickle...")
start = time.time()
for _ in range(nTries):
    data = pickle.load(open("Data/PatientsWithOutcomes/RawNpArrays_Pickled/core1.p","rb"))
    label = data['PtSensitive']
end = time.time()
avg_time = (end - start)/float(nTries)
print("Average time to load: "+str(avg_time)+"s")

print("# ----------------------------------------")
print("Numpy.save()...")
start = time.time()
outcomeArr  = np.genfromtxt('Data/coreToSensitivityMap.csv', delimiter=',') # Get Array with patient outcomes
for _ in range(nTries):
    data = np.load("Data/PatientsWithOutcomes/RawNpArrays/core1.npy")
    label = getPatientId(1,outcomeArr)
end = time.time()
avg_time = (end - start)/float(nTries)
print("Average time to load: "+str(avg_time)+"s")
print("# ========================================================================")
