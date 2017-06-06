# ========================================================================
# Sript to reorder the columns in the text files so that they're in a consistent
# order.
# ========================================================================

import pandas as pd
import numpy as np
import re
import os
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
dataDir = "../data/patientsWithOutcomes/txtFiles/"
outDir = "../data/patientsWithOutcomes/txtFiles_reordered/"
coreVec = GetAllCoreIds(dataDir)

nameToColDict = {'88Sr-SrBCK(Sr88Di)': 0, '101Ru-RR101(Ru101Di)': 1, '102Ru-RR102(Ru102Di)': 2,
                 '115In-AvantiLipid(In115Di)': 3, '134Xe-XeBCK(Xe134Di)': 4, '141Pr-CD196(Pr141Di)': 5,
                 '142Nd-CD19(Nd142Di)': 6,
                 '143Nd-Vimentin(Nd143Di)': 7, '145Nd-CD163(Nd145Di)': 8, '147Sm-CD20(Sm147Di)': 9,
                 '148Nd-CD16(Nd148Di)': 10, '149Sm-CD25(Sm149Di)': 11, '150Nd-p53(Nd150Di)': 12,
                 '151Eu-CD134(Eu151Di)': 13,
                 '152Sm-CD45(Sm152Di)': 14, '153Eu-CD44s(Eu153Di)': 15, '154Gd-CD14(Gd154Di)': 16,
                 '155Gd-FoxP3(Gd155Di)': 17, '156Gd-CD4(Gd156Di)': 18, '158Gd-E-cadherin(Gd158Di)': 19,
                 '159Tb-p21(Tb159Di)': 20,
                 '161Dy-CD152(Dy161Di)': 21, '162Dy-CD8a(Dy162Di)': 22, '164Dy-CD11b(Dy164Di)': 23,
                 '165Ho-Beta-catenin(Ho165Di)': 24, '166Er-B7-H4(Er166Di)': 25, '168Er-Ki67(Er168Di)': 26,
                 '169Tm-CollagenI(Tm169Di)': 27,
                 '170Er-CD3(Er170Di)': 28, '171Yb-CD68(Yb171Di)': 29, '172Yb-PD-L2(Yb172Di)': 30,
                 '173Yb-B7-H3(Yb173Di)': 31, '174Yb-HLA-DR(Yb174Di)': 32, '175Lu-pS6(Lu175Di)': 33,
                 '176Yb-HistoneH3(Yb176Di)': 34,
                 '191Ir-DNA191(Ir191Di)': 35, '193Ir-DNA193(Ir193Di)': 36, '168Er-B7-H4(Er168Di)': 25}

markerLabelsVec = ['SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin',
                   'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3',
                   'CD4', 'E.cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta.catenin', 'B7.H4', 'Ki67',
                   'CollagenI', 'CD3', 'CD68', 'PD.L2', 'B7.H3', 'HLA.DR', 'pS6', 'HistoneH3', 'DNA191',
                   'DNA193']

for coreId in coreVec:
    print("Rearranging Core: "+str(coreId))

    # Load the data
    coreFName = dataDir + "core_" + str(coreId) + ".txt"
    data = pd.read_table(coreFName)
    markerList = [f for f in list(data) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z','Unnamed: 0']]
    data_reordered = pd.DataFrame(index=np.arange(0, data.shape[0]), columns=tuple(list(data)[0:6])+tuple(markerLabelsVec))
    data_reordered.iloc[:,0:6] = data.iloc[:,0:6]

    for marker in markerList:
        if marker != '166Er-BCL-2(Er166Di)':
            targetCol = nameToColDict[marker]+6
            data_reordered.iloc[:,targetCol] = data[marker]

    # Save data
    outName = outDir + "core_" + str(coreId) + ".txt"
    data_reordered.to_csv(outName, sep='\t',index=False)
