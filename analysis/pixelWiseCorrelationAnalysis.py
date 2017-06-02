# ========================================================
# Script to analyse and plot results related to the pixel wise correlations
# ========================================================
import numpy as np
import os
os.chdir("analysis")
import sys
sys.path.insert(0, '../dataProcessing')
import cell_segmentation as segment
from skimage import exposure,io
from matplotlib.backends.backend_pdf import PdfPages # Library to save plots as pdfs
import matplotlib.pyplot as plt

def GenerateFISHImg(CoreId,markersToPlot,colourVec,fileDir="../data/patientsWithOutcomes/npArraysRaw",box_tl=(0,0),box_width=100):
    # Function to plot a given set of stains in given colours
    markerLabelsVec = ['SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin', 'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3', 'CD4', 'E-cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta-catenin', 'B7-H4', 'Ki67', 'CollagenI', 'CD3', 'CD68', 'PD-L2', 'B7-H3', 'HLA-DR', 'pS6', 'HistoneH3', 'DNA191', 'DNA193']
    # Load the image
    image = np.load(fileDir+"/core_"+str(coreId) + ".npy", "r")
    outImg = np.zeros((image.shape[0],image.shape[1],3))

    for i,marker in enumerate(markersToPlot):
        markerIdx = markerLabelsVec.index(marker)

        # Denoise the image
        currSlice = segment.clean_image(np.copy(image[:,:,markerIdx]),(0,0),100,method="max")

        # Normalise
        currSlice = exposure.rescale_intensity(currSlice)

        # Colour
        currColour = np.reshape(colourVec[i],(1,3))
        currSlice = np.stack([currSlice,currSlice,currSlice],axis=2)
        currSlice = np.round(currSlice*currColour)
        outImg += currSlice

    return outImg


# ========================================================
# Load the cores to be plotted
fileDir="../data/patientsWithOutcomes/npArraysRaw"
modelScores = np.loadtxt("pixelCorrelationAnalysis/correlationModelScores.csv",delimiter=",",skiprows=1)
modelScores = modelScores[np.argsort(modelScores[:,5])]
coreIdVec = modelScores[:,0].astype(int)
# markersToPlot = ["CollagenI","CD163","HistoneH3"]
markersToPlot = ["CollagenI","CD163"]
# markersToPlot = ["CD44s","AvantiLipid","HistoneH3"]
# markersToPlot = ["Ki67","B7-H4","HistoneH3"]
# markersToPlot = ["HistoneH3","Vimentin","HistoneH3"]
# colourVec = [[255,0,0],[0,0,255],[0,255,0]]
colourVec = [[0,255,0],[255,0,0]]
fName = "collagen_cd163_NoHistone.pdf"
# fName = "cd44s_avantiLipid.pdf"
# fName = "ki67_b7h4.pdf"
# fName = "histone_vimentin.pdf"

pp = PdfPages(fName)
f = plt.figure(1)
for i,coreId in enumerate(coreIdVec):
    print(str(i + 1) + " of " + str(len(coreIdVec)) + " - Generating image for core " + str(coreId))
    img = GenerateFISHImg(coreId,markersToPlot,colourVec,fileDir=fileDir)
    plt.imshow(img)
    # plt.title("Core: " + str(coreId) + ", Prediction:" + str(modelScores[i,6])+", True:" + str(modelScores[i,1]))
    plt.title("Core: " + str(coreId) + ", Correlation:" + str(modelScores[i,5])+", PtSnty:" + str(modelScores[i,1]))
    pp.savefig(f)

# Close the files and clean up
pp.close()
plt.close()