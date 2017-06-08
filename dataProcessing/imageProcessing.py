# ========================================================================
# Script to turn fluidigm txt files into numpy array, ready to go
# for the training process.
# ========================================================================
# import pandas as pd
import numpy as np
import DataProc as dtp
import re as re
import pickle
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages # Library to save plots as pdfs
import os

# Control which of the operations the script should carry out
MAKE_NUMPY_AND_FIND_DIMENSION = True
DO_ANALYSE_DIMENSIONS = False
DO_CROP = False
DO_TRANSFORM = False

# ========================================================================
# Function to get coreID from file name via regex matching
def getCoreId(fName):
    regex = regex = re.compile(r'\d+')
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

# Function to get patient outcome from coreId
def getPatientId(coreId, outcomeArr):
    patientId = outcomeArr[np.where(outcomeArr[:,0] == coreId),2]
    if patientId.size==0:
        patientId = -1 # Return -1 as to indicate no patient id
    else:
        patientId = int(patientId[0][0])
    return patientId

# Function to plot core to pdf
def plotToPdf(image,fName=None,CoreId=None):

    markerLabels = ['SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin',
                   'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3',
                   'CD4', 'E.cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta.catenin', 'B7.H4', 'Ki67',
                   'CollagenI', 'CD3', 'CD68', 'PD.L2', 'B7.H3', 'HLA.DR', 'pS6', 'HistoneH3', 'DNA191',
                   'DNA193']

    # Generate filename
    if fName==None and CoreId != None:
        fName = "CoreId"+str(CoreId)+".pdf"
    elif fName==None and CoreId != None:
        raise Exception("No filename for pdf provided!")

    # Open pdf and figure
    pp = PdfPages(fName)
    f=plt.figure(1)

    # Plot and save

    for stain in range(37):
        if np.isnan(image[0,0,stain]):
            plt.imshow(np.ones_like(image[:, :, stain])*255)
        else:
            plt.imshow(image[:,:,stain])
        plt.title("Core: "+str(CoreId)+", Label:"+markerLabels[stain])
        pp.savefig(f)

    # Close the files and clean up
    pp.close()
    plt.close()

# ========================================================================
# Load the files and pickle them. Find the dimensions of each image
if MAKE_NUMPY_AND_FIND_DIMENSION:
    rawFileNames = dtp.GetFileNames('../data/patientsWithOutcomes/txtFiles_reordered/') # Get file names
    outcomeArr  = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',') # Get Array with patient outcomes

    imgDimArr = np.zeros((len(rawFileNames),4))

    # Set up environment
    if not os.path.exists("../data/patientsWithOutcomes/pdfsOriginals/"):
        os.mkdir("../data/patientsWithOutcomes/pdfsOriginals/")
    if not os.path.exists("../data/patientsWithOutcomes/npArraysRaw/"):
        os.mkdir("../data/patientsWithOutcomes/npArraysRaw/")


    for i,fName in enumerate(rawFileNames):

        # Extract coreID
        coreId = getCoreId(fName)

        print(str(i+1) +" of "+str(len(rawFileNames))+" - Processing Core: "+ str(coreId))

        # Find out if patient was platinum sensitive
        ptSensitiveVec = getPatientOutcome(coreId, outcomeArr)

        # Get the patient id
        patientId = getPatientId(coreId, outcomeArr)

        # Extract Image Data
        markerLabels,_,image=dtp.GenArrs(fName)

        # Store its dimensions
        imgDimArr[i,:] = np.array([int(coreId), image.shape[0], image.shape[1], image.shape[2]])

        # Pickle the image XXX NOT ANYMORE - TOO INEFFICIENT - SAVING AS NUMPY IS BETTER XXX
        # data={'markerLabels': markerLabels, 'Image': image, 'PtSensitive': ptSensitiveVec, 'PatientId': patientId}
        # out=open("Data/PatientsWithOutcomes/RawNpArrays/core"+str(coreId)+".p","wb")
        # pickle.dump(data,out)
        # out.close()

        # Generate a pdf
        fName = "../data/patientsWithOutcomes/pdfsOriginals/core_"+str(coreId)+"_Original"+".pdf"
        plotToPdf(image,fName,coreId)

        # Save as np array
        outName = "../data/patientsWithOutcomes/npArraysRaw/core_"+str(coreId)+".npy"
        np.save(outName,image)

        # Save the dimensions array
        np.savetxt("../data/patientsWithOutcomes/imageDimensions.csv", imgDimArr,fmt='%10.0f', delimiter=',', newline='\n', header='', footer='', comments='# ')

# ========================================================================
# Find the minimum dimensions and check that each image has 37 channels
if DO_ANALYSE_DIMENSIONS:
    imgDimArr  = np.genfromtxt('../data/patientsWithOutcomes/imageDimensions.csv', delimiter=',') # Get Array with patient outcomes

    print('========================================================================')
    print('Total Number of images in array: '+str(imgDimArr.shape[0]))
    print('Minimum x dimension: '+ str(imgDimArr.min(axis=0)[2])+' px')
    print('Minimum y dimension: '+ str(imgDimArr.min(axis=0)[1])+' px')
    print('========================================================================')

    # Which image is so small?
    minCores = np.argmin(imgDimArr,axis=0)
    imgDimArr[minCores[1],:]
    imgDimArr[minCores[2],:]

    # Which images have the maximal dimensions
    maxCores = np.argmax(imgDimArr,axis=0)
    imgDimArr[maxCores[1],:]
    imgDimArr[maxCores[2],:]


    # look at the distribution of sizes:
    f=plt.figure(1)
    plt.subplot(2,1,1)
    plt.hist(imgDimArr[:,1], bins='auto')  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of Image X Dimensions")

    plt.subplot(2,1,2)
    plt.hist(imgDimArr[:,2], bins='auto')  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of Image Y Dimensions")

    pp = PdfPages('imgSizeDistribution.pdf')
    pp.savefig(f)
    pp.close()
    plt.close()

# ========================================================================
# Crop the images by embedding them into a CANONICAL_DIMS_X x CANONICAL_DIMS_Y array of zeros as follows:
# 1.	Generate 600x600 array filled with 0s.
# 2.	If the image is bigger in X, take its central 600 columns. Else take all columns of the image and embed them in
#       the center of the 0 array (thereby padding it).
# 3.	Repeat for Y.
# NOTE: If exact centering isn't possible because the number of pixels is even, I position the rectangle closer to the
# left (in x) and top (in y) (using a floor() operation).

if DO_CROP:
    CANONICAL_DIMS_X = 600
    CANONICAL_DIMS_Y = 600
    N_STAINS = 37
    outcomeArr  = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',') # Get Array with patient outcomes
    rawFileNames = dtp.GetFileNames('../data/patientsWithOutcomes/txtFiles/') # Get file names


    for i,inFName in enumerate(rawFileNames):

        # Extract coreID
        coreId = getCoreId(inFName)
        # coreId = 73 # Provide specific corea

        print(str(i+1) +" of "+str(len(rawFileNames))+" - Cropping core "+str(coreId))

        # Load the data
        image = np.load("../data/patientsWithOutcomes/npArraysRaw/core_"+str(coreId)+".npy","r")

        # Find its dimensions
        imgDims = image.shape

        # Embed the image in the CANONICAL_DIMS_X x CANONICAL_DIMS_Y template (thereby doing cropping and padding)
        # Generate array to hold cropped image
        processedImg = np.zeros((CANONICAL_DIMS_X,CANONICAL_DIMS_Y,N_STAINS))

        # Compute difference between image and template to decide on cropping and padding
        diff_x = imgDims[1]-CANONICAL_DIMS_X # Difference to desired size in x dimension
        diff_y = imgDims[0]-CANONICAL_DIMS_Y # Difference to desired size in y dimension

        # Determine insertion points in x dimension
        insertionPtImg_x = 0 # Use image from this column onwards
        insertionPtTmp_x = 0 # Embed in the template from this column onwards (non-zero leads to padding)
        endPtImg_x = 0 # Use image up to this column
        endPtTmp_x = 0 # Embed in the template up to this column (for centering if the image is too small)
        if diff_x > 0: # If image is bigger than the desired size crop by inserting only the central area of the image into the template
            insertionPtImg_x = int(math.floor(diff_x/2.0))
            insertionPtTmp_x = 0
            endPtImg_x = insertionPtImg_x + CANONICAL_DIMS_X
            endPtTmp_x = CANONICAL_DIMS_X
        else:
            insertionPtImg_x = 0
            insertionPtTmp_x = int(math.floor(abs(diff_x)/2.0))
            endPtImg_x = imgDims[1]
            endPtTmp_x = insertionPtTmp_x + imgDims[1]

        # Determine insertion points in y dimension
        insertionPtImg_y = 0 # Use image from this column onwards
        insertionPtTmp_y = 0 # Embed in the template from this column onwards (non-zero leads to padding)
        endPtImg_y = 0 # Use image up to this row
        endPtTmp_y = 0 # Embed in the template up to this column (for centering if the image is too small)
        if diff_y > 0: # If image is bigger than the desired size crop by inserting only the central area of the image into the template
            insertionPtImg_y = int(math.floor(diff_y/2.0))
            insertionPtTmp_y = 0
            endPtImg_y= insertionPtImg_y + CANONICAL_DIMS_Y
            endPtTmp_y = CANONICAL_DIMS_Y
        else:
            insertionPtImg_y = 0
            insertionPtTmp_y = int(math.floor(abs(diff_y)/2.0))
            endPtImg_y = imgDims[0]
            endPtTmp_y = insertionPtTmp_y + imgDims[0]

        # Embed
        processedImg[insertionPtTmp_y:endPtTmp_y,insertionPtTmp_x:endPtTmp_x,:] = image[insertionPtImg_y:endPtImg_y,insertionPtImg_x:endPtImg_x,:]

        # Plot to PDF
        fName = "../data/patientsWithOutcomes/pdfsCropped/core_"+str(coreId)+"_Cropped"+".pdf"
        plotToPdf(processedImg,fName)

        # Save the numpy array
        outName = "../data/patientsWithOutcomes/npArraysCropped/core_"+str(coreId)+"_Cropped"+".npy"
        np.save(outName,processedImg)

# ========================================================================
# Transform the data
if DO_TRANSFORM:
    outcomeArr  = np.genfromtxt('../data/patientsWithOutcomes/coreToSensitivityMap_121Cores.csv', delimiter=',') # Get Array with patient outcomes
    rawFileNames = dtp.GetFileNames('../data/patientsWithOutcomes/txtFiles/') # Get file names

    for i,inFName in enumerate(rawFileNames):

        # Extract coreID
        coreId = getCoreId(inFName)
        # coreId = 73 # Provide specific core

        print(str(i+1) +" of "+str(len(rawFileNames))+" - Transforming core "+str(coreId))

        # Load the data
        image = np.load("../data/patientsWithOutcomes/npArraysCropped/core_"+str(coreId)+"_Cropped.npy")

        # Transform the image
        processedImg = np.log1p(image)

        # Plot to PDF
        fName = "../data/patientsWithOutcomes/pdfsLogTransformed/core_"+str(coreId)+"_Log"+".pdf"
        plotToPdf(processedImg,fName)

        # Save the numpy array
        outName = "../data/patientsWithOutcomes/npArraysLogTransformed/core_"+str(coreId)+"_Log"+".npy"
        np.save(outName,processedImg)

# ========================================================================
# Other stuff
# Old cropping code
# # Crop in the x dimension
# pxToCrop_x = imgDims[1]-CANONICAL_DIMS_X
#
# image = np.delete(image,list(range(imgDims[1]-int(math.ceil(pxToCrop_x/2.0)),imgDims[1])),axis=1) # Crop on the right
# image = np.delete(image,list(range(0,int(math.floor(pxToCrop_x/2.0)))),axis=1) # Crop on the left
#
# # Crop in the y dimension
# pxToCrop_y = imgDims[0]-CANONICAL_DIMS_Y
#
# image = np.delete(image,list(range(imgDims[0]-int(math.ceil(pxToCrop_y/2.0)),imgDims[0])),axis=0) # Crop on the top
# image = np.delete(image,list(range(0,int(math.floor(pxToCrop_y/2.0)))),axis=0) # Crop on the bottom

