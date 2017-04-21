
#IMPORTS AND GLOBALS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from Utils import *

PROJ_PATH=os.path.expanduser("~/OvarianCancer")
N_SLIDES=21
N_LABELS=37

#DATA PREPROC

def GenArrs(path):
  raw=pd.read_csv(path,sep="\t")
  #arrShape=(raw.shape[1]-6,raw['X'].max()+1,raw['Y'].max()+1)
  #rawShape=(arrShape[1],arrShape[2],arrShape[0])
  arrShape=(raw.shape[1]-6,raw['Y'].max()+1,raw['X'].max()+1)
  rawShape=(arrShape[1],arrShape[2],arrShape[0])
  byTypes=np.zeros(arrShape)
  labels=list(raw)[6:]
  byPixels=raw.as_matrix(labels)
  byPixels=byPixels.reshape(rawShape)
  for i in range(len(labels)):
    byTypes[i]=byPixels[:,:,i]
  return labels,byTypes,byPixels

def GetFileNames(dirLoc):
  dirPath=PROJ_PATH+dirLoc
  filenames=[dirPath+"/"+file for file in os.listdir(dirPath) if 'Summary' not in file and '.txt' in file]
  return filenames

def PickleAll():
  filePaths=GetFileNames("/Feb 01")
  for i,path in enumerate(filePaths):
    lablels,byTypes,byPixels=GenArrs(path)
    data={'labels':lablels,'ab':byTypes,'pix':byPixels}
    out=open(PROJ_PATH+"/Processed/slide"+str(i)+".p","wb")
    pickle.dump(data,out)
    out.close()

def LoadSlide(iSlide):
  return pickle.load(open(PROJ_PATH+"/Processed/slide"+str(iSlide)+".p","rb"))

def LoadAll():
  print("loading data")
  ret=[]
  for i in range(N_SLIDES):
    ret.append(LoadSlide(i))
  print("done loading")
  return ret

#DATA TRANSFORMATION

def Pix1D(pixels):
  return pixels.reshape((pixels.shape[0] * pixels.shape[1], pixels.shape[2]))

def All1D():
  allDat=[]
  for i in range(N_SLIDES):
    data=LoadSlide(i)
    allDat.append(Pix1D(data['pix']))
  return np.concatenate(allDat,axis=0)

def BoundData(pixels):
  shape,pix1D=Pix1D(pixels)
  mins=pix1D.min(axis=0)
  pix1D=pix1D/(pix1D.max(axis=0)-mins)-mins
  return pix1D.reshape(shape)

def LogData(pixels,nLogs):
  for i in range(nLogs):
    pixels=np.log1p(pixels)
  return pixels

#STATS


#DATA VISUALIZATION

def PlotAB(ax,data,label,index,nLogs=0,cmap='nipy_spectral'):
  pixels=data['ab'][index]
  print(pixels.shape)
  if nLogs!=0: pixels=LogData(pixels,1)
  PlotImgColormap(ax, pixels, cmap, True)
  SetLabels(ax,label)

def PlotAll(pixels,cmap='nipy_spectral'):
  dataAll=np.sum(pixels,axis=2)
  cax=plt.imshow(dataAll,cmap)
  plt.colorbar(cax)
  plt.show()

#RUN

#todo: otsu thresholding
#todo: look at uncorrelated data

def PlotAllLabel(allDat,iLabel,nLogs=0,nSlide=None):
  j=iLabel
  label=allDat[0]['labels'][iLabel]
  if nSlide is not None:
    axs=GenAxs(1,1)
    data=allDat[nSlide]
    PlotAB(axs[0],data,str(nSlide)+":"+str(j)+":"+label,j,nLogs)
  else:
#  for j,label in enumerate(allDat[0]['labels']):
    axs=GenAxs(4,6)
    for i in range(N_SLIDES):
      #data=LoadSlide(i)
      data=allDat[i]
      #axs=GenAxs(1,1)
      PlotAB(axs[i],data,str(i)+":"+str(j)+":"+label,j,nLogs)
  plt.show()

data=LoadAll()
for i in range(N_LABELS):
  PlotAllLabel(data,i,1)
#PickleAll()

#allDat=All1D()

#PlotAB(data['pix'],6)
#shape,allDat=Pix1D(data['pix'])

#print(allDat.shape)
#corrMat=np.corrcoef(allDat.T)
#axs=GenAxs(1,1)
#PlotImg(axs[0],corrMat,colorMap='seismic',center=0)
#plt.show()

#for i in range(N_SLIDES):
#  data=LoadSlide(i)
#  Pix1D(data['pix'])
#CoorelationMatrix(data['pix'])

#for i in range(19):
#  data=LoadSlide(i)
#
#  dataAll=np.sum(pixMin,axis=2)
#  cax=plt.imshow(dataAll,cmap='nipy_spectral')
#  plt.colorbar(cax)
#  plt.show()
