# ===============================================================
# Script to compare different models.
# ===============================================================
dataDir = "~/Desktop/linearClassifierAnalysis/"
outName = "ModelComparison"
outDir = "~/Desktop/linearClassifierAnalysis/"
library(ggplot2)
library(gridExtra)
source("multiplot.R")
setwd(dataDir)

# ===============================================================
# Concatenate the files from the different runs into a single file
ConcatenatePerformanceFiles = function(dataDir,outName) {
  prevDir = getwd()
  setwd(dataDir)
  for (fName in list.files()) {
    if(grepl("performance", fName)) {
      system(paste("cat",fName,">>",outName,sep=" "))
    }
  }
  setwd(prevDir)
}

# ===============================================================
# Extract run time information for a model from the training logs
getRunTimeInformation = function(pathToTrainingLogs) {
  prevDir = getwd() # Save current directory so to return to it later
  setwd(pathToTrainingLogs)
  timeArr = data.frame()
  for (fName in list.files()) {
    # Extract the data set id
    setId = as.numeric(unlist(strsplit(gsub("[^0-9]", "", unlist(fName)), "")))

    # Load the data
    trainingResults = read.csv(fName,header=F)

    # Load the data
    trainingResults = read.csv(fName,header=F)
    names(trainingResults) = c("Epoch","Entropy","ClassificationError","TimePerEpoch","TimePerBatch")

    tmp = cbind(rep(setId,dim(trainingResults)[1]), trainingResults$TimePerEpoch, trainingResults$TimePerBatch)
    names(tmp) = c("SetId","TimePerEpoch","TimePerBatch")
    timeArr = rbind(timeArr,tmp)
  }
  setwd(prevDir)
  return(timeArr)
}

# ================= Read in Data =================================
# Read in data
modelVec = c("naiveClassifier_BatchSize_10","naiveClassifier_BatchSize_25")
fullModelNameVec = c("Naive Classifier - Batch Size 10","Naive Classifier - Batch Size 25")
fullModelNameVec = gsub(" ", "\n", fullModelNameVec)
modelPerformanceArr = data.frame()
runTimeArr = data.frame()

for (model in modelVec) {
  # Performance on test set
  # Concatenate performance file for benchmark from individual runs
  perfFileDir = paste(dataDir,model,sep="")
  perfSumName = paste(dataDir,"performance_",model,".csv",sep="")
  ConcatenatePerformanceFiles(perfFileDir,perfSumName)

  # Now read the file for analysis
  tmp = read.csv(perfSumName) #read.csv(paste("performance_",model,".csv",sep=""))
  tmp = cbind(tmp,rep(model,dim(tmp)[1]))
  names(tmp) = c("SetId","Entropy","ClassificationError","Model")
  modelPerformanceArr = rbind(modelPerformanceArr,tmp)

  # Run times during training
  logFileDir = paste("~/Desktop/linearClassifierAnalysis/",model,"/trainingLogFiles",sep="")
  timeArr = getRunTimeInformation(logFileDir)
  tmp = cbind(timeArr,rep(model,dim(timeArr)[1]))
  names(tmp) = c("SetId","TimePerEpoch","TimePerBatch","Model")
  runTimeArr = rbind(runTimeArr,tmp)
}

# ==================== Plot ======================================
pltTitle = "Performance on 12 Unseen images \n after Training on 90 Images"

# The entropy for each model
p1 = ggplot(modelPerformanceArr,aes(modelPerformanceArr$Model,modelPerformanceArr$Entropy,fill=modelPerformanceArr$Model)) + geom_violin() + theme_bw() + labs(x="", y="Entropy") + theme(legend.position="none") + scale_x_discrete(labels=fullModelNameVec) + geom_point() + geom_jitter(height=0)

# The classification error for each model
p2 = ggplot(modelPerformanceArr,aes(modelPerformanceArr$Model,modelPerformanceArr$ClassificationError,fill=modelPerformanceArr$Model),fill=modelPerformanceArr$Model) + geom_violin() + theme_bw() + labs(x="", y="Classification Error") + theme(legend.position="none") + scale_x_discrete(labels=fullModelNameVec) + geom_point() + geom_jitter(height=0)

# The time per epoch for each model
p3 = ggplot(runTimeArr,aes(runTimeArr$Model,runTimeArr$TimePerEpoch,fill=runTimeArr$Model)) + geom_violin() + theme_bw() + labs(x="", y="Time per Epoch (in s)") + theme(legend.position="none") + scale_x_discrete(labels=fullModelNameVec) + geom_point() + geom_jitter(height=0)

# The time per batch for each model
p4 = ggplot(runTimeArr,aes(runTimeArr$Model,runTimeArr$TimePerBatch,fill=runTimeArr$Model)) + geom_violin() + theme_bw() + labs(x="", y="Time per Batch (in s)") + theme(legend.position="none") + scale_x_discrete(labels=fullModelNameVec) + geom_point() + geom_jitter(height=0)

# Save to pdf
pdf(paste(outDir,outName,".pdf",sep=""), onefile = TRUE)
# multiplot(p1,p2, cols=1)
grid.arrange(p1,p2, top = pltTitle,layout_matrix = matrix(c(1,2), ncol=2, byrow=TRUE))
grid.arrange(p3,p4 ,layout_matrix = matrix(c(1,2), ncol=2, byrow=TRUE))
dev.off()


# ================ Old Snippets =================
# Concatenate Performance Files
# # Generate performance file for benchmark form individual runs
# perfFileDir = "~/Desktop/linearClassifierAnalysis/naiveClassifier_BatchSize_25"
# perfSumName = "~/Desktop/linearClassifierAnalysis/performance_naiveClassifier_BatchSize_25.csv"
# ConcatenatePerformanceFiles(perfFileDir,perfSumName)
#
# # Generate performance file for benchmark form individual runs
# perfFileDir = "~/Desktop/linearClassifierAnalysis/naiveClassifier_BatchSize_10"
# perfSumName = "~/Desktop/linearClassifierAnalysis/performance_naiveClassifier_BatchSize_10.csv"
# ConcatenatePerformanceFiles(perfFileDir,perfSumName)
