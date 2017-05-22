# ===============================================================
# Script to analyse the weight matrices from the linear classifiers.
# ===============================================================
dataDir = "~/Desktop/linearClassifierAnalysis/weights_naiveClassifier_BatchSize_10"
outName = "weightAnalysis"
outDir = "~/Desktop/linearClassifierAnalysis/"
library(ggplot2)
source("multiplot.R")

# ====================== Read in the data ==========================
workingDir = getwd()
setwd(dataDir)
weightArr = data.frame()
for (fName in list.files(dataDir)) {
  if(grepl("summaries", fName)) { # Load the summary csv files
    # Get the set id
    setId = as.numeric(tail(strsplit(fName,split="_")[[1]],n=2)[1])

    # Read and append the data
    tmp = read.csv(fName,header=FALSE)
    tmp = cbind(tmp,rep(setId,dim(tmp)[1]))
    names(tmp) = c("FilterId","MeanWeight","PtSnty","SetId")
    weightArr = rbind(weightArr,tmp)


  }
}
names(weightArr) = c("FilterId","MeanWeight","PtSnty","SetId")
setwd(workingDir)
# ========================== Plot ==============================
pltTitle = "Mean Linear Classifier Weight Associated with each Stain"
markerLabelsVec = c('SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin', 'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3', 'CD4', 'E-cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta-catenin', 'B7-H4', 'Ki67', 'CollagenI', 'CD3', 'CD68', 'PD-L2', 'B7-H3', 'HLA-DR', 'pS6', 'HistoneH3', 'DNA191', 'DNA193')
markerLabelsVec = gsub(" ", "\n", markerLabelsVec)


responders = subset(weightArr,PtSnty==1)
nonResponders = subset(weightArr,PtSnty==0)

# For responders
p1 = ggplot(responders,aes(as.factor(responders$FilterId),responders$MeanWeight,fill=responders$FilterId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Mean Weight") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + ggtitle("Responders")

# For non-responders
p2 = ggplot(nonResponders,aes(as.factor(nonResponders$FilterId),nonResponders$MeanWeight,fill=nonResponders$FilterId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Mean Weight") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + ggtitle("Non-Responders")

# Save to pdf
pdf(paste(outDir,outName,".pdf",sep=""), onefile = TRUE)
multiplot(p1,p2,cols=1)
dev.off()
