# ===============================================================
# Script to compare the raw stain levels between different images.
# ===============================================================
dataDir = "~/Desktop/Fluidigm_Project/ovarianCancer/analysis/"
outName = "stainAnalysis"
outDir = "~/Desktop/Fluidigm_Project/ovarianCancer/analysis/"
library(ggplot2)
source("multiplot.R")

# ====================== Read in the data ==========================
workingDir = getwd()
setwd(dataDir)
stainSummaryArr = read.csv("rawStainSummaries.csv",header=F)
names(stainSummaryArr) = c("StainId","MeanStain","TotStain","PtSnty","CoreId")
setwd(workingDir)

# ========================== Plot ==============================
pltTitle = "Mean Linear Classifier Weight Associated with each Stain"
markerLabelsVec = c('SrBCK', 'RR101', 'RR102', 'AvantiLipid', 'XeBCK', 'CD196', 'CD19', 'Vimentin', 'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3', 'CD4', 'E-cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta-catenin', 'B7-H4', 'Ki67', 'CollagenI', 'CD3', 'CD68', 'PD-L2', 'B7-H3', 'HLA-DR', 'pS6', 'HistoneH3', 'DNA191', 'DNA193')
markerLabelsVec = gsub(" ", "\n", markerLabelsVec)

responders = subset(stainSummaryArr,PtSnty==1)
nonResponders = subset(stainSummaryArr,PtSnty==0)

# Mean stain for responders
p1 = ggplot(responders,aes(as.factor(responders$StainId),responders$MeanStain,fill=responders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Mean Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + scale_y_continuous(limits=c(0,160)) + ggtitle("Responders")

# Mean stain for non-responders
p2 = ggplot(nonResponders,aes(as.factor(nonResponders$StainId),nonResponders$MeanStain,fill=nonResponders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Mean Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + scale_y_continuous(limits=c(0,160)) + ggtitle("Non-Responders")

# Abs stain for responders
p3 = ggplot(responders,aes(as.factor(responders$StainId),responders$TotStain,fill=responders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Tot Abs Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + scale_y_continuous(limits=c(0,160000)) + ggtitle("Responders")

# Mean stain for non-responders
p4 = ggplot(nonResponders,aes(as.factor(nonResponders$StainId),nonResponders$TotStain,fill=nonResponders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Tot Abs Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + scale_y_continuous(limits=c(0,160000)) + ggtitle("Non-Responders")


# Save to pdf
pdf(paste(outDir,outName,".pdf",sep=""), onefile = TRUE)
multiplot(p1,p2,cols=1)
multiplot(p3,p4,cols=1)
dev.off()

# ========================== Normalise across each stain ==============================
stainSummaryArr_Normalised = stainSummaryArr
for (stainId in seq(0,36)) {
  # Find the maxs
  maxVal_Mean = max(stainSummaryArr_Normalised$MeanStain[stainSummaryArr_Normalised$StainId==stainId])
  maxVal_Tot = max(stainSummaryArr_Normalised$TotStain[stainSummaryArr_Normalised$StainId==stainId])
  
  # Normalise
  stainSummaryArr_Normalised$MeanStain[stainSummaryArr_Normalised$StainId==stainId] = stainSummaryArr_Normalised$MeanStain[stainSummaryArr_Normalised$StainId==stainId]/maxVal_Mean
  stainSummaryArr_Normalised$TotStain[stainSummaryArr_Normalised$StainId==stainId] = stainSummaryArr_Normalised$TotStain[stainSummaryArr_Normalised$StainId==stainId]/maxVal_Tot
  
}

# ========================== Plot Again ==============================
responders = subset(stainSummaryArr_Normalised,PtSnty==1)
nonResponders = subset(stainSummaryArr_Normalised,PtSnty==0)

# Mean stain for responders
p1 = ggplot(responders,aes(as.factor(responders$StainId),responders$MeanStain,fill=responders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Mean Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + ggtitle("Responders")

# Mean stain for non-responders
p2 = ggplot(nonResponders,aes(as.factor(nonResponders$StainId),nonResponders$MeanStain,fill=nonResponders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Mean Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + ggtitle("Non-Responders")

# Abs stain for responders
p3 = ggplot(responders,aes(as.factor(responders$StainId),responders$TotStain,fill=responders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Tot Abs Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + ggtitle("Responders")

# Mean stain for non-responders
p4 = ggplot(nonResponders,aes(as.factor(nonResponders$StainId),nonResponders$TotStain,fill=nonResponders$StainId)) + geom_violin() + theme_bw() + labs(x="Stain", y="Tot Abs Stain") + theme(legend.position="none", axis.text.x = element_text(angle = 90, hjust = 1)) + geom_point() + geom_jitter(height=0) + scale_x_discrete(labels=markerLabelsVec) + ggtitle("Non-Responders")


# Save to pdf
pdf(paste(outDir,outName,"_Normalised.pdf",sep=""), onefile = TRUE)
multiplot(p1,p2,cols=1)
multiplot(p3,p4,cols=1)
dev.off()
# ========================== As Barplot ==============================
confInt_MeanStain = summarySE(stainSummaryArr_Normalised, measurevar="MeanStain", groupvars=c("StainId","PtSnty"))
confInt_TotStain = summarySE(stainSummaryArr_Normalised, measurevar="TotStain", groupvars=c("StainId","PtSnty"))

# Use 95% confidence intervals instead of SEM
p1 = ggplot(confInt_MeanStain, aes(x=as.factor(StainId+1), y=MeanStain, fill=factor(PtSnty))) + 
    geom_bar(position=position_dodge(0.9), stat="identity") +
    geom_errorbar(aes(ymin=MeanStain-ci, ymax=MeanStain+ci),
        width=.8,                    # Width of the error bars
        position=position_dodge(0.9)) + 
    scale_x_discrete(
        limits=seq(37,1),
        labels=markerLabelsVec) + 
    ylab("Mean Value of Stain Across Image (Normalised)") +
    xlab("") +
    scale_fill_hue(name="Response to Platinum Treatment", # Legend label, use darker colors
        breaks=c(1,0),
        labels=c("Response","No-Response")) +
    theme_bw() +
    theme(legend.position=c(.7,.75)) +
    coord_flip()

p2 = ggplot(confInt_TotStain, aes(x=as.factor(StainId+1), y=TotStain, fill=factor(PtSnty))) + 
    geom_bar(position=position_dodge(0.9), stat="identity") +
    geom_errorbar(aes(ymin=TotStain-ci, ymax=TotStain+ci),
        width=.8,                    # Width of the error bars
        position=position_dodge(0.9)) + 
    scale_x_discrete(
        limits=seq(37,1),
        labels=markerLabelsVec) + 
    ylab("Total Stain Across Image (Normalised)") +
    xlab("") +
    scale_fill_hue(name="Response to Platinum Treatment", # Legend label, use darker colors
        breaks=c(1,0),
        labels=c("Response","No-Response")) +
    theme_bw() +
    theme(legend.position=c(.7,.75)) +
    coord_flip()

# Save to pdf
pdf(paste(outDir,outName,"_NormBarPlot.pdf",sep=""), onefile = TRUE)
p1
p2
dev.off()

# =============== Do t-tests on the raw stain values to check for differences ================
responders = subset(stainSummaryArr_Normalised,PtSnty==1)
nonResponders = subset(stainSummaryArr_Normalised,PtSnty==0)
ttResults = data.frame()
for (stain in seq(0,36)) {
  ttest_Mean = t.test(responders$MeanStain[responders$StainId==stain],nonResponders$MeanStain[nonResponders$StainId==stain])
  ttest_Tot = t.test(responders$TotStain[responders$StainId==stain],nonResponders$TotStain[nonResponders$StainId==stain])
  ttResults = rbind(ttResults,cbind(markerLabelsVec[stain+1],ttest_Mean$p.value,ttest_Tot$p.value))
  
}
names(ttResults) = c("Stain","PValue_Mean","PValue_Tot")
ttResults$PValue_Mean = as.numeric(as.character(ttResults$PValue_Mean))
ttResults$PValue_Tot = as.numeric(as.character(ttResults$PValue_Tot))

# Bonferoni correction for multiple testing
ttResults$PValue_Mean = p.adjust(ttResults$PValue_Mean, method = "bonferroni")
ttResults$PValue_Tot = p.adjust(ttResults$PValue_Tot, method = "bonferroni")

# Is any below 5%?
any(ttResults$PValue < 0.05)

# No
# =============================== PCA Analysis ============================
pdf(paste(outDir,outName,"_PCA.pdf",sep=""), onefile = TRUE)
# Reshape the data into a 'wide' formate so that each row is one image
stainSummaryArr = read.csv("rawStainSummaries.csv",header=F) # Reload original data to make sure there's no contamination
names(stainSummaryArr) = c("StainId","MeanStain","TotStain","PtSnty","CoreId")

meanStain_Wide = data.frame()
for (coreId in unique(stainSummaryArr$CoreId)) {
  tmp_MeanStain = stainSummaryArr$MeanStain[stainSummaryArr$CoreId==coreId]
  tmp_PtSnty = unique(stainSummaryArr$PtSnty[stainSummaryArr$CoreId==coreId])
  meanStain_Wide = rbind(meanStain_Wide,c(coreId, tmp_PtSnty, tmp_MeanStain))
}
names(meanStain_Wide) = c("CoreId","PtSnty",markerLabelsVec)

# Standardise the data, using the caret preproces function. Note: preProcess can also do a Box-Cox transform
require(caret)
meanStainWide_Tranformed = meanStain_Wide
preprocessParams = preProcess(meanStain_Wide[,3:dim(meanStain_Wide)[2]],method=c("center", "scale"),verbose=T)
meanStainWide_Tranformed[,3:dim(meanStain_Wide)[2]] = predict(preprocessParams, meanStain_Wide[,3:dim(meanStain_Wide)[2]])
# summary(meanStainWide_Tranformed)

# PCA with R native
meanStain_PCA = prcomp(meanStainWide_Tranformed[,3:dim(meanStain_Wide)[2]])
summary(meanStain_PCA)

# Plot of the cumulative variance captured by the different PCs
# Eigenvalues
eig = (meanStain_PCA$sdev)^2
# Variances in percentage
variance = eig*100/sum(eig)
# Cumulative variances
cumvar = cumsum(variance)
meanStain_VarCaptured = data.frame(eig = eig, variance = variance,
                     cumvariance = cumvar)

ggplot(meanStain_VarCaptured, aes(seq(1,37),cumvar)) + 
  geom_line() + theme_bw() + 
  ylab("Cumulative Variance Captured by Principal Components (in %)") +
  xlab("Principal Component") +
  ggtitle("Variance Captured by Principal Components")

# Look at how much each marker contributes to the different components
for (pCompId in seq(1,dim(meanStain_PCA$rotation)[2])) {
  pComp = meanStain_PCA$rotation[,pCompId]
  # Normalise
  pComp = pComp*100/sum(abs(pComp))
  pComp = data.frame(MarkerLabels = markerLabelsVec, Contribution=as.numeric(pComp))
  # Plot
  p = ggplot(pComp,aes(x=MarkerLabels,y=Contribution,fill=Contribution)) + 
    geom_bar(position=position_dodge(0.9), stat="identity") +
    theme_bw() +
    ylab("Relative Contribution to Principal Components (in %)") +
    xlab("") +
    ylim(c(-50,50)) +
    scale_fill_gradient(low="blue",high="red") +
    ggtitle(paste("Composition of Principal Component ",pCompId,sep="")) +
    coord_flip()
  print(p)
}

# Do a PCA using Rafaels pca function
source('Utils.R')
PcaPlot(meanStain_Wide,markerLabelsVec,"PtSnty",pcaMethod="svd",scale="none",nBins=0,samePlot=F)

# Plot other components
# pcaResults = data.frame(PtSnty=meanStain_Wide$PtSnty, meanStain_PCA$x[,1:3])
# qplot(x=PC2, y=PC3, data=pcaResults, colour=factor(PtSnty)) +
#   theme(legend.position="none")

# Save to pdf
dev.off()

# =============================== LDA Analysis======================
# Function to carry out LDA with cross validation, taken from https://www.stat.berkeley.edu/~s133/Class2a.html
# It will divide the data into v distinct sets and then train on v-1 sets using the last set for validation.
# It does so for each of the v sets it generated and returns a classification table with the validation results,
# summarised across the full data set (Since it once classified each of the data points without having seen it before). 
vlda = function(v,formula,data,cl){
   require(MASS)
   grps = cut(1:nrow(data),v,labels=FALSE)[sample(1:nrow(data))]
   pred = lapply(1:v,function(i,formula,data){
	    omit = which(grps == i)
	    z = lda(formula,data=data[-omit,])
            predict(z,data[omit,])
	    },formula,data)

   wh = unlist(lapply(pred,function(pp)pp$class))
   return(table(wh,cl[order(grps)]))
}
# ------------------------- Main Part of LDA --------------------------------------
pdf(paste(outDir,outName,"_LDA.pdf",sep=""), onefile = TRUE)
# Reshape the data into a 'wide' formate so that each row is one image
stainSummaryArr = read.csv("rawStainSummaries.csv",header=F) # Reload original data to make sure there's no contamination
names(stainSummaryArr) = c("StainId","MeanStain","TotStain","PtSnty","CoreId")

meanStain_Wide = data.frame()
for (coreId in unique(stainSummaryArr$CoreId)) {
  tmp_MeanStain = stainSummaryArr$MeanStain[stainSummaryArr$CoreId==coreId]
  tmp_PtSnty = unique(stainSummaryArr$PtSnty[stainSummaryArr$CoreId==coreId])
  meanStain_Wide = rbind(meanStain_Wide,c(coreId, tmp_PtSnty, tmp_MeanStain))
}
names(meanStain_Wide) = c("CoreId","PtSnty",markerLabelsVec)

# Standardise the data, using the caret preproces function. Note: preProcess can also do a Box-Cox transform
require(caret)
meanStainWide_Tranformed = meanStain_Wide
preprocessParams = preProcess(meanStain_Wide[,3:dim(meanStain_Wide)[2]],method=c("center", "scale"),verbose=T)
meanStainWide_Tranformed[,3:dim(meanStain_Wide)[2]] = predict(preprocessParams, meanStain_Wide[,3:dim(meanStain_Wide)[2]])

# Remove the coreId column so that it doesn't influence the analysis
meanStainWide_Tranformed = cbind(meanStainWide_Tranformed$PtSnty,meanStainWide_Tranformed[,3:ncol(meanStainWide_Tranformed)])
names(meanStainWide_Tranformed) = c("PtSnty",markerLabelsVec)

# Perform an LDA
meanStain_LDA = lda(PtSnty~.,data=meanStainWide_Tranformed)

# Analyse the LDA
# Plot histograms where on the LDA axis the two groups fall.
# plot(meanStain_LDA) # Native R way of doing it
ldaValArr = data.frame(LdaVal=rep(0,nrow(meanStainWide_Tranformed)),PtSnty=rep(0,nrow(meanStainWide_Tranformed)))
linDisc = meanStain_LDA$scaling
for (i in seq(1,nrow(meanStainWide_Tranformed))) {
  ldaValArr[i,1] = sum(meanStainWide_Tranformed[i,2:ncol(meanStainWide_Tranformed)]*linDisc)
  ldaValArr[i,2] = meanStainWide_Tranformed[i,1]
}

ggplot(ldaValArr,aes(x=LdaVal,fill=factor(PtSnty))) + 
  geom_histogram(alpha=0.8, position="identity") +
  scale_fill_hue(name="Response to \nPlatinum Treatment", # Legend label, use darker colors
        breaks=c(1,0),
        labels=c("Response","No-Response")) +
  xlab("Value on the Linear Discriminant Axis") + 
  theme_bw() +
  ggtitle("Separation of the Groups by the Linear Discriminant")


# Get a feeling for how good the LDA is by using cross-validation
nGroups = 5
nIter = 100
ldaClassErr = rep(0,nIter)
for (i in seq(nIter)) {
  res = vlda(nGroups,PtSnty~.,meanStainWide_Tranformed,meanStainWide_Tranformed$PtSnty)
  ldaClassErr[i] = sum(diag(prop.table(res)))
}
mean(ldaClassErr)
hist(ldaClassErr)

# What stains is the LDA classification built on?
linDisc = meanStain_LDA$scaling # Extract the discriminant
# Normalise
linDisc = linDisc*100/sum(abs(linDisc))
linDisc = data.frame(MarkerLabels = markerLabelsVec, Contribution=as.numeric(linDisc))
# Plot
ggplot(linDisc,aes(x=MarkerLabels,y=Contribution,fill=Contribution)) + 
  geom_bar(position=position_dodge(0.9), stat="identity") +
  theme_bw() +
  ylab("Relative Contribution to Linear Discriminant (in %)") +
  xlab("") +
  ylim(c(-20,20)) +
  scale_fill_gradient(low="red",high="green4") +
  ggtitle(paste("Composition of Linear Discriminant ",sep="")) +
  coord_flip()

# Zoom in
ggplot(linDisc,aes(x=MarkerLabels,y=Contribution,fill=Contribution)) + 
  geom_bar(position=position_dodge(0.9), stat="identity") +
  theme_bw() +
  ylab("Relative Contribution to Linear Discriminant (in %)") +
  xlab("") +
  ylim(c(-15,15)) +
  scale_fill_gradient(low="red",high="green4") +
  ggtitle(paste("Composition of Linear Discriminant - Zoomed",sep="")) +
  coord_flip()
# --------------------------------------------
# Interpret the LDA
# 1) Do DNA193 and DNA191 actually contribute to the classification?
# To check this I will compute the contribution of DNA193 and DNA 191 for each
# core. If adding their signals together always gives 0 this means the classifier
# is actually kind of ignoring them.
dnaStainContribArr = data.frame(Contrib=rep(0,nrow(meanStainWide_Tranformed)),CoreId=rep(0,nrow(meanStainWide_Tranformed)))
linDisc = meanStain_LDA$scaling
dna191Id = which(markerLabelsVec=="DNA191")
dna193Id = which(markerLabelsVec=="DNA193")
for (i in seq(1,nrow(meanStainWide_Tranformed))) {
  
  dnaStainContribArr[i,1] = meanStainWide_Tranformed[i,dna191Id]*linDisc[dna191Id] + meanStainWide_Tranformed[i,dna193Id]*linDisc[dna193Id]
  dnaStainContribArr[i,2] = meanStain_Wide$CoreId[i]
}

# No, they don't balance out. Strange
# 2) What happens if I don't include them in the LDA?
markerLabelsVec_Cleaned = c('CD196', 'CD19', 'Vimentin', 'CD163', 'CD20', 'CD16', 'CD25', 'p53', 'CD134', 'CD45', 'CD44s', 'CD14', 'FoxP3', 'CD4', 'E-cadherin', 'p21', 'CD152', 'CD8a', 'CD11b', 'Beta-catenin', 'B7-H4', 'Ki67', 'CollagenI', 'CD3', 'CD68', 'PD-L2', 'B7-H3', 'HLA-DR', 'pS6', 'HistoneH3')
meanStainWide_NoGeneralMarkers = meanStainWide_Tranformed[,c("PtSnty",markerLabelsVec_Cleaned)]
# Perform an LDA
meanStain_LDA = lda(PtSnty~.,data=meanStainWide_NoGeneralMarkers)

# Analyse the LDA
# Plot histograms where on the LDA axis the two groups fall.
# plot(meanStain_LDA) # Native R way of doing it
ldaValArr = data.frame(LdaVal=rep(0,nrow(meanStainWide_NoGeneralMarkers)),PtSnty=rep(0,nrow(meanStainWide_NoGeneralMarkers)))
linDisc = meanStain_LDA$scaling
for (i in seq(1,nrow(meanStainWide_NoGeneralMarkers))) {
  ldaValArr[i,1] = sum(meanStainWide_NoGeneralMarkers[i,2:ncol(meanStainWide_NoGeneralMarkers)]*linDisc)
  ldaValArr[i,2] = meanStainWide_NoGeneralMarkers[i,1]
}

ggplot(ldaValArr,aes(x=LdaVal,fill=factor(PtSnty))) + 
  geom_histogram(alpha=0.8, position="identity") +
  scale_fill_hue(name="Response to \nPlatinum Treatment", # Legend label, use darker colors
        breaks=c(1,0),
        labels=c("Response","No-Response")) +
  xlab("Value on the Linear Discriminant Axis") + 
  theme_bw() +
  ggtitle("Separation of the Groups by the Linear Discriminant with Generic Stains Removed")


# Get a feeling for how good the LDA is by using cross-validation
nGroups = 5
nIter = 100
ldaClassErr = rep(0,nIter)
for (i in seq(nIter)) {
  res = vlda(nGroups,PtSnty~.,meanStainWide_NoGeneralMarkers,meanStainWide_NoGeneralMarkers$PtSnty)
  ldaClassErr[i] = sum(diag(prop.table(res)))
}
mean(ldaClassErr)
hist(ldaClassErr)

# What stains is the LDA classification built on?
linDisc = meanStain_LDA$scaling # Extract the discriminant
# Normalise
linDisc = linDisc*100/sum(abs(linDisc))
linDisc = data.frame(MarkerLabels = markerLabelsVec_Cleaned, Contribution=as.numeric(linDisc))
# Plot
ggplot(linDisc,aes(x=MarkerLabels,y=Contribution,fill=Contribution)) + 
  geom_bar(position=position_dodge(0.9), stat="identity") +
  theme_bw() +
  ylab("Relative Contribution to Linear Discriminant (in %)") +
  xlab("") +
  ylim(c(-15,15)) +
  scale_fill_gradient(low="red",high="green4") +
  ggtitle(paste("Composition of Linear Discriminant with Non-Indicative Stains Removed",sep="")) +
  coord_flip()
dev.off()

# =============================== Logistic Regression ======================
pdf(paste(outDir,outName,"_LogisticRegression.pdf",sep=""), onefile = TRUE)
# Reshape the data into a 'wide' formate so that each row is one image
stainSummaryArr = read.csv("rawStainSummaries.csv",header=F) # Reload original data to make sure there's no contamination
names(stainSummaryArr) = c("StainId","MeanStain","TotStain","PtSnty","CoreId")

meanStain_Wide = data.frame()
for (coreId in unique(stainSummaryArr$CoreId)) {
  tmp_MeanStain = stainSummaryArr$MeanStain[stainSummaryArr$CoreId==coreId]
  tmp_PtSnty = unique(stainSummaryArr$PtSnty[stainSummaryArr$CoreId==coreId])
  meanStain_Wide = rbind(meanStain_Wide,c(coreId, tmp_PtSnty, tmp_MeanStain))
}
names(meanStain_Wide) = c("CoreId","PtSnty",markerLabelsVec)

# Standardise the data, using the caret preproces function. Note: preProcess can also do a Box-Cox transform
require(caret)
meanStainWide_Tranformed = meanStain_Wide
preprocessParams = preProcess(meanStain_Wide[,3:dim(meanStain_Wide)[2]],method=c("center", "scale"),verbose=T)
meanStainWide_Tranformed[,3:dim(meanStain_Wide)[2]] = predict(preprocessParams, meanStain_Wide[,3:dim(meanStain_Wide)[2]])

# Remove the coreId column so that it doesn't influence the analysis
meanStainWide_Tranformed = cbind(meanStainWide_Tranformed$PtSnty,meanStainWide_Tranformed[,3:ncol(meanStainWide_Tranformed)])
names(meanStainWide_Tranformed) = c("PtSnty",markerLabelsVec)
summary(meanStainWide_Tranformed)

# Fit a logistic model to the full set of covariates
meanStain_LogitModel = glm(PtSnty ~.,family=binomial(link='logit'),data=meanStainWide_Tranformed)

# Analyse the results
summary(meanStain_LogitModel)
anova(meanStain_LogitModel, test="Chisq")

# Check for co-linearity
library(car)
vif(meanStain_LogitModel)
plot(meanStainWide_Tranformed)


# Plot the coefficients
confLevel = 0.95
logitMCoeffs = meanStain_LogitModel$coefficients
logitMStdErrs = summary(meanStain_LogitModel)$coefficients[,2]
# Normalise
normFact = sum(abs(logitMCoeffs))
logitMCoeffs = logitMCoeffs*100/normFact
logitMStdErrs = logitMStdErrs*100/normFact
confIntLogitCoeffs = data.frame(MarkerLabels=markerLabelsVec,MeanCoeff=as.numeric(logitMCoeffs[2:length(logitMCoeffs)]),SE=logitMStdErrs[2:length(logitMStdErrs)],CI=qt(confLevel/2+.5, nrow(meanStainWide_Tranformed))*logitMStdErrs[2:length(logitMStdErrs)])
# Plot
ggplot(confIntLogitCoeffs,aes(x=MarkerLabels,y=MeanCoeff,fill=MeanCoeff)) + 
  geom_bar(position=position_dodge(0.9), stat="identity") +
  geom_errorbar(aes(ymin=MeanCoeff-CI, ymax=MeanCoeff+CI),
                width=.8,                    # Width of the error bars
                position=position_dodge(0.9)) + 
  theme_bw() +
  ylab("Relative Contribution to Linear Discriminant (in %)") +
  xlab("") +
  scale_fill_gradient(low="red",high="green4") +
  ggtitle(paste("Coefficients of Logistic Model",sep="")) +
  coord_flip()
dev.off()
# ========================== Helper Functions ==============================
## Gives count, mean, standard deviation, standard error of the mean, and confidence interval (default 95%).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   groupvars: a vector containing names of columns that contain grouping variables
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)
summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
    library(plyr)

    # New version of length which can handle NA's: if na.rm==T, don't count them
    length2 <- function (x, na.rm=FALSE) {
        if (na.rm) sum(!is.na(x))
        else       length(x)
    }

    # This does the summary. For each group's data frame, return a vector with
    # N, mean, and sd
    datac <- ddply(data, groupvars, .drop=.drop,
      .fun = function(xx, col) {
        c(N    = length2(xx[[col]], na.rm=na.rm),
          mean = mean   (xx[[col]], na.rm=na.rm),
          sd   = sd     (xx[[col]], na.rm=na.rm)
        )
      },
      measurevar
    )

    # Rename the "mean" column    
    datac <- rename(datac, c("mean" = measurevar))

    datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean

    # Confidence interval multiplier for standard error
    # Calculate t-statistic for confidence interval: 
    # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
    ciMult <- qt(conf.interval/2 + .5, datac$N-1)
    datac$ci <- datac$se * ciMult

    return(datac)
}
