# =======================================================
# Function to decorrelate the variables in an R regression model (of class lm or glm).
# It drops the variable with the highest VIF until all VIFs are below targetVIF. VIF is a
# measure of how well one variable can be expressed as a linear combination of the others.
DecorrelateVariables = function(model,targetVIF,verbose=T) {
  library(car)
  repeat{
    # Identify the variable with the maximum VIF 
    vifVec = vif(model)
    maxVIF = max(vifVec)
    # vif() output changes if there are factor variables in the model.
    # Adjust for this.
    if(is.null(dim(vifVec))) { # No factor
      varToDrop = names(vifVec)[which.max(vifVec)]
    } else{ # Factor
      varToDrop = names(vifVec[,1])[which.max(vifVec[,1])]
    } 
    
    # If the maximum VIF is below the target stop
    if (maxVIF<=targetVIF){
      break
    } else { # Else, drop that variable
      model = update(model,paste0(".~.-",varToDrop))
      if(verbose) {print(paste0("Dropping ",varToDrop," with VIF ",maxVIF))}
    }
  }
  return(model)
}

# =======================================================
# Function to compute the accuracy of the model in predicting the provided input variables
LmAccuracy = function(model,classifier=F) {
  if(classifier) {
    accuracy = mean(ifelse(model$fitted.values>0.5,1,0)==model$y)
  } else {
    accuracy = mean(abs(model$residuals))
  }
}

# =======================================================
# Function to do alternating AIC reduction via step() and VIF reduction
#(de-correlation) by dropping the variable with the highest VIF.
AICVIFCoElimination=function(model,targetVIF=10,classifier=T,verbose=T){
  output=data.frame(aic=c(),accuracy=c(),maxVIF=c(),remainingVar=c(),model=c())
  vifVec = vif(model)
  maxVIF = max(vifVec)
  repeat{
    # Drop variables to minimise AIC
    if(verbose) {print("Minimising AIC...")}
    model=step(model,trace=0)
    nVariables = length(attr(terms(model),"variables"))-2
    accuracy = LmAccuracy(model,classifier)
    if(verbose) {print(paste("Remaining Variables:",nVariables,"AIC:",
                             model$aic,"Accuracy:",accuracy))}
    output=rbind(output,cbind(model$aic,accuracy,maxVIF,nVariables,c(model$formula)))
    
    # Drop the variable with the maximum VIF
    if(nVariables<2) break
    vifVec = vif(model)
    maxVIF = max(vifVec)
    # vif() output changes if there are factor variables in the model.
    # Adjust for this.
    if(is.null(dim(vifVec))) { # No factor
      varToDrop = names(vifVec)[which.max(vifVec)]
    } else { # Factor
      varToDrop = names(vifVec[,1])[which.max(vifVec[,1])]
    } 
    if(verbose) {print(paste0("Minimising Co-linearity - Dropping ",varToDrop))}
    model = update(model,paste0(".~.-",varToDrop)) # Drop it
    accuracy = LmAccuracy(model,classifier)
    nVariables = length(model$coefficients)-1
    if(verbose) {print(paste("Remaining Variables:",nVariables,"AIC:",
                             model$aic,"Accuracy:",accuracy))}
    output=rbind(output,cbind(model$aic,accuracy,maxVIF,nVariables,c(model$formula)))
    if(length(vifVec)<2) break
  }
  return(output)
}

# =======================================================
# Function to do a v-fold cross validations (v different ways of splitting
# the data into training and testing). This code is adopted from:
# https://www.stat.berkeley.edu/~s133/Class2a.html
LogisticCrossVal = function(nIter,v,formula,data){
  accuracyVec = rep(0,nIter)
  for (i in seq(nIter)) {
    # Split the data into training and testing.
    # It will assign each core into one of nFold groups. When it's this folds turn
    # the cores in this fold will be the testing set.
    nSamples = nrow(data)
    grps = cut(1:nSamples,nFolds,labels=FALSE)[sample(1:nSamples)]
    
    # Do the validation
    pred = lapply(1:nFolds,function(i,formula,data){
      omit = which(grps == i)
      z = glm(formula,family=binomial(link='logit'),data=data[-omit,])
      predictions = predict(z,data[omit,],type='response')
      predictions = ifelse(predictions > 0.5,1,0)
      ClasificError = 1-mean(predictions != data[omit,]$PtSnty)
    },formula,data)
    
    accuracyVec[i] = mean(unlist(pred))
  }
  
  return(accuracyVec)
}

# =======================================================
# Function to plot the coefficients of a model together with their standard errors as a bar plot.
PlotCoefficients = function(model,confLevel=0.95,yLim=c(-25,25),yPos=20,starSize=7,errBarWidth=.8) {
  logitMCoeffs = model$coefficients
  logitMStdErrs = summary(model)$coefficients[,2]
  
  # Normalise
  normFact = sum(abs(logitMCoeffs))
  logitMCoeffs = logitMCoeffs*100/normFact
  logitMStdErrs = logitMStdErrs*100/normFact
  confIntLogitCoeffs = data.frame(MarkerLabels=names(logitMCoeffs)[2:length(logitMCoeffs)],
                                  MeanCoeff=as.numeric(logitMCoeffs[2:length(logitMCoeffs)]),
                                  SE=logitMStdErrs[2:length(logitMStdErrs)],
                                  CI=rep(0,length(logitMStdErrs[2:length(logitMStdErrs)])))
  
  # Compute the confidence interval
  nSamples = nrow(model$data)
  confIntLogitCoeffs$CI = qt(confLevel/2+.5,nSamples)*confIntLogitCoeffs$SE
  
  # Order by size of contribution
  confIntLogitCoeffs$MarkerLabels = factor(confIntLogitCoeffs$MarkerLabels,
                                           levels = confIntLogitCoeffs$MarkerLabels[
                                             order(confIntLogitCoeffs$MeanCoeff,decreasing=F)
                                             ])
  
  # Plot
  p=ggplot(confIntLogitCoeffs,aes(x=MarkerLabels,y=MeanCoeff,fill=MeanCoeff)) +
    geom_bar(position=position_dodge(0.9), stat="identity") +
    geom_errorbar(aes(ymin=MeanCoeff-CI, ymax=MeanCoeff+CI),
                  width=errBarWidth,                    # Width of the error bars
                  position=position_dodge(0.9)) +
    theme_bw() +
    ylim(yLim) +
    ylab("Relative size of the coefficient (in %)") +
    xlab("") +
    scale_fill_gradient(low="red",high="green4") +
    ggtitle(paste("Coefficients of Reduced Logistic Model",sep=""))
  
  # Add stars to indicate significance
  pValVec = summary(model)$coefficients[,4]
  for(marker in confIntLogitCoeffs$MarkerLabels) {
    if (pValVec[marker] < 0.01) {
      p = p + annotate("text", x = marker, y = yPos,
                       label = "**", size = starSize)
    } else if (pValVec[marker] < 0.05) {
      p = p + annotate("text", x = marker, y = yPos,
                       label = "*", size = starSize)
    }
  }
  
  # Flip the axes
  p = p + coord_flip()
  return(p)
}