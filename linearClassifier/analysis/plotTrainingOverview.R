# ===============================================================
# Script to generate summary plots for the linear classifier benchmarking runs.
# ===============================================================
dataDir = "~/Desktop/linearClassifierAnalysis/naiveClassifier_BatchSize_25/trainingLogFiles"
outName = "TrainingOverview_naiveClassifier25"
outDir = "~/Desktop/linearClassifierAnalysis/"
library(ggplot2)
source("multiplot.R")
setwd(dataDir)


# Open the pdf
pdf(paste(outDir,outName,".pdf",sep=""), onefile = TRUE)

for (fName in list.files()) {
  # Extract the data set id
  setId = as.numeric(unlist(strsplit(gsub("[^0-9]", "", unlist(fName)), "")))

  # Load the data
  trainingResults = read.csv(fName,header=F)
  names(trainingResults) = c("Epoch","Entropy","ClassificationError","TimePerEpoch","TimePerBatch")

  # Plot
  pltTitle = paste(c("Training Performance - Set", setId), collapse = " ")
  # Entropy over time
  p1 = ggplot(trainingResults,aes(trainingResults$Epoch,trainingResults$Entropy)) + geom_point() + theme_bw() + labs(x="Epoch", y="Entropy") + ggtitle(pltTitle)
  # Classification Error over time
  p2 = ggplot(trainingResults,aes(trainingResults$Epoch,trainingResults$ClassificationError)) + geom_line() + theme_bw() + scale_y_continuous(limits = c(0, 1)) + labs(x="Epoch", y="Classification Error")

  multiplot(p1,p2, cols=1)
}

dev.off()

# ===============================================================
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }

  if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}