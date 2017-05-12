import matplotlib.pyplot as plt
import numpy as np
import Utils
from matplotlib.backends.backend_pdf import PdfPages # Library to save plots as pdfs

MODEL_NAME = "simpleNet"
resultsArr = np.genfromtxt('trainingResults/trainingError_simpleNet.csv', delimiter=',') # Get Array with patient outcomes

resultsArr

axs=Utils.GenAxs(1,1)
pp = PdfPages("trainingResults/" + MODEL_NAME + ".pdf")

# plt.title("Training Error")
plt.ylabel('Log(Cross Entropy)')
plt.xlabel('Epoch')
Utils.PlotLine(axs[0],resultsArr[:205,0],np.log(resultsArr[:205,1]),"r-")

# Save to pdf
f = plt.gcf()
pp.savefig(f)
plt.close(f)

axs=Utils.GenAxs(1,1)
plt.ylabel('Classification Error')
plt.xlabel('Epoch')
Utils.PlotLine(axs[0],resultsArr[:205,0],resultsArr[:205,2],"r-")

# Save to pdf
f = plt.gcf()
pp.savefig(f)
plt.close(f)

pp.close()
