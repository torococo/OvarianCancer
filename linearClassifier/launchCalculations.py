# ========================================================
# Script to launch the linear classifier scripts on the server.
# ========================================================

from subprocess import Popen
import os

# ========================================================
# Helper Functions
def ClusterRun(nodes,ppn,mem,wallTime,jobName,runCommands,filePath):
  pbsStr="#!/bin/sh\n#PBS -l nodes="+str(nodes)+":ppn="+str(ppn)+",mem="+str(mem)+"gb"+",walltime="+wallTime+"\n"+runCommands
  runFile=open(filePath,'w')
  runFile.write(pbsStr)
  Popen("qsub "+"-N"+jobName+" "+filePath,shell=True)

def naiveLinearRunBatch25(iSet):
  # Define the run parameters
  batchSize = 25
  nEpochs = 25
  scriptName = "naiveLinearClassifierJob_"+str(i)+"_BatchSize_"+str(batchSize)
  jobName =  "nLC"+str(i)+"_BatchSize_"+str(batchSize)
  print("Launching Job: "+jobName)

  # Set up the environment
  outDir = "naiveLinearClassifier/naiveClassifier_BatchSize_"+str(batchSize)
  if not os.path.exists(outDir):
    os.makedirs(outDir)
    os.makedirs(outDir+"/models")
    os.makedirs(outDir+"/trainingLogFiles")
    os.makedirs(outDir+"/jobFiles")

  # Prepare the pbs run file
  runCode="cd /home/80014744/fluidigmProject/linearClassifier/\n"
  runCode+="LD_LIBRARY_PATH=\"$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/\" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python` naiveLinearClassifier.py "+str(iSet)+" "+str(batchSize)+" "+str(nEpochs)+" "+outDir
  ClusterRun(1,1,5,"1:00:00",jobName,runCode,"/home/80014744/fluidigmProject/linearClassifier/naiveLinearClassifier/naiveClassifier_BatchSize_"+str(batchSize)+"/jobFiles/"+scriptName+".qsub")

def naiveLinearRunBatch10(iSet):
  # Define the run parameters
  batchSize = 10
  nEpochs = 25
  scriptName = "naiveLinearClassifierJob_"+str(i)+"_BatchSize_"+str(batchSize)
  jobName =  "nLC"+str(i)+"_BatchSize_"+str(batchSize)
  print("Launching Job: "+jobName)

  # Set up the environment
  outDir = "naiveLinearClassifier/naiveClassifier_BatchSize_"+str(batchSize)
  if not os.path.exists(outDir):
    os.makedirs(outDir)
    os.makedirs(outDir+"/models")
    os.makedirs(outDir+"/trainingLogFiles")
    os.makedirs(outDir+"/jobFiles")

  # Prepare the pbs run file
  runCode="cd /home/80014744/fluidigmProject/linearClassifier/\n"
  runCode+="LD_LIBRARY_PATH=\"$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/\" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python` naiveLinearClassifier.py "+str(iSet)+" "+str(batchSize)+" "+str(nEpochs)+" "+outDir
  ClusterRun(1,1,5,"1:00:00",jobName,runCode,"/home/80014744/fluidigmProject/linearClassifier/naiveLinearClassifier/naiveClassifier_BatchSize_"+str(batchSize)+"/jobFiles/"+scriptName+".qsub")

def naiveLinearRunBatch10Long(iSet):
  # Define the run parameters
  batchSize = 10
  nEpochs = 100
  scriptName = "naiveLinearClassifierJob_"+str(i)+"_BatchSize_"+str(batchSize)+"_nEpochs_"+str(nEpochs)
  jobName =  "nLC"+str(i)+"_BatchSize_"+str(batchSize)+"_nEpochs_"+str(nEpochs)
  print("Launching Job: "+jobName)

  # Set up the environment
  outDir = "naiveLinearClassifier/naiveClassifier_BatchSize_"+str(batchSize)+"_nEpochs_"+str(nEpochs)
  if not os.path.exists(outDir):
    os.makedirs(outDir)
    os.makedirs(outDir+"/models")
    os.makedirs(outDir+"/trainingLogFiles")
    os.makedirs(outDir+"/jobFiles")

  # Prepare the pbs run file
  runCode="cd /home/80014744/fluidigmProject/linearClassifier/\n"
  runCode+="LD_LIBRARY_PATH=\"$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/\" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python` naiveLinearClassifier.py "+str(iSet)+" "+str(batchSize)+" "+str(nEpochs)+" "+outDir
  ClusterRun(1,1,5,"2:00:00",jobName,runCode,"/home/80014744/fluidigmProject/linearClassifier/"+outDir+"/jobFiles/"+scriptName+".qsub")

def naiveLinearRunBatch25Long(iSet):
  # Define the run parameters
  batchSize = 25
  nEpochs = 100
  scriptName = "naiveLinearClassifierJob_"+str(i)+"_BatchSize_"+str(batchSize)+"_nEpochs_"+str(nEpochs)
  jobName =  "nLC"+str(i)+"_BatchSize_"+str(batchSize)+"_nEpochs_"+str(nEpochs)
  print("Launching Job: "+jobName)

  # Set up the environment
  outDir = "naiveLinearClassifier/naiveClassifier_BatchSize_"+str(batchSize)+"_nEpochs_"+str(nEpochs)
  if not os.path.exists(outDir):
    os.makedirs(outDir)
    os.makedirs(outDir+"/models")
    os.makedirs(outDir+"/trainingLogFiles")
    os.makedirs(outDir+"/jobFiles")

  # Prepare the pbs run file
  runCode="cd /home/80014744/fluidigmProject/linearClassifier/\n"
  runCode+="LD_LIBRARY_PATH=\"$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/\" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python` naiveLinearClassifier.py "+str(iSet)+" "+str(batchSize)+" "+str(nEpochs)+" "+outDir
  ClusterRun(1,1,5,"2:00:00",jobName,runCode,"/home/80014744/fluidigmProject/linearClassifier/"+outDir+"/jobFiles/"+scriptName+".qsub")


# ========================================================

# Run it

nTrainingPermutations = 10
for i in range(1,nTrainingPermutations+1):
  # naiveLinearRunBatch10(i)
  # naiveLinearRunBatch25(i)
  naiveLinearRunBatch10Long(i)
  naiveLinearRunBatch25Long(i)