# ========================================================
# Proof of concept 2 layer CNN on the fluidigm data
# ========================================================
# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Global Constants
IMG_DIMS = 600 # Dimensions of the input images
N_STAINS = 37 # Number of stains in the image

# ========================================================
# Load data
# Generate fake pickled data
# for i in range(1,10):
#     print('Processing Core '+str(i))
#     tmpArr = (np.ndarray((IMG_DIMS,IMG_DIMS,N_STAINS))+1)*i
#     labels = ['88Sr-SrBCK(Sr88Di)', '101Ru-RR101(Ru101Di)', '102Ru-RR102(Ru102Di)', '115In-AvantiLipid(In115Di)', '134Xe-XeBCK(Xe134Di)', '141Pr-CD196(Pr141Di)', '142Nd-CD19(Nd142Di)', '143Nd-Vimentin(Nd143Di)', '145Nd-CD163(Nd145Di)', '147Sm-CD20(Sm147Di)', '148Nd-CD16(Nd148Di)', '149Sm-CD25(Sm149Di)', '150Nd-p53(Nd150Di)', '151Eu-CD134(Eu151Di)', '152Sm-CD45(Sm152Di)', '153Eu-CD44s(Eu153Di)', '154Gd-CD14(Gd154Di)', '155Gd-FoxP3(Gd155Di)', '156Gd-CD4(Gd156Di)', '158Gd-E-cadherin(Gd158Di)', '159Tb-p21(Tb159Di)', '161Dy-CD152(Dy161Di)', '162Dy-CD8a(Dy162Di)', '164Dy-CD11b(Dy164Di)', '165Ho-Beta-catenin(Ho165Di)', '166Er-B7-H4(Er166Di)', '168Er-Ki67(Er168Di)', '169Tm-CollagenI(Tm169Di)', '170Er-CD3(Er170Di)', '171Yb-CD68(Yb171Di)', '172Yb-PD-L2(Yb172Di)', '173Yb-B7-H3(Yb173Di)', '174Yb-HLA-DR(Yb174Di)', '175Lu-pS6(Lu175Di)', '176Yb-HistoneH3(Yb176Di)', '191Ir-DNA191(Ir191Di)', '193Ir-DNA193(Ir193Di)']
#     sensitivity = np.zeros(2)
#     sensitivity[i%2] = 1
#     data = {'Markerlabels': labels, 'Image': tmpArr, 'PtSensitive': sensitivity}
#     out=open("DummyProcessedData/core"+str(i)+".p","wb")
#     pickle.dump(data,out)
#     out.close()

# def LoadCore(coreId):
#   return pickle.load(open("DummyProcessedData/core"+str(coreId)+".p","rb"))

# ========================================================
# Define the CNN graph

# Helper Functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 1st convolutional layer
x = tf.placeholder(tf.float32, shape=[None, IMG_DIMS * IMG_DIMS * N_STAINS], name="x")

nFilters_conv1 = 10
W_conv1 = weight_variable([5, 5, N_STAINS, nFilters_conv1]) # Filters of the 1st layer
b_conv1 = bias_variable([nFilters_conv1]) # Bias in the 1st layer

x_image = tf.reshape(x, [-1, IMG_DIMS, IMG_DIMS, N_STAINS])
# print(x_image.get_shape())

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd convolutional layer
nFilters_conv2 = 10
W_conv2 = weight_variable([5, 5, nFilters_conv1, nFilters_conv2])
b_conv2 = bias_variable([nFilters_conv2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 1st fully connected layer
# print(h_pool2.get_shape())
W_fc1 = weight_variable([int(IMG_DIMS/4) * int(IMG_DIMS/4) * nFilters_conv2, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, int(IMG_DIMS/4) * int(IMG_DIMS/4) * nFilters_conv2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32, name="keepProb")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd fully connected layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define the error function
y_ = tf.placeholder(tf.float32, shape=[None, 2], name="CorrectLabels") # Correct labels
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction = tf.argmax(y_conv,1)
# ========================================================
# Train the network
# Function to get patient outcome from coreId
def getPatientOutcome(coreId, outcomeArr):
    ptSensitive = outcomeArr[np.where(outcomeArr[:,0] == coreId),1]
    if ptSensitive.size==0:
        ptSensitiveVec = np.ones(2)*(-1) # Return -1 as to indicate no outcome available
    else:
        ptSensitiveVec = np.zeros(2)
        ptSensitiveVec[int(ptSensitive[0][0])] = 1
    return ptSensitiveVec

# Function to partition training data into batches.
# Returns a list of arrays, where each array is one batch
# and the array contains the indices for the images in that batch
def partitionTrainingData(batchSize, trainingSetIdxVec):
    if trainingSetIdxVec.shape[0] % batchSize != 0: raise Exception("batch size does not divide evenly by input size!")
    np.random.shuffle(trainingSetIdxVec)
    partionedData = []
    nBatches = int(len(trainingSetIdxVec)/batchSize)
    for i in range(nBatches):
        batch = trainingSetIdxVec[i * batchSize:(i + 1) * batchSize]
        partionedData.append(batch)
    return partionedData

# Function to load a single core to memory
def LoadCore(coreId):
    inName = "Data/patientsWithOutcomes/npArraysCropped/core_"+str(coreId)+"_Cropped"+".npy"
    return np.load(inName)

# Function to load a batch to memory. Returns a tuple: ([Flattened Images], [Labels])
def loadBatch(batchIdxVec, outcomeArr):
    batchSize = len(batchIdxVec)
    batch = ([np.zeros(IMG_DIMS*IMG_DIMS*N_STAINS) for _ in range(batchSize)], [np.zeros(2) for _ in range(batchSize)])
    for i, imgIdx in enumerate(batchIdxVec):
        coreId = int(outcomeArr[imgIdx,0])
        image = LoadCore(coreId)
        ptSensitiveVec = getPatientOutcome(coreId, outcomeArr)
        batch[0][i] = np.copy(image.flatten())
        batch[1][i] = np.copy(ptSensitiveVec)

    return batch

outcomeArr  = np.genfromtxt('data/patientsWithOutcomes/coreToSensitivityMap_122Cores.csv', delimiter=',') # Get Array with patient outcomes
trainingSetIdxVec = np.arange(1,81)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#
# tmpArr = (np.ndarray((IMG_DIMS,IMG_DIMS,N_STAINS),dtype=np.float32)+1)*1
# tmp = tmpArr.reshape((1,IMG_DIMS*IMG_DIMS*N_STAINS))
# dres = sess.run(prediction, {"x:0": tmp, "keepProb:0": 0.5})


for i in range(10):
    batchIdxVec = partitionTrainingData(4, trainingSetIdxVec)

    for ii,ibatch in enumerate(batchIdxVec):
        batch = loadBatch(ibatch, outcomeArr)
        _,err = sess.run([train_step,accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print("Iteration: %s, Training Accuracy: %s"%(ii,err))
#
#
#
# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# if i%1 == 0:
#   train_accuracy = accuracy.eval(feed_dict={
#       x:batch[0], y_: batch[1], keep_prob: 1.0})
#   print("step %d, training accuracy %g"%(i, train_accuracy))