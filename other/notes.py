# ========================================================
# Code snipptes about different things during the project.
# ========================================================
# --------------------------------------------------------
# Accessing the sessions and different parts of the graph in Rafael's API
myNet.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) # Show a list of all variables in the graph
sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) # Show a list of all variables in the graph of the current session
myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0] # Access an individual variable name
dum = myInterface.sess.run(myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) # Get the values of all tensors in a CNN as a long list of arrays
len(dum[0]) # length of the first tensor
print(dum[0][0,0,0,0]) # prints an element from that tensor
print(myInterface.sess.run(myNet.OutputLayerTF, feed_dict={'inputsPL'+":0":inputArr[testingSet],'outputsPL'+":0":outcomeArr[testingSet],'dropoutPL'+":0":1})) # Run a specific tensor on a particular set of images


# --------------------------------------------------------
# An example of restoring a saved CNN
myNet2 = Utils.ConvolutionalNetwork(sampleArchitecture,[CANONICAL_DIMS,CANONICAL_DIMS,N_STAINS],2) # Set up a new session
myInterface2 = myNet2.CreateTFInterface()
myInterface2.StartSession()
print(dum[0][0,0,0,0]) # At this point this is a random value (because we initialised it randomly
new_saver = tf.train.import_meta_graph("trainingResults/" + MODEL_NAME+".meta") # Instantiate the saver
new_saver.restore(myInterface2.sess, "trainingResults/" + MODEL_NAME) # Loads the model
dum = myInterface2.sess.run(myNet2.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
print(dum[0][0,0,0,0]) # Now this holds the same value as the saved network
myInterface2.sess.run(myNet2.AccuracyTF,feed_dict={'inputsPL'+":0":inputArr[:5],'outputsPL'+":0":outcomeArr[:5],'dropoutPL'+":0":1}) # Run the saved CNN on an image

# --------------------------------------------------------
# Example of how to use the restorer method to the interface API
dum = myInterface.sess.run(myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) # Get the values of all tensors in a CNN as a long list of arrays
print(dum[0][0,0,0,0]) # prints an element from that tensor

myInterface.ImportGraph("trainingResults/" + MODEL_NAME)

dum = myInterface.sess.run(myNet.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) # Get the values of all tensors in a CNN as a long list of arrays
print(dum[0][0,0,0,0]) # prints an element from that tensor
