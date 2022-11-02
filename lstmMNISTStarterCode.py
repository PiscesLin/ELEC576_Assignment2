import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import matplotlib.pyplot as plt

if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# Load MNIST dataset,
# tensorflow 2.10.0 doesn't have examples.tutorials.mnist,
# we need to download it from earlier version.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learningRate = 1e-3
trainingIters = 100000
batchSize = 10
displayStep = 100

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 256 #number of neurons for the RNN, here I choose 256 with GRU and Adam optimizer to acquire the best accuracy.
nClasses = 10

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(value = x, num_or_size_splits = nSteps, axis = 0)

	# uncomment, if you want to use RNN
	# rnnCell = rnn_cell.BasicRNNCell(nHidden)
	# outputs, states = tf.compat.v1.nn.static_rnn(rnnCell, x, dtype=tf.float32)

	# uncomment, if you want to use LSTM
	# lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
	# outputs, states = tf.compat.v1.nn.static_rnn(lstmCell, x, dtype=tf.float32)

	# uncomment, if you want to use GRU
	gruCell = rnn_cell.GRUCell(nHidden)
	outputs, states = tf.compat.v1.nn.static_rnn(gruCell, x, dtype = tf.float32)

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.losses.softmax_cross_entropy(logits = pred, onehot_labels = y)

# optimizer = tf.train.MomentumOptimizer(learning_rate = learningRate, momentum = 0.5).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate = learningRate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()

acc_list = list()
loss_list = list()

with tf.Session() as sess:
	sess.run(init)
	step = 1

	while step * batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize)
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict = {x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = accuracy.eval(feed_dict = {x: batchX, y: batchY})
			loss = cost.eval(feed_dict = {x: batchX, y: batchY})
			print("Iter " + str(step * batchSize) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
			acc_list.append(acc)
			loss_list.append(loss)
		step += 1
	print('Optimization finished')

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels
	print("Testing Accuracy:", sess.run(accuracy, feed_dict = {x: testData, y: testLabel}))

'''
# fig, ax = plt.subplots()
# fig, bx = plt.subplots()
# ax.plot(range(len(acc_list)), loss_list, 'r', label = 'RNN Trainig Loss for Momentum Optimizer')
# ax.legend(loc = 'upper right')
# bx.plot(range(len(acc_list)), acc_list, 'c', label = 'RNN Trainig Accuracy for Momentum Optimizer')
# bx.legend(loc = 'lower right')

# ax.plot(range(len(acc_list)), loss_list, 'r', label = 'RNN Trainig Loss for Adagrad Optimizer')
# ax.legend(loc = 'upper right')
# bx.plot(range(len(acc_list)), acc_list, 'c', label = 'RNN Trainig Accuracy for Adagrad Optimizer')
# bx.legend(loc = 'lower right')

# ax.plot(range(len(acc_list)), loss_list, 'r', label = 'RNN Trainig Loss for Adam Optimizer')
# ax.legend(loc = 'upper right')
# bx.plot(range(len(acc_list)), acc_list, 'c', label = 'RNN Trainig Accuracy for Adam Optimizer')
# bx.legend(loc = 'lower right')
# plt.show()
'''

# uncomment, if you want to see the plot result of RNN
# fig, ax = plt.subplots()
# fig, bx = plt.subplots()
# ax.plot(range(len(acc_list)), loss_list, 'r', label = 'RNN Trainig Loss')
# ax.legend(loc = 'upper right', shadow = True, fontsize = 'x-large')
# bx.plot(range(len(acc_list)), acc_list, 'c', label = 'RNN Trainig Accuracy')
# bx.legend(loc = 'lower right', shadow = True, fontsize = 'x-large')
# plt.show()

# uncomment, if you want to see the plot result of LSTM
# fig, ax = plt.subplots()
# fig, bx = plt.subplots()
# ax.plot(range(len(acc_list)), loss_list, 'r', label = 'LSTM Trainig Loss')
# ax.legend(loc = 'upper right', shadow = True, fontsize = 'x-large')
# bx.plot(range(len(acc_list)), acc_list, 'c', label = 'LSTM Trainig Accuracy')
# bx.legend(loc = 'lower right', shadow = True, fontsize = 'x-large')
# plt.show()

# uncomment, if you want to see the plot result of GRU
fig, ax = plt.subplots()
fig, bx = plt.subplots()
ax.plot(range(len(acc_list)), loss_list, 'r', label = 'GRU Trainig Loss')
ax.legend(loc = 'upper right', shadow = True, fontsize = 'x-large')
bx.plot(range(len(acc_list)), acc_list, 'c', label = 'GRU Trainig Accuracy')
bx.legend(loc = 'lower right', shadow = True, fontsize = 'x-large')
plt.show()