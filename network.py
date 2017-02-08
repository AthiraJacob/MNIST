'''
Network architecture
Inputs: Input batch, keep probability for dropouts
Outputs: Output from network
'''

import tensorflow as tf

def init_weights(shape):
	# Initialize weight matrix from a normal distribution
	init = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(init)

def init_biases(shape):
	# Initialize bias matrix as a constant vector
	init = tf.constant(0.1,shape = shape)
	return tf.Variable(init)

def conv2d(x, W): #Convolutional layer
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x): #Max pooling
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def net(x,keep_prob):
	x_image = tf.reshape(x, [-1,28,28,1])

    #Convolution layer 1 
	W_conv1 = init_weights([5, 5, 1, 32])
	b_conv1 = init_biases([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#Convolution layer 2
	W_conv2 = init_weights([5, 5, 32, 64])
	b_conv2 = init_biases([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# Fully connected layers
	W_fc1 = init_weights([7 * 7 * 64, 1024])
	b_fc1 = init_biases([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = init_weights([1024, 10])
	b_fc2 = init_biases([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	return y_conv