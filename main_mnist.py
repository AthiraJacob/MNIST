
''' 
Main function to run. Builds a CNN to classify MNIST data
Adapted from https://www.tensorflow.org/tutorials/mnist/pros/ 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import os
import argparse
import sys
sess = tf.InteractiveSession()
import input_data

from network import net
from confuseMatrix import create_confMat

curr_dir = '/cis/home/ajacob/Documents/cnn/'

# Parse input arguments
FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=curr_dir + 'data/',
                      help='Directory for storing input data')
parser.add_argument('--nEpochs', type=int, default = 1, help = 'Number of epochs to run for')
parser.add_argument('--imgNoise', type=float, default = 0, help = 'Std of gaussian noise to be added to training images')
parser.add_argument('--labelNoise', type=int, default = 0, help = 'Percentage of training image labels to randomize')

FLAGS, unparsed = parser.parse_known_args()

#read data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, img_noisestd = FLAGS.imgNoise, label_noise = FLAGS.labelNoise)

#Build network
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
y_conv = net(x,keep_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

n_epochs = mnist.train.epochs_completed
i = 0


while(n_epochs < FLAGS.nEpochs):
  batch = mnist.train.next_batch(50) # Obtain mini-batches as tuples of the form (imgs,labels): ([50x784],[50x10])
  
  if i%100 == 0:  #Monitor train accuracy 
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) #Train
  i+=1

  if mnist.train.epochs_completed == n_epochs+1: #Monitor test accuracy at every epoch
  	n_epochs = mnist.train.epochs_completed
  	test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  	print("% ==== Iteration" + str(n_epochs) + " : Test accuracy = " + str(test_accuracy)+ " ==== %") 

  	cm,class_wise = create_confMat(y_conv.eval(feed_dict={x: mnist.test.images, keep_prob : 1.0}),mnist.test.labels)
  	print("Confusion Matrix: ")
  	print(cm)
  	print("Class-wise accuracy:")
  	print(class_wise)
  	print("**======================================**")


