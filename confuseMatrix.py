'''
Calculate confusion matrix and class wise accuracies
Inputs: Predicted values, True values
Outputs: Confusion matrix, Per-class accuracy
'''

import pandas as pd
import tensorflow as tf
import numpy as np

def create_confMat(y_pred, y_true):

	# Convert from 1/0 labels to single digits
	y_true = pd.Series(tf.argmax(y_true,1).eval(), name = 'Actual')
	y_pred = pd.Series(tf.argmax(y_pred,1).eval(), name = 'Predicted')

	df_confusion = pd.crosstab(y_true,y_pred) #Create confusion matrix
	cm = df_confusion.as_matrix()

	class_wise = pd.Series(np.diag(cm)/np.sum(cm,1))  #Class wise accuracy


	return df_confusion, class_wise

