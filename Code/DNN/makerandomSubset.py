# Use scikit-learn to grid search the batch size and epochs
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import csv
import sys
from random import *

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=18, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
with open('./../data/trainDatafinalData.csv', 'rb') as inf:
	data = list(csv.reader(inf, skipinitialspace=True))

	data = np.array(data)

	subset = []
	tempDataset = set()

	le = len(data)-1
	while len(tempDataset) <2000000:
		tempDataset.add(randint(0, le))

	print np.shape(list(tempDataset))
	tempDataset = list(tempDataset)

	for i in tempDataset:
		subset.append(data[i])

	#data = data[1:1000000]
	#subset = subset[1:1000000]
	# X = subset[:,:-1]
	# Y = subset[:,-1]
	# X = X.astype(np.float)
	# Y = Y.astype(np.int)
print np.shape(subset)

with open("subset.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(subset)


