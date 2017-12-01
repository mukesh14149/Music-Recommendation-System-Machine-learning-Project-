# Use scikit-learn to grid search the dropout rate
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

import csv
import sys

f5 = open("nn_droput_regularization_result.txt", 'w')
sys.stdout = f5

# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(40, input_dim=18, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
with open('subset.csv', 'rb') as inf:
	data = list(csv.reader(inf, skipinitialspace=True))
	data = np.array(data)
	data = data[1:1000000]
	X = data[:,:-1]
	Y = data[:,-1]
	X = X.astype(np.float)
	Y = Y.astype(np.int)
	
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
mean_train_scores = grid_result.cv_results_['mean_train_score']
std_train_scores = grid_result.cv_results_['std_train_score']
mean_fit_times = grid_result.cv_results_['mean_fit_time']
mean_score_times = grid_result.cv_results_['mean_score_time']

for mean, stdev, param, mean_train_score, std_train_score, mean_fit_time, mean_score_time in zip(means, stds, params, mean_train_scores, std_train_scores, mean_fit_times, mean_score_times):
    print("%f (%f) with: %r extra info => mean_train_score %f, std_train_score %f, mean_fit_time %f, mean_score_time %f" % (mean, stdev, param, mean_train_score, std_train_score, mean_fit_time, mean_score_time))