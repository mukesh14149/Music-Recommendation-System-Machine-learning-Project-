# Use scikit-learn to grid search the batch size and epochs
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import csv
import sys
import random

f5 = open("nn_batch_epoch_result.txt", 'w')
sys.stdout = f5
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
with open('subset.csv', 'rb') as inf:
	data = list(csv.reader(inf, skipinitialspace=True))
	print np.shape(data)
	data = np.array(data)
	data = data[1:1000000]
	X = data[:,:-1]
	Y = data[:,-1]
	X = X.astype(np.float)
	Y = Y.astype(np.int)

print np.shape(X)

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
# batch_size = [80]
# epochs = [50]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring="roc_auc", return_train_score=True)
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