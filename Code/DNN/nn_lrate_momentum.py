# Use scikit-learn to grid search the learning rate and momentum
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import csv
import sys

f5 = open("nn_lrate_momentum_result.txt", 'w')
sys.stdout = f5

# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, momentum=0):
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=18, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	optimizer = SGD(lr=learn_rate, momentum=momentum)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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

print np.shape(X)

# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring="roc_auc")
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