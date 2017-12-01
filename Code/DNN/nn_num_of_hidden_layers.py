import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.layers import Input
from keras.models import Model
import csv
import sys

f5 = open("nn_num_of_hidden_layers_result.txt", 'w')
sys.stdout = f5

def neural_train(layer1 = 1,layer2 = 1,layer3 = 1,layers = 1):
	#input_tensor = Input(shape=(18,))
	#x = Dense(units = layer1,activation='relu')(input_tensor)
		# create model
	model = Sequential()
	model.add(Dense(layer1, input_dim=18, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
	#model.add(Dropout(0.2))
	#model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

	if layers == 2:
		model.add(Dense(layer2,activation = 'relu'))
	if layers ==3 :
		model.add(Dense(layer2,activation = 'relu'))
		model.add(Dense(layer3,activation = 'relu'))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

	# model = Model(input_tensor,output_tensor)
	model.compile(optimizer = 'rmsprop', loss='binary_crossentropy',metrics = ['accuracy'])
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

layer1 = [10, 20]
layer2 = [5, 8]
layer3 = [3, 4]
epochs = [50]
layers = [2,3]
param_grid = dict(epochs = epochs,layer1 = layer1,layer2 = layer2,layer3 = layer3,layers=layers)
model = KerasClassifier(build_fn = neural_train)
gsv_model = GridSearchCV(model,param_grid=param_grid, scoring="roc_auc")
grid_result = gsv_model.fit(X,Y)

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
