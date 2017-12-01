# Use scikit-learn to grid search the learning rate and momentum
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import csv
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.constraints import maxnorm
from keras.layers import Dropout
import pickle

f5 = open("nn_algo_output.txt", 'w')
sys.stdout = f5

# load dataset
with open('./../data/trainDatafinalData.csv', 'rb') as inf:
	data = list(csv.reader(inf, skipinitialspace=True))
	data = np.array(data)
	data = data[1:]
	X = data[:,:-1]
	Y = data[:,-1]
	X = X.astype(np.float)
	Y = Y.astype(np.int)

print np.shape(X)


# dropout in hidden layers with weight constraint
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=18, kernel_initializer='normal', activation='softsign', kernel_constraint=maxnorm(1)))
	model.add(Dropout(0.1))
	model.add(Dense(8, kernel_initializer='normal', activation='softsign', kernel_constraint=maxnorm(1)))
	model.add(Dropout(0.1))
	model.add(Dense(3, kernel_initializer='normal', activation='softsign', kernel_constraint=maxnorm(1)))
	model.add(Dropout(0.1))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

seed = 7
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=60, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs = -1)

print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# save mode
with open('neuralNetworkModel.pkl', 'wb') as handle:
  pickle.dump(pipeline, handle)

# # create model
# model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# # define the grid search parameters
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# param_grid = dict(learn_rate=learn_rate, momentum=momentum)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring="roc_auc")
# grid_result = grid.fit(X, Y)