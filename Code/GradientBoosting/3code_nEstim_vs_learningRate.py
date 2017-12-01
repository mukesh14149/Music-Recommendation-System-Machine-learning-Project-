import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from scipy import stats
from bhtsne import tsne
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hypertools as hyp
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import tree
import sys




def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
	
	# Get Test Scores Mean and std for each grid search
	scores_mean = cv_results['mean_test_score']
	
	scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))


	# Plot Grid search scores
	_, ax = plt.subplots(1,1)

	# Param1 is the X-axis, Param 2 is represented as a different curve (color line)
	for idx, val in enumerate(grid_param_2):
	    ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

	ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
	ax.set_xlabel(name_param_1, fontsize=16)
	ax.set_ylabel('CV Average Score', fontsize=16)
	ax.legend(loc="best", fontsize=15)
	ax.grid('on')
	plt.savefig(name_param_1+'_'+name_param_2+'.png')
	





f5 = open("output_learning_rate-n_estimators.txt", 'w')
sys.stdout = f5


train = pd.read_csv('./../../trainDatafinalData.csv')
x = train.drop(['target'], axis=1)[:50]
y = np.array(train['target'])[:50]

test = pd.read_csv('./../../testDatafinalData.csv')
x_test=test.drop(['id'], axis=1)[:50]


learning_rate=[0.1, 0.05, 0.02, 0.01]
n_estimators=[400,500,600]


parameters={'learning_rate' : learning_rate,'n_estimators':n_estimators }

#first INSERT values for max_depth,min_sample_leaf,max_feature

model_gb = GradientBoostingClassifier(subsample=0.8,random_state=10)
model=GridSearchCV(model_gb,parameters,scoring='roc_auc')
model.fit(x, y)
print model.best_params_
print model.grid_scores_

plot_grid_search(model.cv_results_,learning_rate,n_estimators,'learning_rate','n_estimators')




