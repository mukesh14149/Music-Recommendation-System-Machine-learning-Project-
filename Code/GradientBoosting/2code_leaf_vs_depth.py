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


f5 = open("output_leaf-depth.txt", 'w')
sys.stdout = f5


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
	






train = pd.read_csv('./../../trainDatafinalData.csv')
x = train.drop(['target'], axis=1)
y = np.array(train['target'])

test = pd.read_csv('./../../testDatafinalData.csv')
x_test=test.drop(['id'], axis=1)



max_depth=[8, 13, 16]
min_samples_leaf= [20,100,150]


print "max_depth",max_depth
print "min_samples_leaf",min_samples_leaf

parameters={'min_samples_leaf' : min_samples_leaf,'max_depth':max_depth }

model_gb = GradientBoostingClassifier(learning_rate=0.1,subsample=0.8,random_state=10,n_estimators=300,max_features='auto')
model=GridSearchCV(model_gb,parameters,scoring='roc_auc',)
model.fit(x, y)
print model.best_params_
print model.grid_scores_
print model.cv_results_

plot_grid_search(model.cv_results_,max_depth,min_samples_leaf,'max_depth','min_samples_leaf')
















'''
score_sqrt=[]
score_auto=[]
score_log2=[]
for a in model.grid_scores_:
	if(a[0]['max_features']=='log2'):
		score_log2.append(a[1])

for a in model.grid_scores_:
	if(a[0]['max_features']=='auto'):
		score_auto.append(a[1])
for a in model.grid_scores_:
	if(a[0]['max_features']=='sqrt'):
		score_sqrt.append(a[1])

		

score_sqrt=np.array(score_sqrt).reshape(len(min_samples_leaf),len(max_depth))
score_auto=np.array(score_auto).reshape(len(min_samples_leaf),len(max_depth))
score_log2=np.array(score_log2).reshape(len(min_samples_leaf),len(max_depth))




print "shape",np.shape(x)
print "shape",np.shape(y)
print "leaf shape:",np.shape(min_samples_leaf)



_,ax=plt.subplots(1,1)
for id,val in enumerate(max_depth):
	print "loop:",score_sqrt[id,:],"--",str(val)
	ax.plot(min_samples_leaf,score_sqrt[id,:],'-o',label='max_depth :'+str(val))

ax.set_title("GS_score(max_features=sqrt)", fontsize=20, fontweight='bold')
ax.set_xlabel('min_sample_split', fontsize=12)
ax.set_ylabel('ROC value', fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid('on')
plt.savefig('leaf_vs_depth_vs_sqrt.png')





_,ax=plt.subplots(1,1)
for id,val in enumerate(max_depth):
	ax.plot(min_samples_leaf,score_auto[id,:],'-o',label='max_depth :'+str(val))

ax.set_title("GS_score(max_features=auto)", fontsize=20, fontweight='bold')
ax.set_xlabel('min_sample_split', fontsize=12)
ax.set_ylabel('ROC value', fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid('on')
plt.savefig('leaf_vs_depth_vs_auto.png')





_,ax=plt.subplots(1,1)
for id,val in enumerate(max_depth):
	ax.plot(min_samples_leaf,score_log2[id,:],'-o',label='max_depth :'+str(val))

ax.set_title("GS_score(max_features=log2)", fontsize=20, fontweight='bold')
ax.set_xlabel('min_sample_split', fontsize=12)
ax.set_ylabel('ROC value', fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid('on')
plt.savefig('leaf_vs_depth_vs_log2.png')


y_pred = model.predict(x_test)
print "creating output for DTC"
c=0
with open('leaf_vs_depth_output.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		f.write(str(i))
		f.write("\n")
		c=c+1



'''
