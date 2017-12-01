import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from scipy import stats
from bhtsne import tsne
from sklearn.grid_search import GridSearchCV  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hypertools as hyp
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

f5 = open("grid_search_result.txt", 'w')
sys.stdout = f5

train = pd.read_csv('./../data/trainDatafinalData.csv')
x = train.drop(['target'], axis=1)
y = np.array(train['target'])
test = pd.read_csv('./../data/testDatafinalData.csv')
x_test=test.drop(['id'], axis=1)


min_samples=range(2,19,4)
max_depth=range(1,20,4)
max_feature=['auto','sqrt','log2']


#roc_auc
parameters={'min_samples_split' : min_samples,'max_depth': max_depth , 'max_features':max_feature}

clf_tree=tree.DecisionTreeClassifier()
model=GridSearchCV(clf_tree,parameters,scoring='roc_auc')
#model = tree.DecisionTreeClassifier()
model.fit(x, y)
print model.best_params_
print model.grid_scores_

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

		
score_sqrt=np.array(score_sqrt).reshape(len(min_samples),len(max_depth))
score_auto=np.array(score_auto).reshape(len(min_samples),len(max_depth))
score_log2=np.array(score_log2).reshape(len(min_samples),len(max_depth))

_,ax=plt.subplots(1,1)
for id,val in enumerate(max_depth):
	ax.plot(min_samples,score_sqrt[id,:],'-o',label='max_depth :'+str(val))

ax.set_title("GS_score(max_features=sqrt)", fontsize=20, fontweight='bold')
ax.set_xlabel('min_sample_split', fontsize=12)
ax.set_ylabel('ROC value', fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid('on')
plt.savefig('dtc_sqrt.png')


_,ax=plt.subplots(1,1)
for id,val in enumerate(max_depth):
	ax.plot(min_samples,score_auto[id,:],'-o',label='max_depth :'+str(val))

ax.set_title("GS_score(max_features=auto)", fontsize=20, fontweight='bold')
ax.set_xlabel('min_sample_split', fontsize=12)
ax.set_ylabel('ROC value', fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid('on')
plt.savefig('dtc_auto.png')





_,ax=plt.subplots(1,1)
for id,val in enumerate(max_depth):
	ax.plot(min_samples,score_log2[id,:],'-o',label='max_depth :'+str(val))

ax.set_title("GS_score(max_features=log2)", fontsize=20, fontweight='bold')
ax.set_xlabel('min_sample_split', fontsize=12)
ax.set_ylabel('ROC value', fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.grid('on')
plt.savefig('dtc_log2.png')


y_pred = model.predict(x_test)
print "creating output for DTC"
c=0
with open('output_DT.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		f.write(str(i))
		f.write("\n")
		c=c+1

