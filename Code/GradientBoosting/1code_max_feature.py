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


f5 = open("output_max_features.txt", 'w')
sys.stdout = f5


train = pd.read_csv('./../../trainDatafinalData.csv')
x = train.drop(['target'], axis=1)[:500]
y = np.array(train['target'])[:500]

test = pd.read_csv('./../../testDatafinalData.csv')
x_test=test.drop(['id'], axis=1)[:500]



#max_depth=[8, 13, 16]
#min_samples_leaf= [20,100,150]
max_features=['auto','sqrt','log2']


parameters={'max_features':max_features }

model_gb = GradientBoostingClassifier(learning_rate=0.1,subsample=0.8,random_state=10,n_estimators=300,max_depth=13,min_samples_leaf=120)
model=GridSearchCV(model_gb,parameters,scoring='roc_auc',)
model.fit(x, y)
print model.best_params_
print model.grid_scores_


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

