import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import cPickle
from sklearn.cross_validation import *
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import csv
import sys
f5 = open("lightgbm13.txt", 'w')
sys.stdout = f5

# load dataset
with open('../../trainDatafinalData.csv', 'r') as inf:
  data = list(csv.reader(inf, skipinitialspace=True))
  data = np.array(data)
  data = data[1:]
  X = data[:,:-1]
  Y = data[:,-1]
  X = X.astype(np.float)
  Y = Y.astype(np.int)

print np.shape(X)
print "Data loading complete"

# X = []
# tempY = []
# with open('seeds_dataset.txt') as f:
#   for line in f:
#     content = line.split()
#     temp = []
#     for i in range(0,len(content)-1):
#       temp.append(float(content[i]))

#     X.append(temp)
#     tempY.append(content[len(content)-1])

# X = np.array(X)   
# Y = tempY

lgb_model = lgb.LGBMClassifier()

parameters = {'boosting_type':['gbdt'],
              'learning_rate': [0.1],
              'num_leaves': [2**8], ##num_leaves we should let it be smaller than 2^(max_depth)
              'max_depth' : [14],  ##result in lightgbm3 which set numleave and max depth 
              #'min_data_in_leaf' :[100], ##Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset.
              'verbosity' : [0],
              'metric' : ['auc'],
              'max_bin': [228],  #Use small for faster, for better accuracy take it large
              'num_boost_round': [800],
              'bagging_fraction': [1],
              'bagging_freq': [1],
              'min_child_weight': [16],
              'bagging_seed': [1], 
              'feature_fraction': [1],
              'feature_fraction_seed': [1],
              'n_estimators': [50]} #number of trees, change it to 1000 for better results


clf = GridSearchCV(lgb_model, parameters, n_jobs=-1, 
                   cv=StratifiedKFold(Y, n_folds=2, shuffle=True),
                   verbose=0, refit=True)
grid_result = clf.fit(X, Y)




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


with open('lg_grd.pkl', 'wb') as fid:
    cPickle.dump(grid_result, fid)

test = pd.read_csv('../../testDatafinalData.csv')
x_test=test.drop(['id'], axis=1)

y_pred = grid_result.predict(x_test)
print "creating output for DTC"
c=0
with open('output_LGB.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		f.write(str(i))
		f.write("\n")
		c=c+1

'''

X_train, X_test,y_train, y_test=train_test_split(x, y, test_size=0.30, random_state=200)
dt_fit = model.fit(X_train, y_train)
y_pred=model.predict(X_test)

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for DecisionTree')
plt.plot(fpr,tpr,label="DecisionTree ROC, auc="+str(auc))
plt.legend(loc=0)
plt.savefig("ROC_curve_decisionTree.png")
'''

