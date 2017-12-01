import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from scipy import stats
from bhtsne import tsne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hypertools as hyp
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer,  CountVectorizer,TfidfTransformer  
from sklearn.grid_search import GridSearchCV  
from time import time 
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.naive_bayes import GaussianNB


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

data_path = './'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
member = members.drop(['registration_year', 'expiration_year'], axis=1)
members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')

train = train.fillna(-1)
test = test.fillna(-1)

# # Preprocess dataset
cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
	if train[col].dtype == 'object':
	        train[col] = train[col].apply(str)
	        test[col] = test[col].apply(str)
	
	        le = LabelEncoder()
	        train_vals = list(train[col].unique())
	        test_vals = list(test[col].unique())
	        le.fit(train_vals + test_vals)
	        train[col] = le.transform(train[col])
	        test[col] = le.transform(test[col])
	print(col + ': ' + str(len(train_vals)) + ', ' + str(len(test_vals)))

print train.head()


x = np.array(train.drop(['target'], axis=1))
y = train['target'].values



clf=SVC(probability=True,random_state=1)
clf.fit(x,y)
y_pred=clf.predict(testx)

print "creating output for SVC"
c=0
with open('output_svc.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		f.write(str(i))
		f.write("\n")
		c=c+1

print clf

#insemble-adaboost-binary-classifiacation
'''

gnb = GaussianNB()
gnb.fit(x,y)

print "fitting done!"


print x.shape
print y.shape
print test.head()

y_pred = gnb.predict(test.drop(['id'], axis=1))

print "creating output for NB"
c=0
with open('output_NB.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		f.write(str(i))
		f.write("\n")
		c=c+1



print gnb
print gnb.class_prior_




parameters = {'fit_intercept':('True', 'False'), 'normalize':('True', 'False'), 'copy_X':('True', 'False')}

model = LinearRegression()
model =GridSearchCV(LinearRegression(), parameters)
model = model.fit(x, y)

y_pred = model.predict(test.drop(['id'], axis=1))

print "creating output for LinearR"
c=0
with open('output_LiR.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		if(i > .5 ):
			i=1
		else:
			i=0
	
		f.write(str(i))
		f.write("\n")
		c=c+1

print model
print("Scores for alphas:")
print(model.grid_scores_)
print("Best estimator:")
print(model.best_estimator_)
print("Best score:")
print(model.best_score_)
print("Best parameters:")
print(model.best_params_)


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
model = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
model = model.fit(x, y)

y_pred = model.predict(test.drop(['id'], axis=1))

print "creating output for LogisticR"
c=0
with open('output_LoR1.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		if(i > .5 ):
			i=1
		else:
			i=0
	
		f.write(str(i))
		f.write("\n")
		c=c+1
c=0
with open('output_LoR2.csv','w') as f:
	f.write('id,target\n')
	for i in y_pred:
		f.write(str(c))
		f.write(",")
		f.write(str(i))
		f.write("\n")
		c=c+1


print model
print("Scores for alphas:")
print(model.grid_scores_)
print("Best estimator:")
print(model.best_estimator_)
print("Best score:")
print(model.best_score_)
print("Best parameters:")
print(model.best_params_)

'''

