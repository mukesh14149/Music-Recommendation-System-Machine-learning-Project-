import pandas as pandas
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import info_gain, info_gain_ratio


#Data  Loading after removing outlier and standarisation
train = pd.read_csv('\pathtofile')
X = np.array(train.drop['target'],axis=1)
y = train['target'].values


#Recursive feature elemination
estimator = ExtraTreesClassifier()
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
print selector.support_ 
print selector.ranking_


#Correlation Attribute
k = 10000000 #for k cross validation
X = X[0:5000000]
y = y[0:5000000]
int ll = len(X)/10
start = 0
end =k
elements = []
for i in range(ll):
	ranker = {}
	for i in range(0,25):
		gh = np.correlate(X[[start,end],i],y)
		ranker[i] = gh
	lis = sorted(ranker.iterkeys())	
	#take last 5 elements
	elements.append(lis[20:25])
	start  = k
	end = k+k
	print lis
#take top x features.


#CFSSubsetEval
k = 10000000 #for k cross validation
X = X[0:5000000]
y = y[0:5000000]
threshold = 0  #update this threashold 
threshold2 = 0  #update this threashold 

int ll = len(X)/10
start = 0
end =k
elements = []
for i in range(ll):
	ranker = {}
	temp = []
	#Forward selection
	for i in range(0,25):
		gh = np.correlate(X[[start,end],i],y)
		flag = True
		for i in temp:
			tt = np.correlate(X[[start,end],i],i)
			if(tt<threshold):
				flag = False		
		if flag==True && gh>threshold2:
			temp.add(X[[start,end],i])		
print elements


#Info Gain
X_new = SelectKBest(info_gain).fit_transform(X, y)
print X_new.shape
print X_new


#WrapperSubsetEval
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, y)
print clf.feature_importances_  
model = SelectFromModel(clf, prefit=True
X_new = model.transform(X)
print X_new.shape
print X_new