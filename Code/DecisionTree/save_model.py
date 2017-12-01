import arff, numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import label_binarize
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
import cPickle



train = pd.read_csv('./../../trainDatafinalData.csv')
x = train.drop(['target'], axis=1)
y = np.array(train['target'])

print "loaded data!"

model = tree.DecisionTreeClassifier(max_features='sqrt',min_samples_split=10,max_depth=13)
model.fit(x,y)

print "fitting done!"


with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(model, fid)    

# load it again
#with open('my_dumped_classifier.pkl', 'rb') as fid:
#    model = cPickle.load(fid)

model = tree.DecisionTreeClassifier(max_features='sqrt',min_samples_split=10,max_depth=13)

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

Methodology



