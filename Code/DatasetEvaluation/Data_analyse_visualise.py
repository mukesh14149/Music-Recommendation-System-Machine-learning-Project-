import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hypertools as hyp
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA


data_path = './'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')


# print len(train['bd'])
# data = {}
# for i in train['bd']:
#     if i in data:
#         data[i] = data[i] + 1
#     else:
#         data[i] = 1 
# print data
# #data.pop(0,None)
# print max(data, key=data.get)
# plt.figure(1)
# plt.bar(range(len(data)),data.values(),align='center')
# plt.xticks(range(len(data)),data.keys())

# cols = train.columns.tolist()
# cols.remove('msno') #remove user_id from list of columns
# cols.remove('song_id')
# new_df = train
# arr=['bd']
# for col in arr:
#     P = np.percentile(train[col],[5,95])
#     new_df = new_df.merge(train[(train[col]>P[0]) & (train[col]<P[1])],how='inner')

# print(new_df.head())
# print(new_df.shape)
# print len(new_df['bd'])
# data = {}
# for i in new_df['bd']:
#     if i in data:
#         data[i] = data[i] + 1
#     else:
#         data[i] = 1 
# print data
# #data.pop(0,None)
# print max(data, key=data.get)
# plt.figure(2)
# plt.bar(range(len(data)),data.values(),align='center')
# plt.xticks(range(len(data)),data.keys())
# plt.show()
# new_df = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]


###############################################################
#Visualise whole dataset by pca dimesnsion reduction tool
X = np.array(new_df.drop(['target'], axis=1))
y = new_df['target'].values
X = preprocessing.scale(X)
color=["#000000","#FF0000","#FFFF00","#808000", "#008000","#00FFFF","#0000FF","#800080","#FF00FF","#008080"]
#hyp.plot(X,'.',ndims=2,group=[color[i] for i in y])
X_tsne = PCA(n_components=2,svd_solver='full').fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=[color[i] for i in y])
plt.show()
###############################################################

##############################################################
#Visualise what are the songs present in songs.csv and check how many users(both in train and test)
#listen particular type of songs
count = 0
temp1 = train['msno'].tolist()
temp2 = test['msno'].tolist()
k=0
for i in temp1:
    if i not in temp2:
        count = count+1
    k=k+1
print count,len(temp1),len(temp2)
count = 0
temp1 = train['song_id'].tolist()
temp2 = songs['song_id'].tolist()
k=0
data ={}
t1 = len(np.unique(train['song_id']))
t2 = len(np.unique(songs['song_id']))
print t1,t2
for i in temp2:
    data[i] = 0
for j in temp1:
    #print k
    try:
        data[j] = data[j] + 1
    except: 
        print j   
        b = 0
    #k = k +1
for i in temp2:
    if data[i] ==0:
        count = count+1
print count,len(temp2)
#################################################################


################################################################
#Analyse description of the listener's source of music
print len(train['source_system_tab'])
data = {}
for i in train['source_system_tab']:
    if i in data:
        data[i] = data[i] + 1
    else:
        data[i] = 1 
print data
print max(data, key=data.get)
print len(train['source_screen_name'])
data = {}
for i in train['source_screen_name']:
    if i in data:
        data[i] = data[i] + 1
    else:
        data[i] = 1 
print data
print max(data, key=data.get)

print len(train['source_type'])
data = {}
for i in train['source_type']:
    if i in data:
        data[i] = data[i] + 1
    else:
        data[i] = 1 
print data
print max(data, key=data.get)
#################################################################


#################################################################
#Analyse Gender proportion in all cities
print len(members['city'])
data = {} #city female male nan
for i,j in zip(members['city'],members['gender']):
    if i in data:
        if j == 'female':
            data[i] = [data[i][0] + 1, data[i][1] + 1, data[i][2], data[i][3]]
        elif j == 'male':
            data[i] = [data[i][0] + 1, data[i][1], data[i][2]+1, data[i][3]]
        else:
            data[i] = [data[i][0] + 1, data[i][1], data[i][2], data[i][3]+1]        
    else:
        if j == 'female':
            data[i] = [1, 1, 0,0]
        elif j == 'male':
            data[i] = [1, 0, 1,0]
        else:
            data[i] = [1, 0, 1,1]    
print data
data.pop(1,None)
print max(data, key=data.get)
#################################################################


#################################################################
#Analyse male-female proportion in dataset
# 34403
# {nan: 19902, 'male': 7405, 'female': 7096}
print len(members['gender'])
data = {}
for i in members['gender']:
    if i in data:
        data[i] = data[i] + 1
    else:
        data[i] = 1 
print data
print max(data, key=data.get)
###################################################################


###############################################################
#Analyse age range and how many times that age present
print len(members['bd'])
data = {}
for i in members['bd']:
    if i in data:
        data[i] = data[i] + 1
    else:
        data[i] = 1 
print data
#data.pop(0,None)
print max(data, key=data.get)
plt.bar(range(len(data)),data.values(),align='center')
plt.xticks(range(len(data)),data.keys())
plt.show()
#################################################################


###############################################################
#Analyse at particular age how many female or male are there.
print len(members['bd'])
data = []
for i,j in zip(members['bd'],members['gender']):
    if j == 'female':
        data.append(i)
#print np.unique(data)
dic ={}
for i in data:
    if i in dic:
        dic[i] = dic[i] + 1
    else:
        dic[i] = 1 
print dic
print max(dic, key=dic.get)
#################################################################


#Used this for plotting result    
# color=["#000000","#FF0000","#FFF000"]
# x=[]
# for i in range(0,len(members['bd'])):
#     x.append(i)

# plt.scatter(x, members['bd'])
# plt.show()
# plt.close()

