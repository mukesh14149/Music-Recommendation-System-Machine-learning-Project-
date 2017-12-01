import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import xgboost as xgb

from datetime import datetime
from scipy import stats

data_path = ''
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

#Extract year from isrc format
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language', 'composer', 'lyricist']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

print "start registeration"
#Calculate days from 2000 year as a registeration days
regis_days = []
for line in (members['registration_init_time']):
    x1 = datetime.strptime(str(line), '%Y%m%d')
    d0 = datetime(2000, 1, 1)
    delta1 = x1 - d0
    regis_days.append(delta1.days)

members['registeration_num_of_days'] = pd.Series(regis_days)

print "start expiration"
#Calculate days from 2000 year as a expiration days
regis_days = []
for line in (members['expiration_date']):
    x1 = datetime.strptime(str(line), '%Y%m%d')
    d0 = datetime(2000, 1, 1)
    delta1 = x1 - d0
    regis_days.append(delta1.days)

members['expiration_num_of_days'] = pd.Series(regis_days)
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))

members = members.drop(['registration_init_time'], axis=1)

print "start merge"
members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')

count = 0
su = 0
for i, val in enumerate(train['bd']):
    if (val> 10 and val < 100):
        su = su + val
        count = count + 1
         



#Assign mean age to those user whose
#age is less then 10 and greater then 100
meanVal = su/count
arr = train['bd']
arr = np.array(arr, dtype=pd.Series)


for i, val in enumerate(arr):
    if (val< 10 or val > 100):
        arr[i]=meanVal


train['bd'] = pd.Series(arr)
print "mean done"

#Replace all nan value to -1
train.replace(r'\s+', np.nan, regex=True)
test.replace(r'\s+', np.nan, regex=True)
train = train.fillna(-1)
test = test.fillna(-1)

#One hot encoding of genre_ids feature of dataset.
# res = []
# res1 = []
# tt = train['genre_ids'].unique()
# tt1 = test['genre_ids'].unique()

# for i in tt:
#     if type(i) == int:
#         res.append('-1')    
#     elif '|' in i:
#         li = i.split('|')
#         for jj in li:
#             res.append(jj)        
#     else:
#          res.append(i)

# for i in tt1:
#     if type(i) == int:
#         res1.append('-1')    
#     elif '|' in i:
#         li = i.split('|')
#         for jj in li:
#             res1.append(jj)        
#     else:
#          res1.append(i)

# differ = []
# for i in res1:
#     if i not in res:
#         differ.append(i)

# for j,i in enumerate(set(res)):
#     print j
#     train[i] = pd.Series(np.zeros(len(train['genre_ids']),int))
#     test[i] = pd.Series(np.zeros(len(train['genre_ids']),int))


# for j,i in enumerate(set(differ)):
#     print j
#     train[i] = pd.Series(np.zeros(len(train['genre_ids']),int))
#     test[i] = pd.Series(np.zeros(len(train['genre_ids']),int))


# for j,i in enumerate(train['genre_ids']):
#     if type(i) == int:
#         train['-1'][j] = 1
#     elif '|' in i:
#         li = i.split('|')
#         for jj in li:
#             train[jj][j] = 1        
#     else:
#          train[i][j] = 1

# print train.shape
# # Preprocess dataset
cols = list(train.columns)
#cols.remove('target')

print "label started"
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

print "z score started"
tempzscore = np.abs(stats.zscore(train))
zs = []
for i in train:
    zs.append(np.train[i])

# removing outlier base on zscore
train = train[(np.abs(stats.zscore(train))<5).all(axis=1)]

#removing outliers based on Interquartile Range
scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(x)
train = train[(np.abs(robust_scaled_df)<3).all(axis=1)]

#Standardization of the dataset
train = preprocessing.scale(train)

# print "done"
train.to_csv('trainDataPreData.csv', index=None, mode='a')
test.to_csv('testDataPreData.csv', index=None, mode='a')
