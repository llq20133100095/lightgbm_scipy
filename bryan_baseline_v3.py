# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np


#generate the "userFeature.csv"
def get_user_feature():
    if os.path.exists('./data/userFeature.csv'):
        user_feature=pd.read_csv('./data/userFeature.csv')
    else:
        userFeature_data = []
        with open('./data/userFeature.data', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv('./data/userFeature.csv', index=False)
        gc.collect()
    return user_feature

#Preprocessing
def get_data():
    if os.path.exists('./data/data.csv'):
        return pd.read_csv('./data/data.csv')
    else:
        ad_feature = pd.read_csv('./data/adFeature.csv')
        train=pd.read_csv('./data/train.csv')
        predict=pd.read_csv('./data/test1.csv')
        #chagne the label "-1" to "0"
        train.loc[train['label']==-1,'label']=0
        #initial
        predict['label']=-1
        user_feature=get_user_feature()
        #First and end connection of table
        data=pd.concat([train,predict])
        data=pd.merge(data,ad_feature,on='aid',how='left')
        data=pd.merge(data,user_feature,on='uid',how='left')
        #Fill NA/NaN values using the "-1"
        data=data.fillna('-1')
        del user_feature
        return data

def batch_predict(data,index):
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
           'adCategoryId', 'productId', 'productType']
#    one_hot_feature=['LBS','age','consumptionAbility','education','gender','house','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
#           'adCategoryId', 'productId', 'productType']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    for feature in one_hot_feature:
        try:
            #简单来说 LabelEncoder 是对不连续的数字或者文本进行编号
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    
    #get the data an label
    train=data[data.label!=-1]
    train_y=train.pop('label')
    test=data[data.label==-1]
    res=test[['aid','uid']]
    test=test.drop('label',axis=1)
    enc = OneHotEncoder()
    train_x=train[['creativeSize']]
    test_x=test[['creativeSize']]
    
    #concate the one-hot encode
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature+' finish')
    print('one-hot prepared !')
    

    #concate the tf
    cv=CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print(feature + ' finish')
    print('cv prepared !')
    
#    tv=TfidfVectorizer()
#    for feature in vector_feature:
#        tv.fit(data[feature])
#        train_a = tv.transform(train[feature])
#        test_a = tv.transform(test[feature])
#        train_x = sparse.hstack((train_x, train_a))
#        test_x = sparse.hstack((test_x, test_a))
#        print(feature + ' finish')
#    print('tf-idf prepared !')

    del data
    return LGB_predict(train_x, train_y, test_x, res,index)

def LGB_predict(train_x,train_y,test_x,res,index):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=2000)
    res['score'+str(index)] = clf.predict_proba(test_x)[:,1]
    res['score'+str(index)] = res['score'+str(index)].apply(lambda x: float('%.6f' % x))
    print(str(index)+' predict finish!')
    gc.collect()
    res=res.reset_index(drop=True)
    return res['score'+str(index)]

#数据分片处理，对每片分别训练预测，然后求平均
#data=get_data()
data=pd.read_csv('./data/data.csv')

#get the train data and test data
train=data[data['label']!=-1]
test=data[data['label']==-1]
del data
predict=pd.read_csv('./data/test1.csv')
cnt=3
#返回数字的上入整数
size = math.ceil(len(train) / cnt)
result=[]
for i in range(cnt):
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(train) else len(train)
    slice = train[int(start):int(end)]
    result.append(batch_predict(pd.concat([slice,test]),i))
    gc.collect()

result=pd.concat(result,axis=1)
result['score']=np.mean(result,axis=1)
result=result.reset_index(drop=True)
result=pd.concat([predict[['aid','uid']].reset_index(drop=True),result['score']],axis=1)
result[['aid','uid','score']].to_csv('./data/submission.csv', index=False)
os.system('zip ./data/baseline.zip ./data/submission.csv')