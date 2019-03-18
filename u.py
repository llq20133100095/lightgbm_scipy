#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:51:04 2018

@author: llq
"""
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
texts=["dog cat fish","dog cat cat","fish bird", 'bird']

#TF
cv = CountVectorizer()
#cv_fit=cv.fit_transform(texts)
#print(cv.get_feature_names())
#print(cv_fit.toarray())
cv.fit(texts)
a=cv.transform(texts).toarray()

#tf-idf
tfidf=TfidfVectorizer()
tfidf.fit(texts)
b=tfidf.transform(texts).toarray()


txt=pd.read_csv('./data/train.csv')
num=0
num1=0
a=np.array(txt["label"])
for i in range(len(a)):
    if(a[i]==1):
        num+=1
    if(a[i]==-1):
        num1+=1
'''
#csr = sparse.coo_matrix([[1, 5, -3], [4, 0, 1], [1, 3, 0]])
#csr=csr.tocsr()
#csr[0,0]=5


def llq_smote(more_data,less_data, tag_index=None, max_amount=0, std_rate=5, kneighbor=5):

    #connect two sparse matrix
    data=np.vstack((more_data,less_data))
    
    case_state=pd.Series([more_data.shape[0],less_data.shape[0]])
    case_rate = max(case_state) / min(case_state)
    location = []
    if case_rate < 5:
        print('do not need smote')
        data=data.tocsr()
        #train_y
        train_y=data[0]
        #train_x
        data=data[:,1:]

        return data,train_y
        
    else:
        neighbors = NearestNeighbors(n_neighbors=kneighbor).fit(less_data)
        for i in range(less_data.shape[0]):
            location_set = neighbors.kneighbors([less_data.getrow(i).toarray()[0]], return_distance=False)[0]
            #store the 5 neighobors in each less data.
            location.append(location_set)
            print('it processes %s in %s less_data' % (i+1,less_data.shape[0]))
            
        if max_amount > 0:
            amount = max_amount
        else:
            amount = int(max(case_state) / std_rate)
                     
        times=0
        update_case = []
        while times<amount:
            #shuffle:choose a neighbor data
            ran=np.random.randint(1,kneighbor)
            np.random.shuffle(location)
            less_data_index=location[0][ran]
            center_index=location[0][0]
            
            #create new data. And label set to 1.
            gap=np.random.random()
            dif=less_data.getrow(center_index)-less_data.getrow(less_data_index)
            new_case=less_data.getrow(center_index)+gap*abs(dif)
            
            #transform coo to csr matrix
            new_case=new_case.tocsr()
            #set label to 1
            new_case[0,0]=1
            if times==0:
                update_case=new_case
            elif times!=0:
                update_case=sparse.vstack((update_case,new_case))

            print('it cretes %s new data，completeing %.2f' % (times+1, times * 100 / amount))
            times = times + 1              
        
        #concate
        data=sparse.vstack((data.tocsr(),update_case))
        
        #train_y
        train_y=data[0]
        #train_x
        data=data[:,1:]

        return data,train_y
        
data,train_y=llq_smote(train_more_data,train_less_data,tag_index=0)


