# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:47:04 2018

@author: ChenZhengyang
"""

from sklearn import preprocessing
import numpy as np 
import pandas as pd 
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import FastICA
import scipy


dataset1= pd.read_csv("./DATASET/student/student-por.csv")
var_to_encode = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup',
       'paid','activities','nursery','higher','internet','romantic']
for col in var_to_encode:
    dataset1[col] = LabelEncoder().fit_transform(dataset1[col])
y0=list(dataset1['G3'])
# Binarize G3<=11: G3=0   G3>11: G3=1
dataset1[['G3']] = preprocessing.Binarizer(threshold=12).transform(dataset1[['G3']])
x1=dataset1[dataset1.columns.drop('G3')]
y1= list(dataset1['G3'])
scaler = StandardScaler()  
scaler.fit(x1)
x1_n=scaler.transform(x1)
#<-----------------------DATASET1

dataset2 = pd.read_csv("./DATASET/BANK/MT_Train.csv")
dataset2.drop('default',axis=1,inplace=True)
le = LabelEncoder()
var_to_encode = ['job','marital','education','day_of_week','month','housing','loan','poutcome']
for col in var_to_encode:
    dataset2[col] = le.fit_transform(dataset2[col])
dataset2["contact"]=preprocessing.LabelBinarizer().fit_transform(dataset2["contact"])
dataset2[["pdays"]] = preprocessing.Binarizer(threshold=998).transform(dataset2[["pdays"]])
dataset2["y"]=preprocessing.LabelBinarizer().fit_transform(dataset2["y"])
x2=dataset2[dataset2.columns.drop('y')]
y2= list(dataset2["y"])
scaler = StandardScaler()  
scaler.fit(x2)
x2_n=scaler.transform(x2)
#<---------------------DATASET2


#KURTOCTIC OF ICA
for name,data in [['student set',x1],['student set(normalized)',x1_n],['bank set',x2],['bank set(normalized)',x2_n]]:
    kurts=[]
    frmdata=np.array(data)
    data=frmdata.tolist()
    data=np.transpose(data)
    for comp in data:
        kurt=scipy.stats.kurtosis(comp)
        kurts.append(abs(kurt))
    
    pyplot.title('original kurtosis on '+name)
    pyplot.bar(range(len(kurts)),kurts)
    pyplot.plot(np.tile(3,len(kurts)),'r--',linewidth=1,label='kurtosis=3(gaussian)')
    pyplot.ylabel('value')
    pyplot.xlabel('feature id')
    pyplot.xticks(range(len(kurts)))
    pyplot.legend()
    pyplot.show()
    
for name,data in [['student set',x1],['student set(normalized)',x1_n],['bank set',x2],['bank set(normalized)',x2_n]]:
    ica=FastICA(n_components=len(data),tol=0.1,max_iter=2000).fit(data)
    comps=ica.components_
    kurts=[]
    for comp in comps:
        kurt=scipy.stats.kurtosis(comp)
        kurts.append(abs(kurt))
    
    pyplot.title('ICA kurtosis on '+name)
    pyplot.bar(range(len(kurts)),kurts)
    pyplot.plot(np.tile(3,len(kurts)),'r--',linewidth=1,label='kurtosis=3(gaussian)')
    pyplot.ylabel('value')
    pyplot.xlabel('feature id')
    pyplot.xticks(range(len(kurts)))
    pyplot.legend()
    pyplot.show()
    
    
    




















