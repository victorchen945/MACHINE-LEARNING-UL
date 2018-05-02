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
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.mixture import GaussianMixture
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

#RANDOMIZED PROJECTION ITER

for name, data,y in [['student set',x1,y1],['student set normalized',x1_n,y1],['bank set',x2,y2],['bank set normalized',x2_n,y2]]:
    data=np.array(data)
    varians=[]
    for i in range(200):
        rp=GRP(n_components=8,random_state=None).fit(data)
        newdata=rp.transform(data)
        variance=np.var(newdata)
        varians.append(variance)
        data=newdata
    percentvars=(varians/varians[0])
    pyplot.plot(percentvars,linewidth=1.5,label=name)
pyplot.plot(np.tile(1,200),'k--',linewidth=1,label=('start variance'))
pyplot.title('Variance in RP self iteration \n (ratio to the first run)')
pyplot.xlabel('rp iterations')
pyplot.ylabel("variance ratio")
pyplot.legend()
pyplot.show()
    
    
        

#RANDOMIZED PROJECTION KM AND EM PERFORMANCE
"""
for name, data,y in [['student set',x1,y1],['student set normalized',x1_n,y1],['bank set',x2,y2],['bank set normalized',x2_n,y2]]:
    data=np.array(data)
    ariscores,amiscores,vscores=[0],[0],[0]
    for k in range(1,len(data[0])):
        rp=GRP(n_components=k,eps=0.1,random_state=16).fit(data)
        newdataset=rp.transform(data)
        km=KMeans(n_clusters=2,n_init=10,random_state=1).fit(newdataset)
        labels=km.labels_
        ari_score=metrics.adjusted_rand_score(y,labels)
        ariscores.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(y,labels)
        amiscores.append(ami_score)
        v_score=metrics.v_measure_score(y,labels)
        vscores.append(v_score)
    pyplot.title('scores - RP components: '+name)
    pyplot.xlabel('RP output features')
    pyplot.ylabel('scores')
    pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
    pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
    pyplot.plot(vscores,linewidth=2,label='v measure score')
    pyplot.yticks(np.arange(0, 0.5, 0.1))
    pyplot.legend()
    pyplot.show()
    
    ariscores,amiscores,vscores=[0],[0],[0]
    for k in range(1,len(data[0])):
        rp=GRP(n_components=k,eps=0.1,random_state=None).fit(data)
        newdataset=rp.transform(data)
        gmn=GaussianMixture(n_components=2,max_iter=100, random_state=0).fit(data)
        labels=gmn.predict(data)
        ari_score=metrics.adjusted_rand_score(y,labels)
        ariscores.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(y,labels)
        amiscores.append(ami_score)
        v_score=metrics.v_measure_score(y,labels)
        vscores.append(v_score)
    pyplot.title('scores - RP components: '+name +'-EM')
    pyplot.xlabel('RP output features')
    pyplot.ylabel('scores')
    pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
    pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
    pyplot.plot(vscores,linewidth=2,label='v measure score')
    pyplot.yticks(np.arange(0, 0.5, 0.1))
    pyplot.legend()
    pyplot.show()
    
    
    """




















