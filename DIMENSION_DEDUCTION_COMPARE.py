# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 02:48:24 2018

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
from sklearn.mixture import GaussianMixture as GM
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

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

#clustering comparison

    
for name, data,y in [['student set',x1,y1],['student set normalized',x1_n,y1],['bank set',x2,y2],['bank set normalized',x2_n,y2]]:
    data=np.array(data)
    defKM=KMeans(n_clusters=2,random_state=0).fit(data)
    KM_labels=defKM.labels_
    defEM=GM(n_components=2,random_state=0).fit(data)
    EM_labels=defEM.predict(data)
    
        #randomized projection
    ariscoresk_rp,amiscoresk_rp,ariscorese_rp,amiscorese_rp=[0],[0],[0],[0]
    for k in range(1,len(data[0])):
        rp=GRP(n_components=k,eps=0.1,random_state=16).fit(data)
        newdataset=rp.transform(data)
        km=KMeans(n_clusters=2,n_init=10,random_state=1).fit(newdataset)
        labels=km.labels_
        ari_score=metrics.adjusted_rand_score(KM_labels,labels)
        ariscoresk_rp.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(KM_labels,labels)
        amiscoresk_rp.append(ami_score)
        em=GM(n_components=2,random_state=0).fit(newdataset)
        labels=em.predict(newdataset)
        ari_score=metrics.adjusted_rand_score(EM_labels,labels)
        ariscorese_rp.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(EM_labels,labels)
        amiscorese_rp.append(ami_score)
        
        #PCA
    ariscoresk_pca,amiscoresk_pca,ariscorese_pca,amiscorese_pca=[0],[0],[0],[0]
    for k in range(1,len(data[0])):
        pca=PCA(n_components=k).fit(data)
        newdataset=pca.transform(data)
        km=KMeans(n_clusters=2,n_init=10,random_state=1).fit(newdataset)
        labels=km.labels_
        ari_score=metrics.adjusted_rand_score(KM_labels,labels)
        ariscoresk_pca.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(KM_labels,labels)
        amiscoresk_pca.append(ami_score)
        em=GM(n_components=2,random_state=0).fit(newdataset)
        labels=em.predict(newdataset)
        ari_score=metrics.adjusted_rand_score(EM_labels,labels)
        ariscorese_pca.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(EM_labels,labels)
        amiscorese_pca.append(ami_score)
    
    #ICA
    ariscoresk_ica,amiscoresk_ica,ariscorese_ica,amiscorese_ica=[0],[0],[0],[0]
    for k in range(1,len(data[0])):
        ica=FastICA(n_components=k,tol=0.01,max_iter=3000).fit(data)
        newdataset=ica.transform(data)
        km=KMeans(n_clusters=2,n_init=10,random_state=1).fit(newdataset)
        labels=km.labels_
        ari_score=metrics.adjusted_rand_score(KM_labels,labels)
        ariscoresk_ica.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(KM_labels,labels)
        amiscoresk_ica.append(ami_score)
        em=GM(n_components=2,random_state=0).fit(newdataset)
        labels=em.predict(newdataset)
        ari_score=metrics.adjusted_rand_score(EM_labels,labels)
        ariscorese_ica.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(EM_labels,labels)
        amiscorese_ica.append(ami_score)
    
    
    #plotting---------------------->
    pyplot.title("ARI score performance overview \n ("+name+')')
    pyplot.plot(ariscoresk_rp,linewidth=1.5,label='RP-KMeans')
    pyplot.plot(ariscorese_rp,linewidth=1.5,label='RP-EM')
    pyplot.plot(ariscoresk_pca,linewidth=1.5,label='PCA-KMeans')
    pyplot.plot(ariscorese_pca,linewidth=1.5,label='PCA-EM')
    pyplot.plot(ariscoresk_ica,linewidth=1.5,label='ICA-KMeans')
    pyplot.plot(ariscorese_ica,linewidth=1.5,label='ICA-EM')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()
        
    pyplot.title("AMI score performance overview \n ("+name+')')
    pyplot.plot(amiscoresk_rp,linewidth=1.5,label='RP-KMeans')
    pyplot.plot(amiscorese_rp,linewidth=1.5,label='RP-EM')
    pyplot.plot(amiscoresk_pca,linewidth=1.5,label='PCA-KMeans')
    pyplot.plot(amiscorese_pca,linewidth=1.5,label='PCA-EM')
    pyplot.plot(amiscoresk_ica,linewidth=1.5,label='ICA-KMeans')
    pyplot.plot(amiscorese_ica,linewidth=1.5,label='ICA-EM')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()
    
    pyplot.title("RP score performance \n ("+name+')')
    pyplot.plot(ariscoresk_rp,linewidth=1.5,label='ARI-KMeans')
    pyplot.plot(ariscorese_rp,linewidth=1.5,label='ARI-EM')
    pyplot.plot(amiscoresk_rp,linewidth=1.5,label='AMI-KMeans')
    pyplot.plot(amiscorese_rp,linewidth=1.5,label='AMI-EM')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()

    pyplot.title("PCA score performance \n ("+name+')')
    pyplot.plot(ariscoresk_pca,linewidth=1.5,label='ARI-KMeans')
    pyplot.plot(ariscorese_pca,linewidth=1.5,label='ARI-EM')
    pyplot.plot(amiscoresk_pca,linewidth=1.5,label='AMI-KMeans')
    pyplot.plot(amiscorese_pca,linewidth=1.5,label='AMI-EM')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()

    
    pyplot.title("ICA score performance \n ("+name+')')
    pyplot.plot(ariscoresk_ica,linewidth=1.5,label='ARI-KMeans')
    pyplot.plot(ariscorese_ica,linewidth=1.5,label='ARI-EM')
    pyplot.plot(amiscoresk_ica,linewidth=1.5,label='AMI-KMeans')
    pyplot.plot(amiscorese_ica,linewidth=1.5,label='AMI-EM')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()
    
    pyplot.title("KMeans score performance \n ("+name+')')
    pyplot.plot(ariscoresk_rp,linewidth=1.5,label='RP-ARI')
    pyplot.plot(amiscoresk_rp,linewidth=1.5,label='RP-AMI')
    pyplot.plot(ariscoresk_pca,linewidth=1.5,label='PCA-ARI')
    pyplot.plot(amiscoresk_pca,linewidth=1.5,label='PCA-AMI')
    pyplot.plot(ariscoresk_ica,linewidth=1.5,label='ICA-ARI')
    pyplot.plot(amiscoresk_ica,linewidth=1.5,label='ICA-AMI')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()

    pyplot.title("EM score performance \n ("+name+')')
    pyplot.plot(ariscorese_rp,linewidth=1.5,label='RP-ARI')
    pyplot.plot(amiscorese_rp,linewidth=1.5,label='RP-AMI')
    pyplot.plot(ariscorese_pca,linewidth=1.5,label='PCA-ARI')
    pyplot.plot(amiscorese_pca,linewidth=1.5,label='PCA-AMI')
    pyplot.plot(ariscorese_ica,linewidth=1.5,label='ICA-ARI')
    pyplot.plot(amiscorese_ica,linewidth=1.5,label='ICA-AMI')
    pyplot.xlabel('dimension components')
    pyplot.ylabel('scores')
    pyplot.legend()
    pyplot.show()

















