# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 03:52:02 2018

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  as LDA


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

#NEURON NETWORK
from sklearn import cross_validation as cv
from sklearn.neural_network import MLPClassifier as mlpc
import time

def Neural(X,y):
    starttime=time.time()
    x1_train, x1_test, y1_train, y1_test=cv.train_test_split( X,y,test_size=0.33, random_state=0)
    clf=mlpc(max_iter=300,random_state=1).fit(x1_train,y1_train)
    trnscore=clf.score(x1_train,y1_train)
    tstscore=clf.score(x1_test,y1_test)
    trntime=time.time()-starttime
    return (trnscore,tstscore,trntime)
#N=5


#----------->original nn

nnscores=Neural(x1_n,y1)

MAXITER=20
nn_trnscores=np.tile(nnscores[0],MAXITER)
nn_tstscores=np.tile(nnscores[1],MAXITER)
nn_time=np.tile(nnscores[2],MAXITER)

data1=np.array(x1_n)
#---------------------->APPLY CLUSTERING
#independent------------>
newkmdata,newemdata,newldadata=[],[],[]
km=KMeans(n_clusters=2,random_state=0).fit(data1)
kmdata=km.labels_
em=GM(n_components=2,random_state=0).fit(data1)
emdata=em.predict(data1)
lda=LDA(n_components=2).fit(data1,y1)
data1_lda=lda.transform(data1)

x1_nn=x1_n.tolist()
for i in range (len(x1_nn)):
    newkm=(x1_nn[i])
    kmdatai=int(kmdata[i])
    newkm.extend([kmdatai])
    newkmdata.append(newkm)
    
x1_nn=x1_n.tolist()
for i in range (len(x1_nn)):
    newem=(x1_nn[i])
    emdatai=int(emdata[i])
    newem.extend([kmdatai])
    newemdata.append(newem)
    
x1_nn=x1_n.tolist()
for i in range (len(x1_nn)):
    newlda=(data1_lda[i].tolist())
    emdatai=int(emdata[i])
    newlda.extend([kmdatai])
    newldadata.append(newlda)
    
#print (newldadata)
#print (newkmdata)
#print (newemdata)

kmscores=Neural(newkmdata,y1)
emscores=Neural(newemdata,y1)
ldascores=Neural(newldadata,y1)

x=np.arange(3)
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

pyplot.title('clustering as new features performance \n training time comparison')
pyplot.ylim(0,1.0)
pyplot.xticks(range(5))
pyplot.bar(1, kmscores[2],  width=width, label='KM training time')
pyplot.bar(2, emscores[2], width=width, label='EM training time')
pyplot.bar(3, nnscores[2], width=width, label='Original training time')
pyplot.bar(4, ldascores[2], width=width, label='LDA-EM training time')
pyplot.legend()
pyplot.show()


pyplot.title('clustering as new features performance \n test accuracy')
pyplot.ylim(0.8,1.0)
pyplot.xticks(range(5))
pyplot.bar(1, kmscores[1],  width=width, label='KM test score')
pyplot.bar(2, emscores[1], width=width, label='EM test score')
pyplot.bar(3, nnscores[1], width=width, label='Original NN test score')
pyplot.bar(4, ldascores[1], width=width, label='LDA-EM test score')
pyplot.legend()
pyplot.show()


#---------------------->APPLY DIMENSION DEDUCTION
#independent------------>
pca_scores,ica_scores,rp_scores,lda_scores=[[0,0,0]],[[0,0,0]],[[0,0,0]],[[0,0,0]]
for N in range(1,MAXITER):
    pca=PCA(n_components=N).fit(data1)
    data1_pca=pca.transform(data1)
    pca_scores.append(Neural(data1_pca,y1))
    ica=FastICA(n_components=N).fit(data1)
    data1_ica=ica.transform(data1)
    ica_scores.append(Neural(data1_ica,y1))
    rp=GRP(n_components=N).fit(data1)
    data1_rp=rp.transform(data1)
    rp_scores.append(Neural(data1_rp,y1))
    lda=LDA(n_components=N).fit(data1,y1)
    data1_lda=lda.transform(data1)
    lda_scores.append(Neural(data1_lda,y1))

pca_scores_t=np.transpose(pca_scores)
ica_scores_t=np.transpose(ica_scores)
rp_scores_t=np.transpose(rp_scores)
lda_scores_t=np.transpose(lda_scores)

pyplot.title("test accuracy on NN")
pyplot.plot(pca_scores_t[1],linewidth=1.5,label='pca scores')
pyplot.plot(ica_scores_t[1],linewidth=1.5,label='ica scores')
pyplot.plot(rp_scores_t[1],linewidth=1.5,label='rp scores')
pyplot.plot(lda_scores_t[1],linewidth=1.5,label='lda scores')
pyplot.plot(nn_tstscores,'k--',linewidth=1.0,label='original neural \n network performance')
pyplot.legend()
pyplot.ylim(0.5,1.0)
pyplot.xlabel('component amount')
pyplot.ylabel('scores')
pyplot.show()

pyplot.title('training time comparison')
pyplot.plot(pca_scores_t[2],linewidth=1.5,label='pca time')
pyplot.plot(ica_scores_t[2],linewidth=1.5,label='ica time')
pyplot.plot(rp_scores_t[2],linewidth=1.5,label='rp time')
pyplot.plot(lda_scores_t[2],linewidth=1.5,label='lda time')
pyplot.plot(nn_time,'k--',linewidth=1.0,label='original neural \n network training time')
pyplot.legend()
pyplot.xlabel('component amount')
pyplot.ylabel('time')
pyplot.show()







