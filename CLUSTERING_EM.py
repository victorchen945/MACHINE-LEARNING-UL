# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:07:09 2018

@author: ChenZhengyang
"""

from sklearn import preprocessing
import numpy as np 
import pandas as pd 
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA



RAND=None

dataset1= pd.read_csv("./DATASET/student/student-por.csv")

var_to_encode = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup',
       'paid','activities','nursery','higher','internet','romantic']
for col in var_to_encode:
    dataset1[col] = LabelEncoder().fit_transform(dataset1[col])

# Binarize G3<=11: G3=0   G3>11: G3=1
y0=list(dataset1['G3'])
dataset1[['G3']] = preprocessing.Binarizer(threshold=11).transform(dataset1[['G3']])

x1=dataset1[dataset1.columns.drop('G3')]
y1= list(dataset1['G3'])
scaler = StandardScaler()  
scaler.fit(x1)
x1_n=scaler.transform(x1)


#clustering without scaling
gmm=GaussianMixture(n_components=2, max_iter=100,random_state=RAND).fit(x1)
labels=gmm.predict(x1)
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort=PCA(n_components=5).fit(x1)
feature_plot=feature_sort.transform(x1)
#score performance

#clustering with scaling
gmn=GaussianMixture(n_components=2, max_iter=100,random_state=RAND).fit(x1_n)
labelsn=gmn.predict(x1_n)
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort_n=PCA(n_components=5).fit(x1_n)
feature_plot_n=feature_sort_n.transform(x1_n)


#flip matrix for plotting
feature_plot=list(zip(*feature_plot))
feature_plot_n=list(zip(*feature_plot_n))

clrs,clrsn=[],[]
for i in range(len(labels)):
    if labels[i]==0: clrs.append('b')
    else:clrs.append('r')
    if labelsn[i]==0:clrsn.append('b')
    else:clrsn.append('r')
pyplot.title('EM clustering without normalization(STUDENT SET)')
pyplot.scatter(feature_plot[0],feature_plot[1],s=5,c=clrs,linewidths=.1)
pyplot.legend()
pyplot.show()
pyplot.title('EM clustering with normalization(STUDENT SET)')
pyplot.scatter(feature_plot_n[0],feature_plot_n[1],s=5,c=clrsn,linewidths=.1)
pyplot.legend()
pyplot.show()

#preprocessing dataset2

# merge the train and test set because it is unsupervised learning

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
y2= list(dataset2['y'])
scaler = StandardScaler()  
scaler.fit(x2)
x2_n=scaler.transform(x2)

#clustering without scaling
gmm=GaussianMixture(n_components=2,max_iter=100, random_state=RAND).fit(x2)
labels=gmm.predict(x2)
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort=PCA(n_components=5).fit(x2)
feature_plot=feature_sort.transform(x2)
#score performance

#clustering with scaling
gmn=GaussianMixture(n_components=2,max_iter=100, random_state=RAND).fit(x2_n)
labelsn=gmn.predict(x2_n)
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort_n=PCA(n_components=5).fit(x2_n)
feature_plot_n=feature_sort_n.transform(x2_n)


#flip matrix for plotting
feature_plot=list(zip(*feature_plot))
feature_plot_n=list(zip(*feature_plot_n))
#print (feature_plot[0])

# CLUSTER plotting
clrs,clrsn=[],[]
for i in range(len(labels)):
    if labels[i]==0: clrs.append('b')
    else:clrs.append('r')
    if labelsn[i]==0:clrsn.append('b')
    else:clrsn.append('r')
pyplot.title('EM clustering without normalization(BANK SET)')
pyplot.scatter(feature_plot[0],feature_plot[1],s=5,c=clrs,linewidths=.1)
pyplot.legend()
pyplot.show()
pyplot.title('EM clustering with normalization(BANK SET)')
pyplot.scatter(feature_plot_n[0],feature_plot_n[1],s=5,c=clrsn,linewidths=.1)
pyplot.legend()
pyplot.show()

"""
#  performance PLOTTING
siscores1,siscores1n,siscores2,siscores2n=[0],[0],[0],[0]

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    gmm=GaussianMixture(n_components=k, random_state=0).fit(x2)
    labels=gmm.predict(x2)
    ari_score=metrics.adjusted_rand_score(y2,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y2,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y2,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x2,labels)
    siscores2.append(siscore)
pyplot.title('scores - components (bank set)-EM')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('components num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    gmm=GaussianMixture(n_components=k, random_state=0).fit(x2_n)
    labels=gmm.predict(x2_n)
    ari_score=metrics.adjusted_rand_score(y2,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y2,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y2,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x2_n,labels)
    siscores2n.append(siscore)
pyplot.title('scores - components (bank set normalized)-EM')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('components num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    gmm=GaussianMixture(n_components=k, random_state=0).fit(x1)
    labels=gmm.predict(x1)
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x1,labels)
    siscores1.append(siscore)
pyplot.title('scores - components amount(student set / binary classification)-EM')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('components num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    gmm=GaussianMixture(n_components=k, random_state=0).fit(x1_n)
    labels=gmm.predict(x1_n)
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x1_n,labels)
    siscores1n.append(siscore)
pyplot.title('scores - components amount(student set(normalized) / binary classification)-EM')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('components num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    gmm=GaussianMixture(n_components=k, random_state=0).fit(x1)
    labels=gmm.predict(x1)
    ari_score=metrics.adjusted_rand_score(y0,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y0,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y0,labels)
    vscores.append(v_score)
pyplot.title('scores - components amount(student set /multiple classification)-EM')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('components num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    gmm=GaussianMixture(n_components=k, random_state=0).fit(x1_n)
    labels=gmm.predict(x1_n)
    ari_score=metrics.adjusted_rand_score(y0,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y0,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y0,labels)
    vscores.append(v_score)
pyplot.title('scores - components amount(student set(normalized) /multiple classification)-EM')
pyplot.xlabel('components num')
pyplot.ylabel('scores')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()


pyplot.title('silhouette scores on clustering task-EM')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.plot(siscores2,linewidth=2,label='bank set')
pyplot.plot(siscores2n,linewidth=2,label='bank set(normalized)')
pyplot.plot(siscores1,linewidth=2,label='student set')
pyplot.plot(siscores1n,linewidth=2,label='student set(normalized)')
pyplot.legend()
pyplot.show()
"""