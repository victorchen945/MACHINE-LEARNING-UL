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
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

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
km=KMeans(n_clusters=2, max_iter=300,random_state=2).fit(x1)
labels=km.labels_
cen=km.cluster_centers_
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort=PCA(n_components=5).fit(x1)
feature_plot=feature_sort.transform(x1)
cen_plot=feature_sort.transform(cen)
#score performance

#clustering with scaling
kmn=KMeans(n_clusters=2, max_iter=300,random_state=2).fit(x1_n)
labelsn=kmn.labels_
cenn=kmn.cluster_centers_
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort_n=PCA(n_components=5).fit(x1_n)
feature_plot_n=feature_sort_n.transform(x1_n)
cenn_plot=feature_sort_n.transform(cenn)


#flip matrix for plotting
feature_plot=list(zip(*feature_plot))
feature_plot_n=list(zip(*feature_plot_n))
cen_plot=list(zip(*cen_plot))
cenn_plot=list(zip(*cenn_plot))

#print (feature_plot[0])

clrs,clrsn=[],[]
for i in range(len(labels)):
    if labels[i]==0: clrs.append('b')
    else:clrs.append('r')
    if labelsn[i]==0:clrsn.append('b')
    else:clrsn.append('r')
pyplot.title('KMEANS clustering without normalization(STUDENT SET)')
pyplot.scatter(feature_plot[0],feature_plot[1],s=5,c=clrs,linewidths=.1)
pyplot.scatter(cen_plot[0],cen_plot[1],c='k',s=20,marker='x',linewidths=10,label='cluster center')
pyplot.legend()
pyplot.show()
pyplot.title('KMEANS clustering with normalization(STUDENT SET)')
pyplot.scatter(feature_plot_n[0],feature_plot_n[1],s=5,c=clrsn,linewidths=.1)
pyplot.scatter(cenn_plot[0],cenn_plot[1],c='k',s=20,marker='x',linewidths=10,label='cluster center')
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
km=KMeans(n_clusters=2, max_iter=300,random_state=0).fit(x2)
labels=km.labels_
cen=km.cluster_centers_
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort=PCA(n_components=5).fit(x2)
feature_plot=feature_sort.transform(x2)
cen_plot=feature_sort.transform(cen)
#score performance

#clustering with scaling
kmn=KMeans(n_clusters=2, max_iter=300,random_state=0).fit(x2_n)
labelsn=kmn.labels_
cenn=kmn.cluster_centers_
#visualization in 2d needs the dimension reduction to 2d thus we reduce it in PCA
feature_sort_n=PCA(n_components=5).fit(x2_n)
print (feature_sort.explained_variance_)
feature_plot_n=feature_sort_n.transform(x2_n)
cenn_plot=feature_sort_n.transform(cenn)


#flip matrix for plotting
feature_plot=list(zip(*feature_plot))
feature_plot_n=list(zip(*feature_plot_n))
cen_plot=list(zip(*cen_plot))
cenn_plot=list(zip(*cenn_plot))
#print (feature_plot[0])

# CLUSTER plotting
clrs,clrsn=[],[]
for i in range(len(labels)):
    if labels[i]==0: clrs.append('b')
    else:clrs.append('r')
    if labelsn[i]==0:clrsn.append('b')
    else:clrsn.append('r')
pyplot.title('KMEANS clustering without normalization(BANK SET)')
pyplot.scatter(feature_plot[0],feature_plot[1],s=5,c=clrs,linewidths=.1)
pyplot.scatter(cen_plot[0],cen_plot[1],c='k',s=20,marker='x',linewidths=10,label='cluster center')
pyplot.legend()
pyplot.show()
pyplot.title('KMEANS clustering with normalization(BANK SET)')
pyplot.scatter(feature_plot_n[0],feature_plot_n[1],s=5,c=clrsn,linewidths=.1)
pyplot.scatter(cenn_plot[0],cenn_plot[1],c='k',s=20,marker='x',linewidths=10,label='cluster center')
pyplot.legend()
pyplot.show()


#  performance PLOTTING
siscores1,siscores1n,siscores2,siscores2n=[0],[0],[0],[0]

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    km=KMeans(n_clusters=k, max_iter=300,random_state=0).fit(x2)
    labels=km.labels_
    ari_score=metrics.adjusted_rand_score(y2,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y2,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y2,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x2,labels)
    siscores2.append(siscore)
pyplot.title('scores - k amount(bank set)')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    kmn=KMeans(n_clusters=k, max_iter=300,random_state=0).fit(x2_n)
    labels=kmn.labels_
    ari_score=metrics.adjusted_rand_score(y2,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y2,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y2,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x2_n,labels)
    siscores2n.append(siscore)
pyplot.title('scores - k amount(bank set normalized)')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    km=KMeans(n_clusters=k, max_iter=300,random_state=0).fit(x1)
    labels=km.labels_
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x1,labels)
    siscores1.append(siscore)
pyplot.title('scores - k amount(student set / binary classification)')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    kmn=KMeans(n_clusters=k, max_iter=300,random_state=0).fit(x1_n)
    labels=kmn.labels_
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
    if k==1:siscore=0
    else:siscore=metrics.silhouette_score(x1_n,labels)
    siscores1n.append(siscore)
pyplot.title('scores - k amount(student set(normalized) / binary classification)')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    km=KMeans(n_clusters=k, max_iter=300,random_state=0).fit(x1)
    labels=km.labels_
    ari_score=metrics.adjusted_rand_score(y0,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y0,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y0,labels)
    vscores.append(v_score)
pyplot.title('scores - k amount(student set /multiple classification)')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

ariscores,amiscores,vscores=[0],[0],[0]
for k in range(1,50):
    kmn=KMeans(n_clusters=k, max_iter=300,random_state=0).fit(x1_n)
    labels=kmn.labels_
    ari_score=metrics.adjusted_rand_score(y0,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y0,labels)
    amiscores.append(ami_score)
    v_score=metrics.v_measure_score(y0,labels)
    vscores.append(v_score)
pyplot.title('scores - k amount(student set(normalized) /multiple classification)')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
pyplot.ylim(0,0.5)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()

pyplot.title('silhouette scores on clustering task-KMeans')
pyplot.xlabel('k num')
pyplot.ylabel('scores')
pyplot.plot(siscores2,linewidth=2,label='bank set')
pyplot.plot(siscores2n,linewidth=2,label='bank set(normalized)')
pyplot.plot(siscores1,linewidth=2,label='student set')
pyplot.plot(siscores1n,linewidth=2,label='student set(normalized)')
pyplot.legend()
pyplot.show()








