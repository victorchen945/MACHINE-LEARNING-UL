# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:47:04 2018

@author: ChenZhengyang
"""

from sklearn import preprocessing
import numpy as np 
import pandas as pd 
import scipy
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


dataset1= pd.read_csv("./DATASET/student/student-por.csv")
var_to_encode = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup',
       'paid','activities','nursery','higher','internet','romantic']
for col in var_to_encode:
    dataset1[col] = LabelEncoder().fit_transform(dataset1[col])
y0=list(dataset1['G3'])
# Binarize G3<=11: G3=0   G3>11: G3=1
dataset1[['G3']] = preprocessing.Binarizer(threshold=10).transform(dataset1[['G3']])
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

#EIGENVALUE OF PCA
def plot_dist(data,mean,std,minval,maxval,inter,num):
    x = np.arange(minval-2*inter, maxval+2*inter,inter)
    y = normfun(x, mean, std)
    pyplot.plot(x,y,linewidth = 1.5,label='eigenvalue '+str(num))
    #pyplot.hist(data, bins =17, color = 'r',alpha=0.5,rwidth= 0.9, normed=True)
    
def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


for name,data in [['student set',x1],['student set(normalized)',x1_n],['bank set',x2],['bank set(normalized)',x2_n]]:
    pca=PCA(n_components=None).fit(data)
    newdata=pca.transform(data)
    newdata_t=np.transpose(newdata)
    kurts=[]
    for i in range(5):
        k=newdata_t[i]
        kurt=scipy.stats.kurtosis(k)
        kurts.append(kurt)
        plot_dist(k,np.mean(k),np.std(k),np.min(k),np.max(k),0.1,i+1)
    pyplot.xlabel("value")
    pyplot.ylabel("proba")
    pyplot.title("top 5 eigenvalue distribution in "+name)
    pyplot.legend()
    pyplot.show()

    eigen=pca.explained_variance_
    eigen_r=pca.explained_variance_ratio_
    pyplot.title('eigenvalue ratio sorting '+name)
    pyplot.bar(range(len(eigen_r)),eigen_r)
    pyplot.ylabel('ratio')
    pyplot.xlabel('feature id')
    pyplot.xticks(range(len(eigen_r)))
    pyplot.show()

#distribution plot


#performance plot
for name, data,y in [['student set',x1,y1],['student set normalized',x1_n,y1],['bank set',x2,y2],['bank set normalized',x2_n,y2]]:
    data=np.array(data)
    ariscores,amiscores,vscores=[0],[0],[0]
    for k in range(1,len(data[0])):
        pca=PCA(n_components=k).fit(data)
        newdataset=pca.transform(data)
        km=KMeans(n_clusters=2,n_init=10,random_state=1).fit(newdataset)
        labels=km.labels_
        ari_score=metrics.adjusted_rand_score(y,labels)
        ariscores.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(y,labels)
        amiscores.append(ami_score)
        v_score=metrics.v_measure_score(y,labels)
        vscores.append(v_score)
    pyplot.title('scores - PCA components: '+name)
    pyplot.xlabel('PCA output features')
    pyplot.ylabel('scores')
    pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
    pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
    pyplot.plot(vscores,linewidth=2,label='v measure score')
    pyplot.yticks(np.arange(0, 0.5, 0.1))
    pyplot.legend()
    pyplot.show()
    
    ariscores,amiscores,vscores=[0],[0],[0]
    for k in range(1,len(data[0])):
        pca=PCA(n_components=k).fit(data)
        newdataset=pca.transform(data)
        gmn=GaussianMixture(n_components=2,max_iter=100, random_state=0).fit(data)
        labels=gmn.predict(data)
        ari_score=metrics.adjusted_rand_score(y,labels)
        ariscores.append(ari_score)
        ami_score=metrics.adjusted_mutual_info_score(y,labels)
        amiscores.append(ami_score)
        v_score=metrics.v_measure_score(y,labels)
        vscores.append(v_score)
    pyplot.title('scores - PCA components: '+name +'-EM')
    pyplot.xlabel('PCA output features')
    pyplot.ylabel('scores')
    pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
    pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
    pyplot.plot(vscores,linewidth=2,label='v measure score')
    pyplot.yticks(np.arange(0, 0.5, 0.1))
    pyplot.legend()
    pyplot.show()

"""

ariscores,amiscores,homoscores,compscores,vscores=[0],[0],[0],[0],[0]
for k in range(1,20):
    pca=PCA(n_components=20).fit(x1)
    newdataset=pca.transform(x1)
    km=KMeans(n_clusters=k,n_init=10,random_state=1).fit(newdataset)
    labels=km.labels_
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    homo_score=metrics.homogeneity_score(y1,labels)
    homoscores.append(homo_score)
    com_score=metrics.completeness_score(y1,labels)
    compscores.append(com_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
pyplot.title('scores for cluster amounts(not normalized)-STUDENT SET')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
#pyplot.xlim(2,20)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()
    


ariscores,amiscores,homoscores,compscores,vscores=[0],[0],[0],[0],[0]
for k in range(1,10):
    pca=PCA(n_components=k).fit(x1_n)
    newdataset=pca.transform(x1_n)
    km=KMeans(n_clusters=2,n_init=10,random_state=1).fit(newdataset)
    labels=km.labels_
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    homo_score=metrics.homogeneity_score(y1,labels)
    homoscores.append(homo_score)
    com_score=metrics.completeness_score(y1,labels)
    compscores.append(com_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
pyplot.title('scores for PCA features(normalized)-STUDENTSET')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
#pyplot.xlim(2,20)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()


ariscores,amiscores,homoscores,compscores,vscores=[0],[0],[0],[0],[0]
for k in range(1,10):
    pca=PCA(n_components=20).fit(x1_n)
    newdataset=pca.transform(x1_n)
    km=KMeans(n_clusters=k,n_init=10,random_state=1).fit(newdataset)
    labels=km.labels_
    ari_score=metrics.adjusted_rand_score(y1,labels)
    ariscores.append(ari_score)
    ami_score=metrics.adjusted_mutual_info_score(y1,labels)
    amiscores.append(ami_score)
    homo_score=metrics.homogeneity_score(y1,labels)
    homoscores.append(homo_score)
    com_score=metrics.completeness_score(y1,labels)
    compscores.append(com_score)
    v_score=metrics.v_measure_score(y1,labels)
    vscores.append(v_score)
pyplot.title('scores for cluster amounts(normalized)-STUDENT SET')
pyplot.plot(ariscores,linewidth=2,label='adjusted rand index score')
pyplot.plot(amiscores,linewidth=2,label='adjusted mutual info score')
pyplot.plot(vscores,linewidth=2,label='v measure score')
#pyplot.xlim(2,20)
pyplot.yticks(np.arange(0, 0.5, 0.1))
pyplot.legend()
pyplot.show()







"""










