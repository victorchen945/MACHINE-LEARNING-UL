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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  as LDA
from sklearn.decomposition import PCA
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

#LDA distribution

def plot_dist(data,mean,std,minval,maxval,inter,str,flag=0):
    x = np.arange(minval-2*inter, maxval+2*inter,inter)
    y = normfun(x, mean, std)
    if flag==0:pyplot.plot(x,y,linewidth = 1.5,label=str)
    elif flag==1: pyplot.hist(data, bins =17, color = 'r',alpha=0.5,rwidth= 0.9, normed=True)
    
def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

for name, data,y in [['student set',x1,y1],['student set normalized',x1_n,y1],['bank set',x2,y2],['bank set normalized',x2_n,y2]]:
    data=list(np.array(data))
    lda = LDA(n_components=2)
    X1=lda.fit(data,y).transform(data)
    k=np.transpose(X1)[0]
    kurt=scipy.stats.kurtosis(k)
    plot_dist(k,np.mean(k),np.std(k),np.min(k),np.max(k),0.1,'LDA distribution')
    pca = PCA(n_components=2)
    X2=pca.fit(data).transform(data)
    k2=np.transpose(X2)[0]
    kurt=scipy.stats.kurtosis(k2)
    plot_dist(k2,np.mean(k2),np.std(k2),np.min(k2),np.max(k2),0.1,'PCA distribution')
    pyplot.xlabel("value")
    pyplot.ylabel("proba")
    pyplot.title("Compare distribution in "+name)
    pyplot.legend()
    pyplot.show()
    
    plot_dist(k,np.mean(k),np.std(k),np.min(k),np.max(k),0.1,'', flag=1)
    pyplot.xlabel("value")
    pyplot.ylabel("proba")
    pyplot.title("LDA distribution in "+name)
    pyplot.legend()
    pyplot.show()
    plot_dist(k2,np.mean(k2),np.std(k2),np.min(k2),np.max(k2),0.1,'', flag=1)
    pyplot.xlabel("value")
    pyplot.ylabel("proba")
    pyplot.title("PCA distribution in "+name)
    pyplot.legend()
    pyplot.show()
    




"""
    eigen_r=lda.explained_variance_ratio_
    print(eigen_r)
    pyplot.title('eigenvalue ratio sorting '+name)
    pyplot.bar(range(len(eigen_r)),eigen_r)
    pyplot.ylabel('ratio')
    pyplot.xlabel('feature id')
    pyplot.xticks(range(len(eigen_r)))
    pyplot.show()
    """
    
        

