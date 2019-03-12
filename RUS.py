# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:14:12 2019

@author: gaura
"""
#importing all the required libraries
import pandas as pd
import numpy as np
from AdvancedAnalytics import ReplaceImputeEncode
from AdvancedAnalytics import calculate
from AdvancedAnalytics import logreg
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import math

#importing the data as a pandas data frame
df=pd.read_excel("CreditData_RareEvent.xlsx")

#defining a function to encode categorical variables
def my_encoder(z):
    for i in z:
        a=df[i][df[i].notnull()].unique()
        for col_name in a:
            df[i+'_'+str(col_name)]= df[i].apply(lambda x: 1 if x==col_name else 0)
            
#Defining a function to scale the interval variables
def my_scaler(z):
    for i in z:
        df[i]= df[i].apply(lambda z: (z-np.mean(df[i]))/ np.std(df[i]))

#encoding interval and categorical attributes using the above defined functions.
categorical = ['checking','coapp','depends','employed','existcr','foreign','history','housing','installp','job','marital','other','property','resident','savings','telephon']
interval_Columns= ['age','duration']        
my_encoder(categorical)
my_scaler(interval_Columns)

df= df.drop(columns=categorical)
X = np.array(df[df.columns[df.columns!='good_bad']])
Y = df[['good_bad']]
Y['good_bad']=Y['good_bad'].map({'good':1,'bad':0})
Y= np.asarray(Y)

fp_cost = np.array(df['amount'])
fn_cost = np.array(0.15*df['amount'])
np.random.seed(12345)

max_seed= 2**30 - 1
rand_val = np.random.randint(1,high=max_seed ,size=10)

depth = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ratio = ['50:50','60:40','70:30','75:25','80:20','85:15']

rus_ratio = ({0:500,1:500},{0:500,1:750},{0:500,1:1167},{0:500,1:1500},{0:500,1:2000},{0:500,1:2834})
min_loss = 1e64
best_ratio= 0

#cross validation to find best RUS and best depth
for k in range(len(rus_ratio)):
    print("\nDecision Tree Model using" + ratio[k] + "RUS")
    min_loss_d = 1e64
    best_d = 0
    for j in range(len(depth)):
        d= depth[j]
        fn_loss = np.zeroes(len(rand_val))
        fp_loss = np.zeroes(len(rand_val))
