#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:30:03 2019
b bb 
@author: jane_hsieh
Goal: 
    1. compute rank k leverage scores 
    2. compute rank k projection distances

Source paper:
    1. Efficient anomaly detection via matrix sketching
        Read more: http://papers.nips.cc/paper/8030-efficient-anomaly-detection-via-matrix-sketching
    [2. Frequent Directions: Simple and Deterministic Matrix Sketching
        Read More: https://epubs.siam.org/doi/abs/10.1137/15M1009718?
        ]
Data:
K8.data--
    #{columns}=5409; #{rows}=16772
dataXpart_c--
    Note: Xpart of K8.data (#{columns}=5409; #{rows}=16772) with row of missing data dropped
    #{columns}=5408; #{rows}=16592 (i.e.1 column of Y part dropped; 180 rows dropped)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


data_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Datasets/p53 Mutants Data Set/p53_old_2010'
work_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection'
os.chdir(work_dir)
os.getcwd()

#================= Input data: origin (first time) ========================================================

"""
# First time input of data
data = pd.read_csv(data_dir+'/K8.data', header=None, index_col=False, 
                   names=['v'+ str(i) for i in range(5409)], na_values=['?'])  #'header =0': no column names

data.dtypes
'''
    from data.dtypes, we observe that all variables have data type of 'float64', 
    except for v5408 (object type) of which the data are strings ({active, inactive}).
    Indeed, v5408 is the Y part for prediction; Cause it's not used here, we then drop this column
'''

dataXpart = data.iloc[:,:5408]
dataXpart.info()
#Check column-wise missing values
dataXpart.isnull().sum()
'''
    Since there're at least 57 missing values(NaN) for each column(variable), 
    we then drop rows whichever has any missing value.
'''

dataXpart = dataXpart.dropna()  # we drop 180 rows with missing data (16772-16592 = 180)
dataXpart.to_csv(data_dir+'/K8_Xpart_dropna.data')

Mean = dataXpart.mean()
pd.DataFrame({'Mean':Mean},index = ['v'+ str(i) for i in range(5408)]).to_csv(data_dir+'/K8_Xpart_dropna_mean.data')

del data


#======================== Simple EDA: Inspect data ====================================================

#-------------Full svd for data matrix: dataXpart_c ------------------------------------------

dataXpart_c = dataXpart.subtract(Mean)
dataXpart_c.to_csv(data_dir+'/K8_Xpart_dropna_c.data')

U, s, Vt = np.linalg.svd(dataXpart_c)

## check orthogonality
temp = np.dot(U.T,U)
temp = np.dot(Vt,Vt.T)
sns.heatmap(temp)
plt.show()

pd.DataFrame(U).to_csv(data_dir+'/K8_Xpart_dropna_c_U.data', header=None, index=None)
pd.DataFrame(s).to_csv(data_dir+'/K8_Xpart_dropna_c_s.data', header=None, index=None)
pd.DataFrame(Vt).to_csv(data_dir+'/K8_Xpart_dropna_c_Vt.data', header=None, index=None)

#--------------------------------------------------------------------------------------------
#Investigate the number of right eigenvectors-- Vt.T (V) needed extracting
s_sq = s**2
temp = np.cumsum(s_sq);print(temp[0])
temp = temp/temp[-1];


##plot "Cumulated Variance explained"
fig = plt.figure()
plt.plot(temp*100)
plt.xlabel('k: # of eigenvectors (V)')
plt.ylabel('unit: %')
plt.title('Cumulated Variance explained')
#plt.show()

# Specify the axes
plt.axes([0.4,0.3,0.4,0.4]) #(all in units relative to the figure dimensions)
# Plot the sliced series in red using the current axes
plt.plot(range(40), temp[:40]*100) #,color='blue'
plt.title('turing point')
plt.xticks(list(range(0,50,10)))
for x in range(10,31,10):
    plt.axvline(x=x, color='red', linestyle='--')
plt.show()

fig.savefig(data_dir+'/K8_Xpart_dropna_c_CVE.svg')
    '''
    According to the plot of "Cumulated Variance explained", we choose k=10, 20,30
    which is also the numbers of k used in experiments in Resource paper 1.
    '''
"""

#------------------------Input data (needed): >= 2nd time ------------------------------------------------
dataXpart_c = pd.read_csv(data_dir+'/K8_Xpart_dropna_c.data', index_col=0)

#Vt, s first from:  U, s, Vt = np.linalg.svd(dataXpart_c)
Vt = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_Vt.data', header=None, index_col=False, 
                 names=['v'+ str(i) for i in range(5408)])
s = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_s.data', header=None, index_col=False, 
                 names=['s'])
#U = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_U.data', header=None, index_col=False, names=['u'+ str(i) for i in range(16592)])










#===============================================================================================================
#========== Calculate Rank-k leverage scores/projection distance for each of of data matrix:dataXpart_c ============================

#----------------------------------Define functions ------------------------------------------------------------
'''
k = 10  #20,30
i=0
a = dataXpart_c.iloc[i,:]
'''
#function:  Rank_k_Leverage_score(a,k,Vt,s) -> L_k
def Rank_k_leverage_score(a, k, Vt, s):
    '''
    Input:
        a: row vector or series of dim=d (ex,each row vector of data matrix A (dim=(n,d)), streaming row data)
        k: number of (highest) eigenvectors selected
        Vt: array; right eigenvectors(i.e. rows of Vt) from Singular Vector Decomposition 
        s.t. A=U*diag(s)*Vt; dim(Vt)=(d,d)
        s: series of sigular values with dim=min(n,d) (see above)
    Output:
        L_k: Rank-k Leverage scores
    '''
    V_k = np.array(Vt[:k,:]).T
    s_k = s[:k]
    #Rank-k leverage score of a
    L_k = sum( (np.dot(a,V_k)/s_k)**2)
    return L_k

#temp1 =Rank_k_leverage_score(a,k,Vt,s)

#function: Rank_k_projection_distance(a,k,Vt) -> T_k
def Rank_k_projection_distance(a, k, Vt):
    '''
    Input:
        a: row vector or series of dim=d (ex,each row vector of data matrix A (dim=(n,d)), streaming row data)
        k: threshold of (highest) eigenvectors selected; in T_k, it means the (k+1)-th ~ d-th eigenvectors used
        Vt: array; right eigenvectors from Singular Vector Decomposition s.t. A=U*diag(s)*Vt; dim(Vt)=(d,d)
        s: series of sigular values with dim=min(n,d) (see above)
    Output:
        T_k: Rank-k Leverage scores
    '''
    V_k = np.array(Vt[k:,:]).T
    #Rank-k leverage score of a
    T_k = sum( (np.dot(a,V_k))**2 )
    return T_k

#temp =Rank_k_projection_distance(a,k,Vt)
#------------------------------------------------------------------------------------------------------------------
AD_scores = {}
for k in range(10,31,10):
    print(k)
    name = 'L_'+str(k);print(name)
    AD_scores[name] = dataXpart_c.apply(Rank_k_leverage_score,axis=1,args=[k,Vt,s])
    name = 'T_'+str(k);print(name)
    AD_scores[name] = dataXpart_c.apply(Rank_k_projection_distance,axis=1,args=[k,Vt])

AD_scores = pd.DataFrame(AD_scores, index = dataXpart_c.index)
AD_scores.to_csv(data_dir+'/K8_Xpart_dropna_c_ADscores.data')







