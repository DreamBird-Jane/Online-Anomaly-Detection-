#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:22:14 2019

@author: jane_hsieh

Data source: https://github.com/numenta/NAB/tree/master/data/artificialWithAnomaly
    NAB > data > artificialWithAnomaly > (4 data source- multivariate with shared timestamp):
        1. art_daily_flatmiddle.csv
        2. art_daily_jumpsdown.csv
        3. art_daily_jumpsup.csv
        4. art_daily_nojump.csv
Anomaly log: https://github.com/numenta/NAB/blob/master/labels/combined_labels.json 
    time of anomaly: 2014-04-11 00:00:00   
    
Purpose:
    practice function: OD_onlinePCA   and compare with OD_wpca
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math


data_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Datasets/p53 Mutants Data Set/p53_old_2010'
work_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection'
os.chdir(work_dir)
os.getcwd()

#================= Input data: origin (first time) ========================================================
# First time input of data

A = pd.read_csv(data_dir+'/K8_Xpart_dropna_c.data', index_col=0)  #'header =0': no column names


A.dtypes
'''
    from data.dtypes, we observe that all variables have data type of 'float64', 
'''
A.isnull().sum().sum()
'''
    Since there're at least 57 missing values(NaN) for each column(variable), 
    we then drop rows whichever has any missing value.
'''
#Mean=data.mean()
#data_c = data.subtract(Mean)
'''
    Since mean of data is not 0, we subtract data into data_c to mean-center data.
    This is for the later SVD
'''
n = A.shape[0]

#======================== Build ground truth scores of anomalies =================================

# Import ground truth sccores: Rank-k Leverage scores and Projection distance
GT_scores = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_ADscores.data', index_col=0)  #'header =0': no column names


# Decide thresholds (eta) for each {L,P}_{10,20,30} by histograms
## pairplots
sns.pairplot(GT_scores,kind = 'scatter', plot_kws={"s": 5}, diag_kws={"bins":50})
plt.show()
plt.savefig(data_dir+'/K8_Xpart_dropna_c_ADscores_pairplot.jpg')


## histograms
'''
GT_scores.iloc[:,:6].hist(bins=50,figsize=(50,20))
plt.savefig(data_dir+'/K8_Xpart_dropna_c_ADscores_hist.jpg')
plt.show()
'''
size = 5
bins = 50

fg=plt.figure(figsize=(50,20))
for i,name in zip(range(1,7),GT_scores.columns[:6]): 
    print(i,name)
    plt.subplot(3,2,i)
    GT_scores[name].hist(bins=bins)
    plt.xticks(size=size)
    plt.yticks(size=size)
    plt.title(name,size=size+4,loc='center')
plt.suptitle('Histograms of Rank-k Leverage scores(L_k) & Projection Distance(T_k)')
plt.show()
plt.savefig(data_dir+'/K8_Xpart_dropna_c_ADscores_hist.jpg')


'''
fg=plt.figure(figsize=(50,20))

plt.subplot(3,2,1)
GT_scores['L_10'].hist(bins=bins)
plt.ylim((0,2000))
#ax.axvline('1939-01-01', color='red', linestyle='--')
plt.title('L_10',size=size)

plt.subplot(3,2,2)
GT_scores['T_10'].hist(bins=bins)
plt.ylim((0,2000))
plt.title('T_10',size=size)

plt.subplot(3,2,3)
GT_scores['L_20'].hist(bins=bins)
plt.ylim((0,2000))
plt.title('L_20',size=size)

plt.subplot(3,2,4)
GT_scores['T_20'].hist(bins=bins)
plt.ylim((0,2000))
plt.title('T_20',size=size)

plt.subplot(3,2,5)
GT_scores['L_30'].hist(bins=bins)
plt.ylim((0,2000))
plt.title('L_30',size=size)

plt.subplot(3,2,6)
GT_scores['T_30'].hist(bins=bins)
plt.ylim((0,2000))
plt.title('T_30',size=size)

#fg.title('Rank-k Leverage scores(L_k) & Projection Distance(T_k)')
plt.show()
'''

## Investigate thresholds

cut = GT_scores['L_10']> 0.003
n0=sum(cut)
p0=n0/n
print('n0={}, p0={}'.format(n0,p0))
n0==p0*n

cut = GT_scores['T_10']> 50000
n0=sum(cut)
p0=n0/n
print('n0={}, p0={}'.format(n0,p0))
n0==p0*n

cut = GT_scores['L_20']> 0.007
n0=sum(cut)
p0=n0/n
print('n0={}, p0={}'.format(n0,p0))
n0==p0*n

cut = GT_scores['T_20']> 40000
n0=sum(cut)
p0=n0/n
print('n0={}, p0={}'.format(n0,p0))
n0==p0*n

cut = GT_scores['L_30']> 0.008
n0=sum(cut)
p0=n0/n
print('n0={}, p0={}'.format(n0,p0))
n0==p0*n

cut = GT_scores['T_30']> 40000
n0=sum(cut)
p0=n0/n
print('n0={}, p0={}'.format(n0,p0))
n0==p0*n
'''
    Observe the thresholds of each ground truth AD scores: {L,T}_{10,20,30}
    L10: > 0.003, n0=378, p0=0.022782063645130184
    T10: > 50000, n0=551, p0=0.03320877531340405
    L20: > 0.007, n0=227, p0=0.01368129218900675
    T20: > 40000, n0=464, p0=0.027965284474445518
    L30: > 0.008, n0=471, p0=0.028387174541947925
    T30: > 40000, n0=283, p0=0.017056412729026037
    
    Conclusion: for convenience, we set p0=0.025  (n0=415)
'''

#Build binary data for {"Anomaly":1, "Normal": 0}; thresholds: largest (p0*100)% of AD scores => anomalies
p0=0.025
n0 = math.ceil(n* p0)

Anomalies = pd.DataFrame(0,columns=[x+'_isan' for x in GT_scores.columns],index=GT_scores.index)

for name in GT_scores.columns:
    temp = GT_scores[name].sort_values(ascending=False).index[:n0] #return the index set of the largest scores with size n0    
    Anomalies.loc[temp,name+'_isan'] = 1

##'Integral_isan':Report anomalies once the total reported frequencies of anomalies for each instance is larger than 1
Anomalies['Integral_isan'] = (Anomalies.sum(axis=1) >1)*1
n1= Anomalies['Integral_isan'].sum()
print('Integrated number of anomalies is {} with proportion = {:.6f}'.format(n1,n1/n))
GT_scores = pd.concat([GT_scores,Anomalies],axis=1)
'''
    Integrated number of anomalies is 620 with proportion = 0.037367
'''

for name in GT_scores.columns[:6]:
    #Find threshold number according to n0 -- the n0-th largest anomaly score
    temp3 = GT_scores[name].sort_values(ascending=False).iloc[n0-1]
    print('The threshold of anomalies is: {} >= {:.6f}'.format(name,temp3))
'''
    Given p0=0.025:
    The threshold score for variable L_10 is 0.002596
    The threshold score for variable T_10 is 57807.427986
    The threshold score for variable L_20 is 0.004809
    The threshold score for variable T_20 is 41765.371856
    The threshold score for variable L_30 is 0.008970
    The threshold score for variable T_30 is 34391.259233
    
    Additionally, we also build an integrated ground truth scores from all 6 kinds, 
    based on the following rules:
        An instance is reported as anomaly (=1) if it’s recognized by 6 measures as anomalies 
        for more than 2 times (included); o.w., it’s reported as anomaly (0)
        ==> #{anomalies}=620 (eta=0.037367405978784955)

'''

## histograms with axvlines
size = 6
bins = 50

fg=plt.figure(figsize=(50,20))
for i,name in zip(range(1,7),GT_scores.columns[:6]): 
    print(i,name)
    plt.subplot(3,2,i)
    GT_scores[name].hist(bins=bins)
    plt.ylim((0,2000))
    plt.xticks(size=size)
    plt.yticks(size=size)
    temp3 = GT_scores[name].sort_values(ascending=False).iloc[n0-1]
    plt.axvline(temp3, color='red', linestyle='--')
    plt.title(name,size=size+4,loc='center')
plt.suptitle('Histograms of Rank-k Leverage scores(L_k) & Projection Distance(T_k) with Threshold of Anomalies as Top {}%'.format(p0*100))
plt.show()
plt.savefig(data_dir+'/K8_Xpart_dropna_c_ADscores_hist(2).jpg')

#======================================================================================================
#============================ Outlier Detection by osPCA_LS ===========================================
#import osPCA_LeastSquared
from osPCA_LeastSquared import OD_onlinePCA

## parameters for AD   -------*************************************************************

#ratio = 0.01
#beta = 1/(n*ratio)
beta_list = np.arange(5e-7,1e-4+5e-6,5e-6) 
#------------------------

##output of ranks/suspicious scores of all instances, as well as eigenvector u

AD_scores = {}

for beta in beta_list:
    print(beta)
    s_index, s_score, u = OD_onlinePCA(A,beta)
    #u = u/np.linalg.norm(u,2)
    #temp = s_score[s_index]   #order the sequence descendingly by anomaly score

    #Record suspecious scores
    name = 'osPCA_LS_'+str(round(beta*1e6,2))  #scale of beta: 1e-6
    AD_scores[name] = s_score.flatten()
    
AD_scores = pd.DataFrame(AD_scores, index = A.index)
AD_scores.to_csv(data_dir+'/K8_Xpart_dropna_c_ADscores_bscale_1e-6.data')

#===================================================================================================
#========================= Evaluation between ground truth and osPCA AD scores ======================
from sklearn.metrics import roc_auc_score, f1_score

## -------- parameters  -------*************************************************************
beta_list = [round(beta*1e6,2) for beta in np.arange(5e-7,1e-4+5e-6,5e-6)]  #scale of beta: 1e-6
eta_list = [round(eta,3) for eta in np.arange(0.005,0.101,0.005)]

colnames = GT_scores.columns[6:]
indnames = ['b_'+str(b) for b in beta_list] #scale of beta: 1e-6
indnames2 = ['eta_'+str(eta) for eta in eta_list]
#build multi-index
Multi_indnames = pd.MultiIndex.from_product([indnames, indnames2],names=['beta','eta'])


#-------------------------------- Compute evaluation scores: f1 score/ AUC with 7 ground truth scores for each (beta, eta)------------------------------
df_AUC = pd.DataFrame(columns=colnames, index=indnames)
df_f1score = pd.DataFrame(columns=colnames, index=Multi_indnames)


for colname in colnames:
    print(colname)
    
    for indname in indnames:
        temp = 'osPCA_LS_'+indname[2:]
        print(temp)
        df_AUC.loc[indname,colname] = roc_auc_score(GT_scores[colname],AD_scores[temp])
        
        for indname2 in indnames2:
            temp2 = float(indname2[4:])
            print(indname2) 
            
            #n_eta: the number of anomalies according to eta'
            n_eta = math.ceil(n* temp2)
            #Find threshold number according to n_eta -- the n_eta-th largest anomaly score
            temp3 = AD_scores[temp].sort_values(ascending=False).iloc[n_eta-1]
            #Transform AD_scores[temp] into binary data--{Anomaly:1, Normal:0}
            temp4 = (AD_scores[temp]>=temp3)*1
            df_f1score.loc[(indname,indname2),colname] = f1_score(GT_scores[colname],temp4,average='binary')
            '''
            TP=sum((GT_scores[colname]==1) & (temp4==1))
            FP=sum((GT_scores[colname]==0) & (temp4==1))
            FN=sum((GT_scores[colname]==1) & (temp4==0))
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            print('precision={},\t recall={}'.format(round(precision,3),round(recall,3)))
            '''

colnames=[col[:-5] for col in df_AUC.columns] #change colnames
df_AUC.columns = colnames


#Given beta value, find the max(f1score)
df_f1score_max=df_f1score.groupby('beta').max()
df_f1score_max=df_f1score_max.loc[indnames,:]  #reorder the index

df_f1score_max.columns= colnames

#Given beta value, find eta'=argmax(f1score)
df_f1score_argmax= df_f1score.groupby('beta').apply(lambda data: np.argmax(data.values,axis=0))
df_f1score_argmax= df_f1score_argmax.to_dict()
df_f1score_argmax = pd.DataFrame(df_f1score_argmax, index = colnames).T
##So far, it's found the implicit index of eta, now to find the explicit index of eta values
df_f1score_argmax = df_f1score_argmax.applymap(lambda x: eta_list[x])

#---------- Visualization of evaluation scores : f1 score/ AUC ------------------------------------------
max_AUC = df_AUC.max()
argmax_AUC = df_AUC.apply(lambda x: np.argmax(x.values)) 
argmax_AUC = argmax_AUC.map(lambda x: beta_list[x]) 
'''
    it's found that argmax of AUC are all at index_loc = 12 (i.e. beta=60.5*10^-6)
'''

#---AUC  ----------------------------------------------------------------------
plt.figure(figsize=(30,50))
for name in df_AUC.columns:
    #Determine linestyle based on L/T
    if name[0]=='L':
        linestyle='-'
        color = 'blue'
    elif name[0]=='T':
        linestyle=':'
        color = 'green'
    else:
        linestyle='--'
        color = 'red'
    
    #Determine marker based on k
    if name[-2:]== '10':
        marker = 'o'
    elif name[-2:]== '20':
        marker = 'v'
    elif name[-2:]== '30':
        marker = '^'
    else:
        marker=None

    plt.plot(beta_list, df_AUC[name],linestyle=linestyle, color = color, 
             marker=marker, linewidth=1, label=name)

plt.legend(loc='lower right', ncol=7, fontsize=10)  
plt.xticks(beta_list)
plt.yticks(np.arange(0.6,0.91,0.05))
plt.xlabel(r'$\beta$  ' + r'(scale: $10^{-6})$', size = 15)
plt.ylabel('AUC (Area Under Curve)', size = 15)
plt.title('Figure 1. Results of P53 Mutants: AUC', size = 20)
plt.axvline(60.5, color = 'k', linestyle=':')
plt.text(65,0.75,'Argmax(AUC|ground truth) \n = 0.0000605', size=10, color='k')
plt.show()

plt.savefig('Results of P53 Mutants- AUC given beta.png')

#---f1 score--------------------------------------------------
max_f1_score =  df_f1score_max.max()
argmax_f1_score = df_f1score_max.apply(lambda x: np.argmax(x.values)) 
argmax_f1_score = argmax_f1_score.map(lambda x: beta_list[x]) 


plt.figure(figsize=(30,50))
for name in df_f1score_max.columns:
    #Determine linestyle based on L/T
    if name[0]=='L':
        linestyle='-'
        color = 'blue'
    elif name[0]=='T':
        linestyle=':'
        color = 'green'
    else:
        linestyle='--'
        color = 'red'
    
    #Determine marker based on k
    if name[-2:]== '10':
        marker = 'o'
    elif name[-2:]== '20':
        marker = 'v'
    elif name[-2:]== '30':
        marker = '^'
    else:
        marker=None

    plt.plot(beta_list, df_f1score_max[name],linestyle=linestyle, color = color, 
             marker=marker, linewidth=1, label=name, zorder=-1)

plt.legend(loc='lower right', ncol=7, fontsize=10)  
plt.xticks(beta_list)
plt.yticks(np.arange(0.1,0.46,0.05))
plt.xlabel(r'$\beta$  ' + r'(scale: $10^{-6})$', size = 15)
plt.ylabel('F1 score', size = 15)
plt.title('Figure 2. Results of P53 Mutants: F1 score', size = 20)
plt.scatter(argmax_f1_score,max_f1_score, color='cyan', marker='*', s=50, zorder=1)
for i,j in zip(argmax_f1_score,max_f1_score):
    plt.annotate(str(round(j,4)),xy=(i+0.2,j+0.005),size=8, color='k',zorder=2)
plt.show()

plt.savefig('Results of P53 Mutants- F1 score given beta.png')



#========================= Scatter plots =================================================
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal', 
                          markerfacecolor='indigo', markersize=15), 
                   Line2D([0], [0], marker='o', color='w', label='Anomaly', 
                          markerfacecolor='yellow', markersize=15)]

# Create the figure
plt.figure(figsize=(40,50))
plt.scatter(GT_scores['T_10'],AD_scores['osPCA_LS_60.5'],s=5,
            c=GT_scores['T_10_isan']) 
plt.grid(True)

plt.xlabel('T_10 scores', size = 15)
plt.ylabel(r'osPCA scores ($\beta$=6.5*$10^{-5}$)', size = 15)

plt.legend(handles=legend_elements,loc='upper right', title="For T_10")
plt.title('Figure 3. Scatter plot of T_10 vs osPCA '+r'($\beta$=6.5*$10^{-5}$)', size = 20)

plt.show()
plt.savefig('Scatter plot of T_10 vs osPCA (beta_60.5).png')



#------------------------------------------------------------------------------------
# Create the figure
plt.figure(figsize=(40,50))
plt.scatter(GT_scores['L_30'],AD_scores['osPCA_LS_60.5'],s=5,
            c=GT_scores['L_30_isan']) 
plt.grid(True)

plt.xlabel('L_30 scores', size = 15)
plt.ylabel(r'osPCA scores ($\beta$=6.5*$10^{-5}$)', size = 15)

plt.legend(handles=legend_elements,loc='upper right', title="For L_30")
plt.title('Figure 4. Scatter plot of L_30 vs osPCA '+r'($\beta$=6.5*$10^{-5}$)', size = 20)

plt.show()
plt.savefig('Scatter plot of L_30 vs osPCA (beta_60.5).png')

