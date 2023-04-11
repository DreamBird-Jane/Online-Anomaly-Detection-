#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:46:47 2019

@author: jane_hsieh
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


#=================================== Define functions ===========================================================



#---------------------------------- Derive outlier scores after PCA transformation from X_std --> Y_pac  ------------------------------------------------------------

#x= X_std[0,:]
#k =5
#function:  Rank_k_Leverage_score(x, k, V, s2) -> L_k
def Rank_k_leverage_score(x, k, V, s2):
    '''
    Calculage Rank_k_leverage_score after PCA transformation from X_std --> Y_pac    
    Y_pca = pca.fit_transform(X_std)
    
    Input:
        x: row vector or series of dim=d (ex,each row vector of data matrix X (dim=(n,d)), streaming row data)
        k: number of (highest) eigenvectors selected; k <= d
        V: array; eigenvectors / PCs (column vectors) from PCA s.t. V = pca.components_.T; dim(V)=(p,k), k <= d
        s2: series of eigenalues with dim=min(n,p) (see above)
    Output:
        L_k: Rank-k Leverage scores
    '''
    V_k = np.array(V[:,:k])
    s2_k = s2[:k]
    #Rank-k leverage score of a
    L_k = sum( (np.dot(x,V_k)**2)/s2_k ) 
    return L_k


#y= Y_pca[0,:]
#function:  Rank_k_Leverage_score(y, k, s2) -> L_k
def Rank_k_leverage_score_y(y, k, s2):
    '''
    Calculage Rank_k_leverage_score after PCA transformation from X_std --> Y_pac    
    Y_pca = pca.fit_transform(X_std)
    
    Input:
        y: PCA transformation from x; 
           row vector or series of dim=k<=p (ex,each row vector of data matrix Y_pca (dim=(n,k))); y = np.dot(x,V)
        k: number of (highest) eigenvectors selected; k <= d
        s2: series of eigenalues with dim=min(n,p) (see above)
    Output:
        L_k: Rank-k Leverage scores
    '''
    y_k = np.array(y[:k])
    s2_k = s2[:k]
    #Rank-k leverage score of x
    L_k = sum( y_k**2/s2_k) 
    return L_k


#x= X_std[0,:]
#k = 500
#function: Rank_k_projection_distance(a,k,Vt) -> T_k
def Rank_k_projection_distance(x, k, V):
    '''
    Calculage Rank_k_projection_distance after PCA transformation from X_std --> Y_pac    
    Y_pca = pca.fit_transform(X_std)
    Input:
        x: row vector or series of dim=d (ex,each row vector of data matrix X (dim=(n,d)), streaming row data)
        k: threshold of (highest) eigenvectors selected; in T_k, it means the (k+1)-th ~ d-th eigenvectors used
        V: array; eigenvectors / PCs (column vectors) from PCA s.t. V = pca.components_.T; dim(V)=(p,k), k <= d
    Output:
        T_k: Rank-k Leverage scores
    '''
    V_k = np.array(V[:,k:])
    #Rank-k projection score of a
    T_k = sum( (np.dot(x,V_k))**2 )          
    return T_k

#y= Y_pca[0,:]
def Rank_k_projection_distance_y(y, k):
    '''
    Calculage Rank_k_projection_distance after PCA transformation from X_std --> Y_pac    
    Y_pca = pca.fit_transform(X_std)    
    Input:
        y: PCA transformation from x; 
           row vector or series of dim=k<=p (ex,each row vector of data matrix Y_pca (dim=(n,k))); y = np.dot(x,V)
        k: threshold of (highest) eigenvectors selected; in T_k, it means the (k+1)-th ~ d-th eigenvectors used
    Output:
        T_k: Rank-k Leverage scores
    '''
    y_k = np.array(y[k:])
    #Rank-k projection score of x
    T_k = sum( y_k**2 )          
    return T_k











#---------------------------------- Derive directly from raw data (A) &  Singular Vector Decomposition (SVD) ------------------------------------------------------------
#function:  Rank_k_Leverage_score(a,k,Vt,s) -> L_k
def Rank_k_leverage_score_SVD(a, k, Vt, s):
    '''
    Input:
        a: row vector or series of dim=d (ex,each row vector of data matrix A (dim=(n,d)), streaming row data)
        k: number of (highest) eigenvectors selected
        Vt: array; right eigenvectors (row vectors) from Singular Vector Decomposition s.t. A=U*diag(s)*Vt; dim(Vt)=(d,d)
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
def Rank_k_projection_distance_SVD(a, k, Vt):
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
    #Rank-k projection score of a
    T_k = sum( (np.dot(a,V_k))**2 )
    return T_k


"""
# demo
data = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_5766.data', index_col=0)  #'header =0': no column names
U, s, Vt = np.linalg.svd(data)

import PCAbased_AD_Algorithms as PCA_AD #<-- this file!


start2 = time.time() 

AD_scores = {}

for k in [7]:
    print(k)
    name = 'L_'+str(k);print(name)
    AD_scores[name] = data.apply(PCA_AD.Rank_k_leverage_score,axis=1,args=[k,Vt,s])
    name = 'T_'+str(k);print(name)
    AD_scores[name] = data.apply(PCA_AD.Rank_k_projection_distance,axis=1,args=[k,Vt])

end2 = time.time()
print(end2 - start2,'sec. \n') #unit: sec.
print('I.e., the running time is {:.3f} min.'.format((end2 - start2)/60))    


AD_scores = pd.DataFrame(AD_scores, index = data.index)
AD_scores.to_csv(data_dir+'/K8_Xpart_dropna_c_5766_ADscores.data')
"""

