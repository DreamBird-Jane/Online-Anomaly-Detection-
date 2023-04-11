# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:14:26 2019

@author: Jane
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### function testing#####################

# ========= function: PowerMethod ========================================================
def PowerMethod(Cov_A,x,tol,maxiter):
    '''
% This sub-function is computing the eigenvector via power method
%--------------------------------------------------------------------
%   Cov_A: the covarance matrix of data matrix A, each row represent an instance
%   x: initial vector
%   tol: the tolerance for convergence (0<tol<1)
%   maxiter: the max iteration in updating the of eigenvector
%
% output
%   lambda: the resulting eigenvalue
%   v: the resulting eigenvector
%
%--------------------------------------------------------------------
    '''
    relerr = np.inf # while loop starter;also as stopping criterion for measuring error of successive alpha's
    niter = 1
    
    while relerr >= tol and niter < maxiter:
        z= x/np.linalg.norm(x,2)
        x=np.dot(Cov_A,z)
        alpha1 = np.dot(z.T,x)  
        #used to iteratively derive bigest eigenvalue(lambda1) of Cov_A
        #Since  lambda1 = x'*Cov_A*x if x=(unit) eigenvector
    
        if niter >1:
            relerr = abs( (alpha1 - alpha0)/alpha0 );
            
        alpha0 = alpha1
        niter = niter+1
        
    lambda1 = alpha1
    v = z
    return (lambda1,v)
#=========================================================================


#=================Main function: osPCA ==============================================================
def OD_wpca(A,ratio):
    '''
% Outlier detection via over-sampling PCA
%
% This function is used for outlier detection. The main idea is using 
% the variation of the first principal direction detect the outlierness
% of each instance(event) in the leave one out procedure. Here the 
% over-sampling on target instance is also used for enlarge the 
% outlierness
%
%
% input
%   A: the data matrix (dataframe/np.array type), each row represent an instance
%   ratio: the ratio of the oversampling
%          For example, ratio=0.1 means we duplicate the targeted instance
%          with 10 percentage of the whole data
% output
%   suspicious_score: the suspicious score for each instance
%   suspicious_index: the ranking(j) of instances according to their
%                     suspicious score; 
%                     showing the first element as the position(index) of the most suspicious instance
%                     and the last element as the position(index) of the least suspicious instance
%
%                     For example, suspicious_index(i)=j means the ith 
%                     instance is in jth position in the ranking.
%
%---------------------------------------------------------------------------
'''
    #% the threshold in Power Method
    tol=10**(-20);maxiter=500;
    A = pd.DataFrame(A)
    
    n,p = A.shape
    A_m = A.mean()
    
    out_prod = np.dot(A.T,A)/n #outer product of A divided by n, i.e., Cov(A)
    #Cov_A ~ out_prod-A_m'*A_m (population) is covariance matrix
    #Cov_A = A.cov()  #(sample)
    
    lambda1,u = PowerMethod(out_prod-np.outer(A_m,A_m),np.ones((p,1)),tol,maxiter);
    #% start the "LOO" procedure with over-sampling PCA
    sim_pool = np.zeros((n,1));
    
    for i in range(n):
        temp_mu = (A_m + ratio*A.iloc[i,:])/(1+ratio); #% update of mean
        #check if temp_cov correct??????????????????????????????????????????????????????????????
        temp_cov = ( out_prod + ratio*np.outer(A.iloc[i,:],A.iloc[i,:]) )/(1+ratio)
        lambda1,u_temp = PowerMethod(temp_cov,np.ones((p,1)),tol,maxiter);
        sim_pool[i] = abs(np.dot(u.T,u_temp)) # compute the cosine similarity
        
        if (i % 1000)== 0: #report the iteration progress
            print("Iteration",str(i))

    #calculate the suspicious_index and suspicious_score
    import operator
    temp =sorted(enumerate(sim_pool), key=operator.itemgetter(1))
    temp = pd.DataFrame(temp)
    suspicious_index = temp.iloc[:,0]

    suspicious_score = 1-sim_pool

    return (suspicious_index, suspicious_score, u, d)
#=====================================================================================================

