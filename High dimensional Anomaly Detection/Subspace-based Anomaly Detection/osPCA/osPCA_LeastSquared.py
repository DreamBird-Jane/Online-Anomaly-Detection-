# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:01:50 2019

@author: USER
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


############################## Fuctions for deriving eigenvectors ######################################################

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





## ---------------------------------------------Track w (updating PC by instance x) --------------
def Track_w(x, w, d, beta):
    '''
    input:
        x: [pd.Series type] each row of mean-centered instance (dim = p)
        w: original Principal Component (PC) [same with label u]
        d: sum of yi^2 (where yi = u.T * xi_bar; xi_bar = xi - mean(x))
        beta: forgetting factor (weight to decrease the impact of the previous data instances
              by a factor of beta)
    output:
        w: updated PC by the new instance x (not unit scaled)
        d: updated d by adding the info. of new instance x
    '''
    x = np.array(x).reshape((1,-1))
    y = np.dot(x, w)
    d = beta*d + y**2
    e = x.T - w*y
    
    #update x according to x
    w = w + (y*e)/d
    
    return (w,d)


## ---------------------------------------------Track w 2 (updating PC by instance x) --------------
def Track_w_2(x, w, d, n, ratio):
    '''
    input:
        x: [pd.Series type] each row of mean-centered instance (dim = p)
        w: original Principal Component (PC) [same with label u]
        d:
        beta: forgetting factor (weight to decrease the impact of the previous data instances
              by a factor of beta)
    output:
        w: updated PC by the new instance x (not unit scaled)
        d: updated d by adding the info. of new instance x
    '''
    x = np.array(x).reshape((1,-1))
    y = np.dot(x, w)
    d = (1/n)*d + ratio*y**2
    e = x.T - w*y
    
    #update x according to x
    w = w + (y*e)/d
    
    return (w,d)
## ---------------------------------------------Track w (updating PC by instance x with LS method - full ) --------------
def Track_w_ls(x, w, num, den, beta):
    '''
    input:
        x: [pd.Series type] each row of mean-centered instance (dim = p)
        w: original Principal Component (PC) [same with label u]
        d:
        beta: forgetting factor (weight to decrease the impact of the previous data instances
              by a factor of beta)
    output:
        w: updated PC by the new instance x (not unit scaled)
        d: updated d by adding the info. of new instance x
    '''
    x = np.array(x).reshape((1,-1))
    y = np.dot(x, w)
    num = beta*num + y*x.T
    den = beta*den + y**2
    
    #update x according to x
    w = num/den
    
    return (w,num,den)        
        

#################################### functions for suspecious scoring ################################################
# OD_onlinePCA: derive initial eigenvector(u) and over-sampline updated u both by online LS method ====================
def OD_onlinePCA(A, beta):
    '''
    function [suspicious_index suspicious_score u] = OD_onlinePCA(A, beta)
%
% Outlier detection via over-sampling online PCA
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
%   beta: forgetting factor
%         For example, beta=0.9 means we decrease the influence of previous
%         data by a factor of 0.9
% output
%   suspicious_score: the suspicious score for each instance
%   suspicious_index: the ranking of instances according to their
%                     suspicious score
%                     For example, suspicious_index(i)=j means the ith 
%                     instance is in jth position in the ranking.
%   u: updated PC (not unit scaled)
%   d:
%
% References
% Anomaly Detection via Over-Sampling Principal Component Analysis                                                                    
% Send your comment and inquiry to Dr. Yi-Ren Yeh (yirenyeh@gmail.com)        
%
    '''
    
    #Define parameters
    A = pd.DataFrame(A)
    
    n,p = A.shape
    A_m = A.mean()
    
    d = 0.0001 #0.0001
    u = np.ones((p,1))
    
    # update 1st principal components(PC) - u iteratively by each instance
    for i in range(n): 
        u,d = Track_w(A.iloc[i,:]-A_m, u, d, 1)
        #print("iterate {}\n{} \n\n {}\n\n".format(i,u,d))
        
    u = u/np.linalg.norm(u,2)
    
    # start the "LOO" procedure with over-sampling PCA
    sim_pool = np.zeros((n,1));
    ratio = 1/(n*beta)  # for function: Track_w_2
    
    for i in range(n): 
        #update mean(miu) by original mean A_m
        temp_mu = (A_m + ratio*A.iloc[i,:])/(1+ratio)
        x = A.iloc[i,:] - temp_mu
        w = Track_w(x, u, d, beta)[0]
        #w = Track_w_2(x, u, d, n, ratio)[0]
        w = w/np.linalg.norm(w,2)  #updated PC by oversampling ith instance x
        sim_pool[i] = abs(np.dot(u.T,w)) # compute the cosine similarity
        if (i % 1000)== 0:  #report the iteration progress
            print("Iteration",str(i))
            
    #calculate the suspicious_index and suspicious_score
    import operator
    temp =sorted(enumerate(sim_pool), key=operator.itemgetter(1))
    temp = pd.DataFrame(temp)
    suspicious_index = temp.iloc[:,0]

    suspicious_score = 1-sim_pool

    return (suspicious_index, suspicious_score, u, d)


# OD_onlinePCA: derive initial eigenvector(u) and over-sampline updated u both by online LS method ====================
#=================Main function: osPCA ==============================================================
def OD_hybridPCA(A,beta):
    '''
% Outlier detection via over-sampling PCA
%
% This function is used for outlier detection. The main idea is using 
% the variation of the first principal direction (derived by Power Method) detect the outlierness
% of each instance(event) in the leave one out procedure. Here the 
% over-sampling on target instance is also used for enlarge the 
% outlierness (updated principal is derived by Least-squared method)
% hybrid: PowerMethed + Least-Square method
%
%
% input
%   A: the data matrix (dataframe/np.array type), each row represent an instance
%   beta: forgetting factor
%         For example, beta=0.9 means we decrease the influence of previous
%         data by a factor of 0.9 
%   (ratio: the ratio of the oversampling
%          For example, ratio=0.1 means we duplicate the targeted instance
%          with 10 percentage of the whole data)
% output
%   suspicious_score: the suspicious score for each instance
%   suspicious_index: the ranking(j) of instances according to their
%                     suspicious score; 
%                     showing the first element as the position(index) of the most suspicious instance
%                     and the last element as the position(index) of the least suspicious instance
%
%                     For example, suspicious_index(i)=j means the ith 
%                     instance is in jth position in the ranking.
%   u:
%   d:
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
    
    # Derive initial eigenvector by PowerMethod
    u = PowerMethod(out_prod-np.outer(A_m,A_m),np.ones((p,1)),tol,maxiter)[1];
    #% start the "LOO" procedure with over-sampling PCA
    sim_pool = np.zeros((n,1));
    ratio = 1/(n*beta)  # for function: Track_w_2
    '''
    for i in range(n):
        temp_mu = (A_m + ratio*A.iloc[i,:])/(1+ratio); #% update of mean
        #check if temp_cov correct??????????????????????????????????????????????????????????????
        temp_cov = ( out_prod + ratio*np.outer(A.iloc[i,:],A.iloc[i,:]) )/(1+ratio)
        lambda1,u_temp = PowerMethod(temp_cov,np.ones((p,1)),tol,maxiter);
        sim_pool[i] = abs(np.dot(u.T,u_temp)) # compute the cosine similarity
        
        if (i % 1000)== 0: #report the iteration progress
            print("Iteration",str(i))
    '''
    
    A_0 = A.subtract(A_m) 
    P = np.dot(A_0.T,A_0)
    num0 = np.dot(P,u)     # sum of yi*xi_bar  for i=1,...,n
    den0 = np.dot(u.T,num0) #sum o of yi^2 for i= 1,...n


    for i in range(n): 
        #update mean(miu) by original mean A_m
        temp_mu = (A_m + ratio*A.iloc[i,:])/(1+ratio)
        x = A.iloc[i,:] - temp_mu
        w = Track_w(x, u, den0, beta)[0]
        #w = Track_w_2(x, u, d, n, ratio)[0]
        w = w/np.linalg.norm(w,2)  #updated PC by oversampling ith instance x
        sim_pool[i] = abs(np.dot(u.T,w)) # compute the cosine similarity
        if (i % 1000)== 0:  #report the iteration progress
            print("Iteration",str(i))


    #calculate the suspicious_index and suspicious_score
    import operator
    temp =sorted(enumerate(sim_pool), key=operator.itemgetter(1))
    temp = pd.DataFrame(temp)
    suspicious_index = temp.iloc[:,0]

    suspicious_score = 1-sim_pool

    #return (suspicious_index, suspicious_score, u, d)
    return (suspicious_index, suspicious_score, u)
#=====================================================================================================

