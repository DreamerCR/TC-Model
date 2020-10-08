#!/usr/bin/python
# coding=utf-8

import numpy as np
import h5py

#-----------------------------------------------------------------------------------------------------

def min_max_norm(x):  
    x = np.array(x);x=abs(x)
    x[x>1000]=x[x>1000]/1000
    x[x>100]=x[x>100]/100
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x)) 
    return x_norm

def pl_load(pltxt_path,timestep,r):
    pl_file=h5py.File(pltxt_path,'r')
    pl_var_new=pl_file['var_new']
    pl_label=pl_file['label']

    k1 = 0+4*(10-r); k2 = 81-4*(10-r) 

    pl_var_new=np.array(pl_var_new)
    pl_var_new=min_max_norm(pl_var_new)
    pl_var_new=pl_var_new[:,0:timestep,k1:k2,k1:k2,:,:]

    return pl_var_new,pl_label
    
def sf_load(sftxt_path,timestep,r):
    sf_file=h5py.File(sftxt_path,'r')
    sf_var_new=sf_file['var_new']
    sf_label=sf_file['label']

    k1 = 0+4*(10-r); k2 = 81-4*(10-r)

    # fill NaN
    sf_var_new=np.array(sf_var_new)
    sf_var_new=sf_var_new[:,0:timestep,k1:k2,k1:k2]
    loc_nan=np.isnan(sf_var_new)
    sf_var_new[loc_nan]=0

    # Standardization
    sf_var_new=min_max_norm(sf_var_new)
    
    return np.array(sf_var_new), np.array(sf_label)

