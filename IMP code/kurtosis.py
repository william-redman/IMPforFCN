# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:47:47 2022

@author: redman
"""
# Loading modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import copy

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
  
# Setting the dataset as ImageNet32 or 64
dataset = 'ImageNet32'
# dataset = 'ImageNet64'
N_CLASSES = 10
if dataset=='ImageNet32':
  DDc = 3
  DDh = 32
  DDw = 32
elif dataset=='ImageNet64':
  DDc = 3
  DDh = 64
  DDw = 64
elif dataset=='cifar10':
  DDc = 3
  DDh = 32
  DDw = 32

n_rounds = 4
n_rounds_base = 22
n0 = 1024
data_dir = 'C:/Users/redman/Documents/GitHub/SiftingFeatures/experiments/2 layers IMP/Pruning amount/0.9 pruning/'
base_dir = 'C:/Users/redman/Documents/GitHub/SiftingFeatures/experiments/2 layers IMP/Base/'
# Kurtosis
K = np.zeros([n0, n_rounds])
K_base = np.zeros([n0, n_rounds_base])
data_prefix = 'IMP_itdata_'
for count in range(n_rounds):
    with open(data_dir+data_prefix+str(count)+'.pkl', 'rb') as f:
        [arch, ratio, Tpoints, sl, sel, tl, el, weil, mvl, ml] = pickle.load(f)
        W = weil[0][0][0]
        M = ml[0][0]
        Weff = W * M
        for nn in range(n0):
            if np.sum(Weff[:, nn]**2) > 0:
                K[nn, count] = np.sum(Weff[:, nn]**4) / (np.sum(Weff[:, nn]**2)**2)
            else:
                K[nn, count] = np.nan
for count in range(n_rounds_base):           
    with open(base_dir+data_prefix+str(count)+'.pkl', 'rb') as f:
        [arch, ratio, Tpoints, sl, sel, tl, el, weil, mvl, ml] = pickle.load(f)
        W = weil[0][0][0]
        M = ml[0][0]
        Weff = W * M
        for nn in range(n0):
            if np.sum(Weff[:, nn]**2) > 0:
                K_base[nn, count] = np.sum(Weff[:, nn]**4) / (np.sum(Weff[:, nn]**2)**2)
            else:
               K_base[nn, count] = np.nan 
# Plotting
x = (1-0.9)**(np.linspace(0, n_rounds - 1, n_rounds))
x_base = (1-0.3)**(np.linspace(0, n_rounds_base - 1, n_rounds_base))
fig, ax = plt.subplots(1,1, figsize=(10, 5))
plt.tight_layout()
ax.set_xlabel(r"Percentage of unpruned weights ($u$)", fontsize=20, labelpad=0)
ax.set_ylabel(r"IPR", fontsize=20)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1.2,0.0005)
xt = [1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1]
plt.xticks(xt, ['0.1','0.3','1','3','10','30','100'], fontsize=20)
plt.yticks(fontsize=20)
for ii in range(n0):
    ax.plot(x, K[ii, :], '-', color=(0.8, 0.8, 0.8), linewidth=0.5)
    #ax.plot(x_base, K_base[ii, 0] * np.ones(n_rounds_base), 'r-', linewidth=0.5)
    ax.plot(x_base, K_base[ii, :], '-', color=(0.8, 0, 0.8), linewidth=0.5)
ax.plot(x, np.nanmean(K, axis=0), 'ko-')
ax.plot(x_base, np.nanmean(K_base, axis=0), 'bo-')