# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:31:17 2022

@author: redman
"""
# Loading modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import copy
import scipy.stats 
from scipy.stats import binom


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
  
# Select the folder of the computation
data_dir = 'C:/Users/redman/Documents/GitHub/SiftingFeatures/experiments/2 layers IMP/Data statistics/lpf10/'
with open(data_dir+'IMP_findata.pkl', 'rb') as f:
  [arch, ratl, mins] = pickle.load(f)
  ratl = np.mean(ratl, axis=1)
  
# Plotting the best accuracy as a function of sparsity
fig, ax = plt.subplots(1,1, figsize=(10, 5))
plt.tight_layout()
ax.set_xlabel(r"Percentage of unpruned weights ($u$)", fontsize=20, labelpad=0)
ax.set_ylabel(r"Best val. accuracy", fontsize=20)
ax.set_xscale("log")
ax.set_xlim(1.2,0.0005)
xt = [1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1]
plt.xticks(xt, ['0.1','0.3','1','3','10','30','100'], fontsize=20)
plt.yticks(fontsize=20)
ax.plot(1-ratl, 1-np.asarray(mins), 'ko-')
# Find max and put a marker there
maxind = 10 # np.max(np.where(mins < mins[0])) #np.argmin(mins) 
ax.plot(1-ratl[maxind], 1-mins[maxind], 's', ms=10, mew=2, c='k', mfc='w')
print("Best iteration: {} with accuracy {:.2f}%.".format(maxind, 100*(1-mins[maxind])))

NPlot = maxind
data_prefix = 'IMP_itdata_'
with open(data_dir+data_prefix+str(NPlot)+'.pkl', 'rb') as f:
  [arch, ratio, Tpoints, sl, sel, tl, el, weil, mvl, ml] = pickle.load(f)
print("Loaded file {}, found iterations: {}".format(data_prefix+str(NPlot),Tpoints))

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.set_xscale("log")
ax.set_xlim(10.0, sl[-1]*1.1)
ax.set_xlabel("Step", fontsize=20)
ax.set_ylabel("Error", fontsize=20)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.plot(sl,tl,'-',label="Train")
ax.plot(sel,el,'-',label="Validation")
ax.legend(fontsize=20)

# Params
Nlay = 1
Mx = DDw
My = DDh
cutoff = 1e7

fig, ax = plt.subplots(1,2, figsize=(4*2,4.4))
ax[0].set_xlabel(r"$S^{\rm sc}(\mathbf{d})$", fontsize=20)
ax[1].set_xlabel(r"$S^{\rm dc}(\mathbf{d})$", fontsize=20)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.tight_layout()

cmap = np.zeros((2*Mx+1,2*My+1,3))
mm = ml[0][0]
for l in range(1,Nlay):
  mm = mm@ml[l][0]

nnc = 0
# Iterating over nodes
for rr, spc in enumerate(mm.T):
  indices = np.nonzero(spc)[0]
  li = len(indices)
  if li>1:
    nnc += li**2
    # Get original location
    ys, xs = np.divmod(indices, DDw)
    cs, ys = np.divmod(ys, DDh)
    # Compute x and y differences (with clipping) 
    xdiff = np.clip(xs[:-1,np.newaxis]-xs[np.newaxis,1:],-Mx,Mx)
    ydiff = np.clip(ys[:-1,np.newaxis]-ys[np.newaxis,1:],-My,My)
    cldiff = cs[:-1,np.newaxis]-cs[np.newaxis,1:]
    # For each i only get j>i and fill the other too
    for ii in range(li-1):
      for dx, dy, dc in zip(xdiff[ii,ii:],ydiff[ii,ii:],cldiff[ii,ii:]):
        cmap[dy+My,dx+Mx,dc] += 1
        cmap[-dy+My,-dx+Mx,-dc] += 1
  if nnc>cutoff:
      break
# Normalize and plot
scmax = np.max(cmap[:,:,0])
dcmax = np.max(np.sum(cmap[:,:,1:3],axis=2))
ax[0].imshow(cmap[:,:,0], aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=scmax)
ax[1].imshow(cmap[:,:,1]+cmap[:,:,2], aspect='auto', interpolation='none', cmap='plasma', vmin=0, vmax=dcmax)

#Params
Nlay = 1
Bwid = 1

fig, ax = plt.subplots(1,1, figsize=(10,5))
#ax.set_yscale("log")
ax.set_xlabel(r'$C^{{\rm in, {{{}}}}}$'.format(Nlay), fontsize=20)
ax.set_ylabel("Count", fontsize=20)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

mm = ml[0][0]
for l in range(1,Nlay):
  mm = mm@ml[l][0]

Nnei = [len(np.nonzero(spc)[0]) for spc in mm.T]
minb = np.min(Nnei)
maxb = np.max(Nnei)
_, b, _ = ax.hist(Nnei, bins=np.arange(minb,maxb+Bwid,Bwid)-Bwid+0.5, density=False)
xx = (b[1:] + b[:-1])/2
n0 = 1024 * 3
u = 1-ratl[maxind]
yy = binom.pmf(xx, n0, u) * np.size(Nnei)
plt.plot(xx, yy, 'k--')