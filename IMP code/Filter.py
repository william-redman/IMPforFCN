# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:23:13 2022

@author: redman
"""

def highPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    return img

class Filter:
    def __init__:
        