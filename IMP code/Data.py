# Class containing the data and functions to load/create batch
# Assumes downsampled data as found in http://www.image-net.org/
import numpy as np
import tensorflow.keras as keras 
import pickle

def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def highPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    return img

def lowPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img2 = np.zeros([h, w])
    img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)]
    return img2

class Data:
  def __init__(self, setname, location, eval_size=0, load=True, ratio=1.0, preload=True, reclass='n'):
    self.setname = setname
    self.preload = preload

    if self.setname=='ImageNet32':
      self.c = 3
      self.h = 32
      self.w = 32
      self.ncl = 1000
      self.norm = 1./255
    elif self.setname=='ImageNet64':
      self.c = 3
      self.h = 64
      self.w = 64
      self.ncl = 1000
      self.norm = 1./255
    elif self.setname=='cifar10':
      self.c = 3
      self.h = 32
      self.w = 32
      self.ncl = 10
      self.norm = 1./255     
    else:
      print("Dataset unknown!")

    rcpar = reclass.split(',')
    if rcpar[0]=='m' or rcpar[0]=='f':
      self.ncl = int(rcpar[1])

    if load:
        if self.setname=='cifar10':
            self.train_size = 50000
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            x_train = x_train.transpose(0, 3, 1, 2)
            x_test = x_test.transpose(0, 3, 1, 2)
            
            for ii in range(50000):
                for jj in range(3):
                    img = np.reshape(x_train[ii, jj, :, :], (32, 32))
                    img_dft = np.fft.fft2(img)
                    dft_shift = np.fft.fftshift(img_dft)
                    dft_shift=lowPassFiltering(dft_shift, 10)
                    idft_shift = np.fft.ifftshift(dft_shift)  #Move the frequency domain from the middle to the upper left corner
                    ifimg = np.fft.ifft2(idft_shift) 
                    ifimg = np.abs(ifimg)
                    x_train[ii, jj,  :] = ifimg
            for ii in range(10000):
                for jj in range(3):
                    img = np.reshape(x_test[ii, jj, :, :], (32, 32))
                    img_dft = np.fft.fft2(img)
                    dft_shift = np.fft.fftshift(img_dft)
                    dft_shift=lowPassFiltering(dft_shift, 10)
                    idft_shift = np.fft.ifftshift(dft_shift)  #Move the frequency domain from the middle to the upper left corner
                    ifimg = np.fft.ifft2(idft_shift) 
                    x_test[ii, jj, :, :] = ifimg        
            self.t_lab = y_train[:, 0]
            self.t_in = x_train
            self.e_lab = y_test[:, 0]
            self.e_in = x_test
            
        else: 
            
            self.train_size = int(ratio*1281167)
            tmp = unpickle(location+"/train_data_batch_1") 
            raw_t_in = tmp['data']
            self.t_lab = np.asarray(tmp['labels'])-1
            self.mean_in = tmp['mean'].astype(np.float32)
            s = 2
            while len(self.t_lab)<self.train_size:            
                tmp = unpickle(location+"/train_data_batch_"+str(s))
                raw_t_in = np.append(raw_t_in, tmp['data'], axis=0)
                self.t_lab = np.append(self.t_lab, np.asarray(tmp['labels'])-1, axis=0)
                s += 1
            raw_t_in = raw_t_in[:self.train_size]
            self.t_lab = self.t_lab[:self.train_size]
            if self.preload:
                self.t_in = np.reshape(self.norm*(np.asarray(raw_t_in, dtype=np.float32)-self.mean_in), (-1,self.c,self.h,self.w))
            else:
                self.t_in = raw_t_in

            tmp = unpickle(location+"/val_data") #50k test data
            self.e_in = np.reshape(self.norm*(np.asarray(tmp['data'][:eval_size], dtype=np.float32)-self.mean_in), (-1,self.c,self.h,self.w))
            self.e_lab = np.asarray(tmp['labels'][:eval_size])-1

            if rcpar[0]=='m':
                self.e_lab = self.e_lab%self.ncl
                self.t_lab = self.t_lab%self.ncl
            elif rcpar[0]=='f':
                indmap = np.loadtxt(rcpar[2], delimiter=',', dtype=int)
                if np.max(indmap)>self.ncl-1:
                    print("More classes than expected!")
                    exit()
                self.e_lab = indmap[self.e_lab]
                self.t_lab = indmap[self.t_lab]

    return

  def getbatch(self, batch_size):
    batch_ind = np.random.choice(self.train_size, batch_size)
    batch_lab = self.t_lab[batch_ind]
    if self.preload:
      batch_in = self.t_in[batch_ind]
    else:
      batch_in = np.reshape(self.norm*(self.t_in[batch_ind].astype(np.float32)-self.mean_in), (-1,self.c,self.h,self.w))
    return batch_in, batch_lab
