'''
load data from mat file
'''
import numpy as np
import scipy.io as scio
import torch
from utils.derivative import mt1dtm

class Normalizer_out(object):
    '''normalization for field data'''
    def __init__(self, x, eps=0.00001):
        super(Normalizer_out, self).__init__()
        self.mean_real = torch.mean(x.real)
        self.std_real  = torch.std(x.real) #+ eps
        self.mean_imag = torch.mean(x.imag)
        self.std_imag  = torch.std(x.imag) #+ eps
        self.eps = eps
    def decode(self, x0):
        x = torch.ones_like(x0)
        x[...,0] = (x0[...,0] * (self.std_real + self.eps)) + self.mean_real
        x[...,1] = (x0[...,1] * (self.std_imag + self.eps)) + self.mean_imag
        return x
    def to(self,device):
        self.mean_real = self.mean_real.to(device)
        self.std_real  = self.std_real.to(device)
        self.mean_imag = self.mean_imag.to(device)
        self.std_imag  = self.std_imag.to(device)

class Normalizer(object):
    '''normalization for beta, where beta = conductivity * frequency'''
    def __init__(self, x, eps=0.00001):
        super(Normalizer, self).__init__()
        self.mean = torch.mean(torch.log10(x))
        self.std = torch.std(torch.log10(x))
        self.eps = eps
    def encode(self, x):
        x = (torch.log10(x) - self.mean) / (self.std + self.eps)
        return x
    def to(self,device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)


def get_data(file_name):

    key_map = ['rhoxy','phsxy','rhoyx','phsyx']
    data = scio.loadmat(file_name)
    zn = data['zn'][0] 
    yn = data['yn'][0]
    ry  = data['ry'][0]
    sig = data['sig']
    freq = data['freq'][0] 
    nza = data['nza'][0][0]

    response = np.stack([data[key_map[i]] for i in range(len(key_map))], axis=-1)

    return nza,zn, yn, freq, ry, sig,response

def sig_add(sig,nza):
    '''padding for TM mode
    add one row at bottom and one column at right
    remove air layer
    '''

    n_samples,mm,nn = np.shape(sig)
    sig0 = np.zeros((n_samples,mm+1,nn+1))
    sig0[:,:-1,:-1] = sig
    sig0[:,-1, :] = sig[0,-1,-1]
    sig0[:,:-1, -1] = sig[:,:, -1]

    return sig0[:,nza:,:]


def get_loader(train_file,test_file,n_train,n_test,batch_size,f_idx):
    np_dtype = np.float32
    nza,zn, yn, freq0_train, ry, sig0,response0= get_data(train_file)
    zn = zn[nza:]

    t_flatten = torch.nn.Flatten(0,1)
    dy = yn[1:] - yn[:-1]
    dz = zn[1:] - zn[:-1]
    # n_beta = n_train * f_idx[1]

    freq_train0 = (freq0_train[::f_idx[0]][-f_idx[1]:]).reshape(-1,1,1)
    sig = sig0[:n_train]
    response = response0[:n_train,::f_idx[0],...][:,-f_idx[1]:,...]

    sig = sig_add(sig,nza)
    m,ny = np.shape(sig)[-2:]
    imsize = m
    # sig     = torch.from_numpy(sig.astype(np_dtype))
    sig_train= torch.repeat_interleave(torch.from_numpy(sig.astype(np_dtype)),f_idx[1],dim=0)
    freq_train = torch.from_numpy(freq_train0.astype(np_dtype)).repeat(n_train,1,1)
    # result
    # [sig0*f0,...]
    # [sig0*f1,...]
    x_train = sig_train*freq_train
    x_train = x_train.unsqueeze(-1)

    u_bc_train = np.zeros((f_idx[1],imsize),dtype=np.complex64)
    ey1d = np.zeros((f_idx[1],imsize),dtype=np.complex64)
    for ii in range(f_idx[1]):
        ey1d[ii,:],u_bc_train[ii,:] = mt1dtm(freq_train0[ii,0,0],dz,sig[0,:-1,0])    
    u_bc_train = torch.from_numpy(u_bc_train.astype(np.complex64)).repeat(n_train,1)
    ey1d = torch.from_numpy(ey1d.astype(np.complex64)).repeat(n_train,1)


    y_train = t_flatten(torch.from_numpy(response.astype(np_dtype))) # remove value on the boundary
    train_data = torch.utils.data.TensorDataset(x_train, y_train,u_bc_train,ey1d)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True,drop_last=True) 

    _,_,_,freq0_test, ry, sig0,response0= get_data(test_file)
    freq_test0     = (freq0_test[::f_idx[2]][-f_idx[3]:]).reshape(-1,1,1)
    sig      = sig0[:n_test]
    response = response0[:n_test,::f_idx[2],...][:,-f_idx[3]:,...]
    sig = sig_add(sig,nza)
    # sig = torch.from_numpy(sig0[:n_test].astype(np_dtype))
    # x_test = torch.from_numpy(np.expand_dims(sig,-1).astype(np_dtype))
    sig_test = torch.repeat_interleave(torch.from_numpy(sig.astype(np_dtype)),f_idx[3],dim=0)
    freq_test = torch.from_numpy(freq_test0.astype(np_dtype)).repeat(n_test,1,1)
    x_test = sig_test*freq_test
    x_test = x_test.unsqueeze(-1)

    u_bc_test = np.zeros((f_idx[3],imsize),dtype=np.complex64)
    for ii in range(f_idx[3]):
        _,u_bc_test[ii,:] = mt1dtm(freq_test0[ii,0,0],dz,sig[0,:-1,0])
    u_bc_test  = torch.from_numpy(u_bc_test.astype(np.complex64)).repeat(n_test ,1)

    y_test = t_flatten(torch.from_numpy(response.astype(np_dtype)))
    test_data = torch.utils.data.TensorDataset(x_test,y_test,sig_test,freq_test,u_bc_test)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False) 

    x_normalizer = Normalizer(x_train)
    y_normalizer = Normalizer_out(u_bc_train)
    # BC
    ry      = ry.astype(np_dtype)
    yn      = yn.astype(np_dtype)
    zn      = zn.astype(np_dtype)   
    dy = torch.from_numpy(dy.astype(np_dtype))
    dz = torch.from_numpy(dz.astype(np_dtype))


    return zn,yn,dz,dy,ry,train_loader,test_loader, x_normalizer,y_normalizer


if __name__ == '__main__':
    nza,zn,yn,sig,dz,dy,freq,ry,x_train,y_train,imsize=\
        get_loader(train_file='../../Data/data/train_1000.mat',test_file='../../Data/data/test_100.mat',\
            n_train=500,n_test=100,batch_size=10,f_id=25)