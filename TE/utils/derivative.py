'''
numerical differentiation by finite differences
'''

import numpy as np
import torch
import scipy.sparse as scipa 
import scipy.sparse.linalg as scilg

II = complex(0,1)

def sig_decrete(dy,dz,sig):
    # compute volume conductivity

    n_samples,mm,nn = sig.shape
    device = sig.device
    sigc0 = torch.ones((n_samples,mm+1,nn+1),device=device)

    ny = len(dy)
    nz = len(dz)
    #1.compute the system mat
    dz0,dy0 = torch.meshgrid(dz,dy)

    w1 = dy0[0:nz-1,0:ny-1]*dz0[0:nz-1,0:ny-1] # notice: for index, ny==-1,nz==-1
    w2 = dy0[0:nz-1,1:ny]  *dz0[0:nz-1,0:ny-1]
    w3 = dy0[0:nz-1,0:ny-1]*dz0[1:nz,0:ny-1]
    w4 = dy0[0:nz-1,1:ny]  *dz0[1:nz,0:ny-1]
    area = (w1+w2+w3+w4)/4.0
    sigc = (sig[:,0:nz-1,0:ny-1]*w1 + sig[:,0:nz-1,1:ny]*w2 + sig[:,1:nz,:ny-1]*w3 + sig[:,1:nz,1:ny]*w4)/(area*4.0)

    sigc0[:,0,:] = sig[0,0,0]
    sigc0[:,-1,:] = sig[0,-1,-1]
    sigc0[:,1:-1,0] = sigc[:,:,0]
    sigc0[:,1:-1,-1] = sigc[:,:,-1]
    sigc0[:,1:-1,1:-1] = sigc

    return sigc0

def impose_bc(u, u_lr):
    '''impose boundary conditions'''
    _,_,ny,_ = u.size()
    u[:,:,0, 0]  = u_lr.real # left
    u[:,:,0, 1]  = u_lr.imag # left
    u[:,:,-1,0]  = u_lr.real # right
    u[:,:,-1,1]  = u_lr.imag # right
    u[:,0,:, 0]  = u_lr[:,0:1].real.repeat(1,ny) # top
    u[:,0,:, 1]  = u_lr[:,0:1].imag.repeat(1,ny) # top
    u[:,-1,:,0]  = u_lr[:,-1:].real.repeat(1,ny) # bottom
    u[:,-1,:,1]  = u_lr[:,-1:].imag.repeat(1,ny) # bottom



def TE_pde_loss(u0,beta0,u_lr,dy,dz):
    beta = sig_decrete(dy,dz,beta0[...,:-1,:-1,0])
    n_samples = len(beta)
    beta = 2.0*np.pi*beta
    # device = sigc.device
    impose_bc(u0,u_lr)
    u = u0[...,0] + II*u0[...,1]
    
    # beta = omega * sig
    # omega = 2.0*np.pi*freq
    mu = 4.0e-7*np.pi
    ny = len(dy)
    nz = len(dz)

    ex1d = u_lr.unsqueeze(-1).repeat(1,1,ny+1)
    u = u - ex1d
    # using the finite difference method to compute the derivative
    dy00,dz00 = torch.meshgrid(dy,dz)
    dy0 = dy00.T.repeat(n_samples,1,1)
    dz0 = dz00.T.repeat(n_samples,1,1)
    dyc = (dy0[:,0:nz-1,0:ny-1]+dy0[:,0:nz-1,1:ny])/2.0
    dzc = (dz0[:,0:nz-1,0:ny-1]+dz0[:,1:nz,0:ny-1])/2.0
    w1 = dy0[:,0:nz-1,0:ny-1]*dz0[:,0:nz-1,0:ny-1] # notice: for index, ny==-1,nz==-1
    w2 = dy0[:,0:nz-1,1:ny]  *dz0[:,0:nz-1,0:ny-1]
    w3 = dy0[:,0:nz-1,0:ny-1]*dz0[:,1:nz,0:ny-1]
    w4 = dy0[:,0:nz-1,1:ny]  *dz0[:,1:nz,0:ny-1]
    area = (w1+w2+w3+w4)/4.0
    dzdy1 = dzc/dy0[:,0:nz-1,0:ny-1]
    dzdy2 = dzc/dy0[:,0:nz-1,1:ny]
    dydz1 = dyc/dz0[:,0:nz-1,0:ny-1]
    dydz2 = dyc/dz0[:,1:nz,0:ny-1]
    # sigc = (sig[0:nz-1,0:ny-1]*w1 + sig[0:nz-1,1:ny]*w2 + sig[1:nz,:ny-1]*w3 + sig[1:nz,1:ny]*w4)/(area*4.0)
    # val  = dzc/dy0[:,0:nz-1,0:ny-1] + dzc/dy0[:,0:nz-1,1:ny] + dyc/dz0[:,0:nz-1,0:ny-1] +dyc/dz0[:,1:nz,0:ny-1]
    val = dzdy1+dzdy2+dydz1+dydz2


    beta_ref= beta[:,:,0:1].repeat(1,1,ny+1)
    beta_diff = beta - beta_ref
    beta_diff_d = beta_diff[:,1:-1,1:-1] 
    # beta_diff_d = (beta_diff[:,0:nz-1,0:ny-1]*w1 + beta_diff[:,0:nz-1,1:ny]*w2 + \
    #     beta_diff[:,1:nz,:ny-1]*w3 + beta_diff[:,1:nz,1:ny]*w4)/(area*4.0)
    coef = II*mu*beta_diff_d*area
    rhs  = coef * ex1d[:,1:nz,1:ny]

    beta = beta[:,1:-1,1:-1]   
    mtx1 = II*mu*beta*area - val

    f = u[:,:-2,1:-1]*dydz1 + u[:,1:-1,:-2]*dzdy1 +u[:,1:-1,2:]*dzdy2 + \
        u[:,2:,1:-1]*dydz2 + u[:,1:-1,1:-1]*mtx1 + rhs
    MF = 1 #mtx1 #torch.abs(mtx1)
    # MF = torch.mean(torch.stack((torch.abs(dydz1),torch.abs(dydz2),torch.abs(dzdy1),torch.abs(dzdy2),torch.abs(mtx1)),0),0)
    return torch.mean(torch.abs(f/MF*1e6))
    # return torch.mean(((torch.abs(f/MF*1000))**2))


def get_response(out0,sig,dy,dz,ry,yn,nza,freq):
    # compute apparent resistivity and phase from output of the model
    ex = out0[...,0]+II*out0[...,1]
    # ex = out.detach().cpu().numpy()#[...,1:-1,1:-1]

    hys,_ = mt2dhyhz(freq,dy,dz,sig,ex,nza)

    exs = ex[:,nza,:]
    n_samples = len(sig)
    exr = np.zeros((n_samples,len(ry)),dtype=np.complex64)
    hyr = np.zeros((n_samples,len(ry)),dtype=np.complex64)
    for ii in range(n_samples):
        exr[ii,:] = np.interp(ry,yn,exs[ii])
        hyr[ii,:] = np.interp(ry,yn,hys[ii])
    _,rhoxy,phsxy = mt2dzxy(freq[:,:,0],exr,hyr)
    return rhoxy, phsxy

def mt2dhyhz(freq,dy0,dz0,sig,ex,nza):
    #Interpolater of H-field for 2-D Magnetotellurics(MT) TE mode solver.
    omega0 = 2.0*np.pi*freq
    # II = cm.sqrt(-1)
    mu = 4.0e-7*np.pi
    ny = np.size(dy0)
    n_samples = len(sig)
    dy = dy0.reshape(1,-1)*np.ones((n_samples,len(dy0)))
    dz = dz0#.reshape(1,-1)*np.ones((n_samples,len(dz0)))
    #1.compute Hy
    hys = np.zeros((n_samples,ny+1),dtype=complex)    
    #1.1compute Hy at the top left corner
    kk = nza 
    delz = dz[kk]
    sigc = sig[:,kk,0]
    omega = omega0[:,0,0]
    c0 = -1.0/(II*omega*mu*delz) + (3.0/8.0)*sigc*delz
    c1 = 1.0/(II*omega*mu*delz) + (1.0/8.0)*sigc*delz
    hys[:,0] = c0*ex[:,kk,0] + c1*ex[:,kk+1,0]
    #1.2compute Hy at the top right corner
    sigc = sig[:,kk,ny-1]
    c0 = -1.0/(II*omega*mu*delz) + (3.0/8.0)*sigc*delz
    c1 = 1.0/(II*omega*mu*delz) + (1.0/8.0)*sigc*delz
    hys[:,ny] = c0*ex[:,kk,ny] + c1*ex[:,kk+1,ny]
    #1.3compute the Hy at other nodes
    dyj = dy[:,0:ny-1]+dy[:,1:ny]
    sigc = (sig[:,kk,0:ny-1]*dy[:,0:ny-1]+sig[:,kk,1:ny]*dy[:,1:ny])/dyj
    omega = omega0[:,0,:]
    cc = delz/(4.0*II*omega*mu*dyj) # should devided by 8.0?
    c0 = -1.0/(II*omega*mu*delz) + (3.0/8.0)*sigc*delz - cc*3.0*(1.0/dy[:,1:ny]+1.0/dy[:,0:ny-1])
    c1 = 1.0/(II*omega*mu*delz) + (1.0/8.0)*sigc*delz - cc*1.0*(1.0/dy[:,1:ny]+1.0/dy[:,0:ny-1])
    c0l = 3.0*cc/dy[:,0:ny-1]
    c0r = 3.0*cc/dy[:,1:ny]
    c1l = 1.0*cc/dy[:,0:ny-1]
    c1r = 1.0*cc/dy[:,1:ny]
    hys[:,1:ny] = c0l*ex[:,kk,0:ny-1]   + c0*ex[:,kk,1:ny]   + c0r*ex[:,kk,2:ny+1] + \
                c1l*ex[:,kk+1,0:ny-1] + c1*ex[:,kk+1,1:ny] + c1r*ex[:,kk+1,2:ny+1]
    #2.compute Hz
    hzs = np.zeros((n_samples,ny+1),dtype=complex)
    return hys,hzs

    
def mt2dzxy(freq,exr,hyr):
    #compute the impedance, apparent resistivity and phase of TE mode 2-D Magnetotellurics(MT) forward modeling problem
    omega = 2.0*np.pi*freq
    # II = cm.sqrt(-1)
    mu = 4.0e-7*np.pi
    #compute the outputs
    zxy = np.array(exr/hyr,dtype=np.complex64)
    rhote = abs(zxy)**2/(omega*mu)
    phste = np.arctan2(zxy.imag, zxy.real)*180.0/np.pi

    return zxy,rhote,phste

def mt1dte(freq,dz,sig):
    # compute bondary condition of the 1-D magnetotellurics(MT) TE mode 
    miu = 4.0e-7*np.pi
    II = complex(0,1)   
    omega = 2.0*np.pi*freq
    nz = np.size(sig)

    sig = np.hstack((sig,sig[nz-1]))
    dz = np.hstack((dz,np.array(np.sqrt(2.0/(sig[nz]*omega*miu)),dtype=float)))

    
    diagA = II*omega*miu*(sig[0:nz]*dz[0:nz]+sig[1:nz+1]*dz[1:nz+1]) - 2.0/dz[0:nz] - 2.0/dz[1:nz+1]

    
    offdiagA=2.0/dz[1:nz]

    ##system matix
    mtxA = scipa.diags(diagA,format='csc')+scipa.diags(offdiagA,1,format='csc')+scipa.diags(offdiagA,-1,format='csc')
    #compute right hand sides
    ##using boundary conditions:ex[0]=1.0,ex[nz-1]=0.0
    rhs = np.zeros((nz,1),dtype=float)
    rhs[0] = -2.0/dz[0]
    
    lup = scilg.splu(mtxA)
    ex = lup.solve(rhs)
    ex = np.array(np.vstack(([1.0],ex.reshape(-1,1))),dtype=complex)
    return ex.reshape(-1)

# relative L2 error
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = np.linalg.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = np.linalg.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return np.mean(diff_norms/y_norms)
            else:
                return np.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss_out(object):
    ''' 
    multiple output
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        # self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        # input doesn't reshape
        # num_examples = x.size()[0]
        num_out = x.shape[-1]

        diff_norms = np.linalg.norm(x - y,axis=(1))
        y_norms = np.linalg.norm(y, axis=(1))

        if self.reduction:
            if self.size_average:
                return np.mean(diff_norms/y_norms)
            else:
                return np.sum(diff_norms/y_norms)/num_out

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss_out2(object):
    ''' 
    multiple output,分别统计
    '''
    def __init__(self, axis=(1,2), size_average=True, reduction=True):
        #Dimension and Lp-norm type are postive

        self.reduction = reduction
        self.size_average = size_average
        self.axis = axis

    def rel(self, x, y):
        # input doesn't reshape
        # num_examples = x.size()[0]
        # num_out = x.shape[-1]

        diff_norms = np.linalg.norm(x - y,axis=self.axis)
        y_norms = np.linalg.norm(y, axis=self.axis)

        if self.reduction:
            if self.size_average:
                return np.mean(diff_norms/y_norms,axis=0)
            else:
                return np.sum(diff_norms/y_norms,axis=0)

    def __call__(self, x, y):
        return self.rel(x, y)