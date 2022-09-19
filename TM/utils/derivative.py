'''
numerical differentiation by finite differences
'''

import numpy as np
import cmath as cm
import torch
import scipy.sparse as scipa 
import scipy.sparse.linalg as scilg

II = cm.sqrt(-1)

def impose_bc(u, u_lr):
    ''' hard bc enforcement'''
    _,_,ny,_ = u.size()
    u[:,:,0, 0]  = u_lr.real # left
    u[:,:,0, 1]  = u_lr.imag # left
    u[:,:,-1,0]  = u_lr.real # right
    u[:,:,-1,1]  = u_lr.imag # right
    u[:,0,:, 0]  = u_lr[:,0:1].real.repeat(1,ny) # top
    u[:,0,:, 1]  = u_lr[:,0:1].imag.repeat(1,ny) # top
    u[:,-1,:,0]  = u_lr[:,-1:].real.repeat(1,ny) # bottom
    u[:,-1,:,1]  = u_lr[:,-1:].imag.repeat(1,ny) # bottom



def TM_pde_loss(u0,beta,u_lr,ey1d,dy,dz):
    n_samples = len(beta)
    # device = sigc.device
    impose_bc(u0,u_lr)
    u = u0[...,0] + u0[...,1]*II
    
    # beta = omega * sig
    beta = 2.0*np.pi*beta[...,0]
    # omega = 2.0*np.pi*freq
    mu = 4.0e-7*np.pi
    ny = len(dy)
    nz = len(dz)

    ey1d = ey1d.unsqueeze(-1).repeat(1,1,ny+1)
    hx1d = u_lr.unsqueeze(-1).repeat(1,1,ny+1)
    beta_ref= beta[:,:,0:1].repeat(1,1,ny+1)
    beta_diff = beta - beta_ref
    u = u - hx1d


    # using the finite difference method to compute the derivative
    dy00,dz00 = torch.meshgrid(dy,dz)
    dy0 = dy00.T.repeat(n_samples,1,1)
    dz0 = dz00.T.repeat(n_samples,1,1)
    dyc = (dy0[:,0:nz-1,0:ny-1]+dy0[:,0:nz-1,1:ny])/2.0
    dzc = (dz0[:,0:nz-1,0:ny-1]+dz0[:,1:nz,0:ny-1])/2.0
    w1 = 2 * dz0[:,0:nz-1, 0:ny-1] # dz1
    w2 = 2 * dz0[:,1:nz,   0:ny-1] # dz2
    w3 = 2 * dy0[:,0:nz-1, 0:ny-1] # dy1
    w4 = 2 * dy0[:,0:nz-1, 1:ny]   # dy2
    # area = (w1+w2+w3+w4)/4.0
    A = (1.0/beta[:,0:nz-1,0:ny-1] * dy0[:,0:nz-1,0:ny-1] + 1.0/beta[:,0:nz-1,1:ny]*dy0[:,0:nz-1,1:ny])/w1 # (dy1 * rho_11 + dy2 * rho_12) / (2*dz1)
    B = (1.0/beta[:,1:nz  ,0:ny-1] * dy0[:,0:nz-1,0:ny-1] + 1.0/beta[:,1:nz  ,1:ny]*dy0[:,0:nz-1,1:ny])/w2 # (dy1 * rho_21 + dy2 * rho_22) / (2*dz2)
    C = (1.0/beta[:,0:nz-1,0:ny-1] * dz0[:,0:nz-1,0:ny-1] + 1.0/beta[:,1:nz,0:ny-1]*dz0[:,1:nz,0:ny-1])/w3 # (dz1 * rho_11 * dz2 * rho_21) / (2*dy1)
    D = (1.0/beta[:,0:nz-1,1:ny  ] * dz0[:,0:nz-1,0:ny-1] + 1.0/beta[:,1:nz,1:ny  ]*dz0[:,1:nz,0:ny-1])/w4 # (dz1 * rho_12 + dz2 * rho_22) / (2*dy2)
    mtx22 = II * mu * dyc * dzc - A - B - C - D
    mtx12 = A
    mtx21 = C
    mtx32 = B
    mtx23 = D

    A1 = dy0[:,0:nz-1,0:ny-1]*dz0[:,0:nz-1,0:ny-1] # notice: for index, ny==-1,nz==-1
    A2 = dy0[:,0:nz-1,1:ny]  *dz0[:,0:nz-1,0:ny-1]
    A3 = dy0[:,0:nz-1,0:ny-1]*dz0[:,1:nz,0:ny-1]
    A4 = dy0[:,0:nz-1,1:ny]  *dz0[:,1:nz,0:ny-1]
    area = (A1+A2+A3+A4)/4.0
    beta_scale = beta_diff/beta
    beta_diff_d = (beta_scale[:,0:nz-1,0:ny-1]*A1 + beta_scale[:,0:nz-1,1:ny]*A2 + \
        beta_scale[:,1:nz,0:ny-1]*A3 + beta_scale[:,1:nz,1:ny]*A4)/(area*4.0)
    coef = II*mu*beta_diff_d* area
    hx_d  = coef * hx1d[:,1:nz,1:ny]

    beta_t = (beta_scale[:,0:nz-1,0:ny-1] * dy0[:,0:nz-1,0:ny-1] + beta_scale[:,0:nz-1,1:ny]*dy0[:,0:nz-1,1:ny])/(dy0[:,0:nz-1,0:ny-1]+dy0[:,0:nz-1,1:ny]) # (dy1 * sig_11 + dy2 * sig_12) /(dy1+dy2)
    beta_b = (beta_scale[:,1:nz  ,0:ny-1] * dy0[:,1:nz  ,0:ny-1] + beta_scale[:,1:nz  ,1:ny]*dy0[:,1:nz  ,1:ny])/(dy0[:,1:nz  ,0:ny-1]+dy0[:,1:nz  ,1:ny]) # (dy1 * sig_21 + dy2 * sig_22) /(dy1+dy2)
    ey_d = (beta_b - beta_t)/((dz0[:,0:nz-1,0:ny-1]+dz0[:,1:nz,0:ny-1])/2.0)*area*ey1d[:,1:nz,1:ny]
    rhs = hx_d - ey_d
    
    f = u[:,:-2,1:-1]*mtx12 + u[:,1:-1,:-2]*mtx21 +u[:,1:-1,2:]*mtx23 + \
        u[:,2:,1:-1]*mtx32 + u[:,1:-1,1:-1]*mtx22 +rhs

    MF = 1 #torch.abs(mtx22) #torch.abs(mtx1)
    # MF = torch.mean(torch.stack((torch.abs(mtx12),torch.abs(mtx21),torch.abs(mtx23),torch.abs(mtx32),torch.abs(mtx22)),0),0)
    return torch.mean(((torch.abs(f/MF*1e6))))


def get_response(out0,sig,dy,dz,ry,yn,freq):
    # compute apparent resistivity and phase from output of the model
    hx = out0[...,0] + out0[...,1]*II
    # ex = out.detach().cpu().numpy()#[...,1:-1,1:-1]

    eys,_ = mt2deyez(freq,dy,dz,sig,hx)

    hxs = hx[:,0,:]
    n_samples = len(sig)
    hxr = np.zeros((n_samples,len(ry)),dtype=np.complex64)
    eyr = np.zeros((n_samples,len(ry)),dtype=np.complex64)
    for ii in range(n_samples):
        hxr[ii,:] = np.interp(ry,yn,hxs[ii])
        eyr[ii,:] = np.interp(ry,yn,eys[ii])
    _,rhoyx,phsyx = mt2dzyx(freq[:,:,0],hxr,eyr)
    return rhoyx, phsyx

def mt2deyez(freq,dy0,dz0,sig,hx):
    #Interpolater of H-field for 2-D Magnetotellurics(MT) TE mode solver.
    omega0 = 2.0*np.pi*freq
    miu = 4.0e-7*np.pi
    ny = np.size(dy0)
    n_samples = len(sig)
    dy = dy0.reshape(1,-1)*np.ones((n_samples,len(dy0)))
    dz = dz0#
    #1.compute Hy
    eys = np.zeros((n_samples,ny+1),dtype=np.complex64)    
    #1.1compute Hy at the top left corner
    # kk = self.nza 
    kk = 0 # no air layer
    delz = dz[kk]
    sigc = sig[:,kk,0]
    omega = omega0[:,0,0]
    temp_beta = II * omega * miu * delz
    temp_1 = sigc * delz
    c0 = -1.0/temp_1 + (3.0/8.0)*temp_beta
    c1 = 1.0/temp_1 + (1.0/8.0)*temp_beta
    eys[:,0] = c0*hx[:,kk,0] + c1*hx[:,kk+1,0]
    #1.2compute Hy at the top right corner
    sigc = sig[:,kk,ny-1]
    temp_1 = sigc * delz
    c0 = -1.0/temp_1 + (3.0/8.0)*temp_beta
    c1 = 1.0/temp_1+ (1.0/8.0)*temp_beta
    eys[:,ny] = c0*hx[:,kk,ny] + c1*hx[:,kk+1,ny]
    #1.3compute the Hy at other nodes
    # for kj in range(1,ny):
    dyj = (dy[:,0:ny-1]+dy[:,1:ny])/2.0
    tao = 1.0/sig[:,kk,0:ny]
    omega = omega0[:,0,:]
    taoc = (tao[:,0:ny-1]*dy[:,0:ny-1] + tao[:,1:ny]*dy[:,1:ny])/(2*dyj)
    temp_1 = II*omega*miu*delz
    temp_2 = taoc/delz
    temp_3 = delz/dyj
    temp_4 = tao/dy
    c0 =  (3.0/8.0)*temp_1 - temp_2
    c1 =  (1.0/8.0)*temp_1 + temp_2 - (1.0/8.0)*temp_3*(temp_4[:,0:ny-1]+temp_4[:,1:ny])
    c1l = (1.0/8.0)*temp_3*temp_4[:,0:ny-1]
    c1r = (1.0/8.0)*temp_3*temp_4[:,1:ny]
    eys[:,1:ny] = c0*hx[:,kk,1:ny] + c1l*hx[:,kk+1,0:ny-1]+c1*hx[:,kk+1,1:ny]+c1r*hx[:,kk+1,2:ny+1]
    #2.compute Hz
    ezs = np.zeros((n_samples,ny+1),dtype=np.complex64)

    return eys,ezs
    
def mt2dzyx(freq,hxr,eyr):
    #compute the impedance, apparent resistivity and phase of TE mode 2-D Magnetotellurics(MT) forward modeling problem
    omega = 2.0*np.pi*freq
    # II = cm.sqrt(-1)
    miu = 4.0e-7*np.pi
    zyx = np.array(eyr/hxr,dtype=complex)
    rhotm = abs(zyx)**2/(omega*miu)
    phstm = np.arctan2(zyx.imag, zyx.real)*180.0/np.pi

    return zyx,rhotm,phstm

def mt1dtm(freq,dz0,sig0,n_add=10):
    # compute bondary condition of the 1-D magnetotellurics(MT) TM mode 
    mu = 4.0e-7*np.pi
    omega = 2.0*np.pi*freq
    # nz = np.size(sig0)

    dz = np.array([dz0[i]/n_add*np.ones(n_add) for i in range(np.size(dz0))]).flatten()
    sig = np.array([sig0[i]*np.ones(n_add) for i in range(np.size(dz0))]).flatten()
    nz = len(sig)

    sig = np.hstack((sig,sig[nz-1]))
    dz = np.hstack((dz,np.array(np.sqrt(2.0/(sig[nz]*omega*mu)),dtype=float)))
    
    diagA = II*omega*mu*(dz[0:nz]+dz[1:nz+1]) - 2.0/(dz[0:nz]*sig[0:nz]) - 2.0/(dz[1:nz+1]*sig[1:nz+1])
    
    offdiagA=2.0/(dz[1:nz]*sig[1:nz])
    
    ##system matix
    mtxA = scipa.diags(diagA,format='csc')+scipa.diags(offdiagA,-1,format='csc')+scipa.diags(offdiagA,1,format='csc')
    #compute right hand sides
    ##using boundary conditions:ex[0]=1.0,ex[nz-1]=0.0
    # BCs
    rhs = np.zeros((nz,1))
    rhs[0] = -2.0/(dz[0]*sig[0])

    # hy,_ = self.equation_solve(mtxA,rhs)
    lup = scilg.splu(mtxA)
    hx0 = lup.solve(rhs)
    hx = np.concatenate(([complex(1,0)],hx0.reshape(-1)))
    ey0 = (hx[1:]-hx[:-1])/dz[:-1]/sig[:-1] / omega # omega for secondary field
    ey = np.concatenate((ey0,ey0[-1:]))
    idx = np.arange(np.size(sig0)+1)*n_add
    # ey_n = np.concatenate((ey[idx],ey[:-1]))
    # hx_n = np.concatenate((hx[idx],hx[:-1]))
    # return ey_n,hx_n
    return ey[idx].reshape(-1), hx[idx].reshape(-1)

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
    multiple output,
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