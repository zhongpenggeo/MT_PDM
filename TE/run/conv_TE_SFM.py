'''

'''
import os
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
# import torch.nn as nn
# import torch.nn.functional as F
from timeit import default_timer
# from torchinfo import summary
import torch.optim as optim
# from utils.practices import OneCycleScheduler, adjust_learning_rate
import yaml

import sys
sys.path.append("..")

# # from untils.derivative import TE_derivative_complex, TE_derivative_complex_MF
# sys.path.append("../untils/") 
from utils.FNO import *
from utils.load_data import *
from utils.derivative import *

def impose_bc(u):
    u[:,:,0, 0]  =  0
    u[:,:,0, 1]  =  0
    u[:,:,-1,0]  =  0
    u[:,:,-1,1]  =  0
    u[:,0,:, 0]  =  0
    u[:,0,:, 1]  =  0
    u[:,-1,:,0]  =  0
    u[:,-1,:,1]  =  0



def TE_pde_loss(u0,beta0,u_lr,dy,dz):
    beta = sig_decrete(dy,dz,beta0[...,:-1,:-1,0])
    n_samples = len(beta)
    beta = 2.0*np.pi*beta
    # device = sigc.device
    impose_bc(u0)
    u = u0[...,0] + II*u0[...,1]
    
    # beta = omega * sig
    # omega = 2.0*np.pi*freq
    mu = 4.0e-7*np.pi
    ny = len(dy)
    nz = len(dz)

    ex1d = u_lr.unsqueeze(-1).repeat(1,1,ny+1)
    # u = u - ex1d
    #1.compute the system mat
    # 展平为2维方便矩阵计算
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

class conv_TE_pde(object):

    def __init__(self,train_file,test_file,n_train,n_test,batch_size,f_idx,device,\
                    modes1, modes2, width, layer_num, last_size, act_fno,init_func,\
                      tf_lr=1e-3,weight_decay=1e-4,step_size=50,gamma=0.5):
        # self.file_name = file_name
        self.II = complex(0,1)
        # out is 1 channels for complex number
        n_out = 2
        nza,zn,yn,dz,dy,ry,train_loader,test_loader,x_normalizer,y_normalizer\
             = get_loader(train_file,test_file,n_train,n_test,batch_size,f_idx)
        
        # self.ex = ex
        self.nza = nza
        self.n_freq_train = f_idx[1]
        self.n_freq_test  = f_idx[3]
        self.n_train = n_train
        self.n_test  = n_test
        self.yn = yn
        self.zn = zn
        self.dz = dz.to(device)
        self.dy = dy.to(device)
        self.ry = ry
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        # self.x_normalizer.to(device)
        # self.y_normalizer.to(device)

        self.model = FNO2d(modes1, modes2, width, n_out, layer_num, last_size, act_fno,init_func).to(device)
        self.tf_optimizer = optim.Adam(self.model.parameters(), lr=tf_lr,
                        weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.tf_optimizer, step_size=step_size, gamma=gamma)
        # self.scheduler = optim.lr_sheduler.CosineAnnealingLR(self.tf_optimizer, T_max)
        self.loss_test = LpLoss(size_average=False)

    def _loss_tf(self,out,x,u_bc):
            
        loss = TE_pde_loss(out,x,u_bc,self.dy,self.dz)
        return loss

    def _tf_train(self,train_loader,device):
        model = self.model
        tf_optimizer = self.tf_optimizer
        train_l2 = 0.0
        for x, _,u_bc in train_loader:
            x, u_bc = x.to(device), u_bc.to(device)
            tf_optimizer.zero_grad()
            out = model(self.x_normalizer.encode(x))
            # out = model(x)
            # out = self.y_normalizer.decode(out)
            loss = self._loss_tf(out,x,u_bc)
            loss.backward()
            tf_optimizer.step()
            train_l2 += loss.item()
        self.scheduler.step()

        return train_l2

    def _batch_test(self,test_loader,device,loss_func=None):
        model = self.model
        test_l2 = 0.0
        with torch.no_grad():
            for x, y ,sig,freq,u_bc in test_loader:
                batch_size = len(x)
                x, y,sig,freq,u_bc = x.to(device), y.numpy(), sig.numpy(),freq.numpy(),u_bc.to(device)
                out = model(self.x_normalizer.encode(x))
                # out = model(x)
                # out = self.y_normalizer.decode(out)
                impose_bc(out)
                ex1d0 = u_bc.unsqueeze(-1).repeat(1,1,len(self.dy)+1)
                ex1d  = torch.stack([ex1d0.real, ex1d0.imag],-1)
                out = out + ex1d
                rhoxy, phsxy = get_response(out.detach().cpu().numpy(),sig,self.dy.cpu().numpy(),\
                    self.dz.cpu().numpy(),self.ry,self.yn,self.nza,freq)
                d_rho, d_phs = rhoxy-y[...,0],phsxy-y[...,1]
                loss = np.mean((np.abs(d_rho)+np.abs(d_phs))**2)
                if loss_func is not None:
                    # relative error
                    MT_pred = np.stack((np.log10(rhoxy),phsxy),-1)
                    MT_true = np.stack((np.log10(y[...,0]),y[...,1]),-1)
                    loss = loss_func(MT_pred,MT_true)
                test_l2 += loss.item()
        return test_l2
    # save input

    

    def train(self,device,loss_func,tf_epochs,thre_epoch, patience, save_step,save_mode, \
                model_path,model_path_temp, log_file):
        '''train'''
        model = self.model
        val_l2 = np.inf
        stop_counter = 0
        temp_file = None
        for ep in range(tf_epochs):
            t1 = default_timer()
            model.train()
            train_l2 = self._tf_train(self.train_loader,device)
            # train_l2 = self._lf_train(fixed_latent)
            model.eval()
            test_l2 = self._batch_test(self.test_loader,device,loss_func)
            train_l2/= (self.n_train*self.n_freq_train)
            test_l2 /= (self.n_test * self.n_freq_test)

            # save model. Make sure that the model has been saved even if you stop the program manually. 
            # This is useful when you find that the training epoch is enough to stop the program
            if ep % save_step == 0:
                # Delete the previous saved model 
                if temp_file is not None:
                    os.remove(temp_file)
                # only save static dictionary instead of whole model
                torch.save(model.state_dict(), model_path_temp+'_epoch_'+str(ep)+'.pkl')
                temp_file = model_path_temp+'_epoch_'+str(ep)+'.pkl'

            # early stop
            if ep > thre_epoch:
                if test_l2 < val_l2:
                    # val_epoch = ep
                    val_l2 = test_l2
                    stop_counter = 0 
                    if save_mode == 'state_dict':
                        torch.save(model.state_dict(), model_path+'.pkl')
                    else: # save whole model, not recommended.
                        torch.save(model, model_path+'.pt')
                else:
                    stop_counter += 1
                # If the error continues to rise within 'patience' epochs, break
                if stop_counter > patience: 
                    print(f"Early stop at epoch {ep}")
                    print(f"# Early stop at epoch {ep}",file=log_file)
                    break

            t2 = default_timer()
            print(ep, t2-t1, train_l2, test_l2)
            print(ep, t2-t1, train_l2, test_l2,file=log_file)
def main(item):
    t0 = default_timer()

    with open( '../run/config.yml') as f:
        config = yaml.full_load(f)
    config = config[item]
    cuda_id = "cuda:"+str(config['cuda_id'])
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
    train_file = config['TRAIN_PATH']
    test_file  = config['TEST_PATH']
    f_idx       = config['f_idx']
    modes   = config['modes']
    # modes2   = config['modes2']
    width    = config['width']     
    layer_num= config['layer_num']
    last_size= config['last_size']
    act_fno  = config['act_fno']
    init_func = config['init_func']
    n_train   = config['n_train']
    n_test    = config['n_test']
    batch_size = config['batch_size']
    save_mode  = config['save_mode']
    save_step  = config['save_step']
    model_path = "../model/"+item # save path and name of model
    model_path_temp = "../temp/"+item
    log_path = "../Log/"+item+'.log'
    tf_epochs = config['tf_epochs']
    tf_lr     = config['tf_lr']
    weight_decay = config['weight_decay']
    step_size = config['step_size']
    gamma = config['gamma']
    patience = config['patience'] # if there is {patience} epoch that val_error is larger, early stop,
    thre_epoch = config['thre_epoch']# condiser early stop after {thre_epoch} epochs

    log_file = open(log_path,'a+')
    print("####################")
    print("begin to train model")

    loss_func = LpLoss_out(size_average=False)
    # loss_func = LpLoss(size_average=False)

    model = conv_TE_pde(train_file,test_file,n_train,n_test,batch_size,f_idx,device,\
                    modes, modes, width, layer_num, last_size, act_fno,init_func,\
                      tf_lr,weight_decay,step_size,gamma)
    model.train(device,loss_func,tf_epochs,thre_epoch, patience, save_step,save_mode, \
        model_path,model_path_temp,  log_file)

    tn = default_timer()
    print(f'all time:{tn-t0:.3f}s')
    print(f'# all time:{tn-t0:.3f}s',file=log_file)
    log_file.close()

if __name__ == "__main__":
    try:
        item = sys.argv[1]
    except: # usefule in vscode debug mode
        item = 'test_grid_1'
    main(item)


