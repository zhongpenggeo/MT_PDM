'''

LFBGS
'''
import os
import numpy as np
import cmath
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

# from untils.derivative import TE_derivative_complex, TE_derivative_complex_MF
sys.path.append("..") 
from utils.FNO import *
from utils.load_data import *
from utils.derivative import *

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


    #1.compute the system mat
    # 展平为2维方便矩阵计算
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
    return torch.mean(((torch.abs(f/MF*1e3))**2))


class conv_TM_pde(object):

    def __init__(self,train_file,test_file,n_train,n_test,batch_size,f_idx,device,\
                    modes1, modes2, width, layer_num, last_size, act_fno,init_func,\
                      tf_lr=1e-3,weight_decay=1e-4,step_size=50,gamma=0.5):
        # self.file_name = file_name
        self.II = cmath.sqrt(-1)
        # out is 2 channels
        zn,yn,dz,dy,ry,train_loader,test_loader,x_normalizer,y_normalizer\
             = get_loader(train_file,test_file,n_train,n_test,batch_size,f_idx)
        
        # self.ex = ex
        # self.nza = nza
        n_out = 2
        self.n_train = n_train
        self.n_test  = n_test
        self.n_freq_train = f_idx[1]
        self.n_freq_test  = f_idx[3]
        self.yn = yn
        self.zn = zn
        self.dz = dz.to(device)
        self.dy = dy.to(device)
        self.ry = ry
        # self.freq = freq
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        # self.u_bc_train = u_bc_train.to(device)
        # self.u_bc_test  = u_bc_test.to(device)

        
        self.model = FNO2d(modes1, modes2, width, n_out, layer_num, last_size, act_fno,init_func).to(device)
        self.tf_optimizer = optim.Adam(self.model.parameters(), lr=tf_lr,
                        weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.tf_optimizer, step_size=step_size, gamma=gamma)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.tf_optimizer, T_max)
        self.loss_test = LpLoss(size_average=False)
    
    def _loss_tf(self,out,x,u_bc_train,ey1d):
            
        loss = TM_pde_loss(out,x,u_bc_train,ey1d,self.dy,self.dz)
        return loss

    def _tf_train(self,train_loader,device):
        model = self.model
        tf_optimizer = self.tf_optimizer
        train_l2 = 0.0
        for x, _,u_bc,ey1d in train_loader:
            x, u_bc,ey1d = x.to(device), u_bc.to(device), ey1d.to(device)
            tf_optimizer.zero_grad()
            out = model(self.x_normalizer.encode(x))
            # out = self.y_normalizer.decode(out)
            loss = self._loss_tf(out,x,u_bc,ey1d)
            loss.backward()
            tf_optimizer.step()
            train_l2 += loss.item()
        self.scheduler.step()

        return train_l2

    def _batch_test(self,test_loader,device,loss_func=None):
        model = self.model
        test_l2 = 0.0
        with torch.no_grad():
            for x, y,sig,freq,u_bc in test_loader:
                x, y,sig,freq,u_bc = x.to(device), y.numpy(),sig.numpy(),freq.numpy(),u_bc.to(device)
                out = model(self.x_normalizer.encode(x))
                # out = self.y_normalizer.decode(out)
                impose_bc(out,u_bc)
                rhoyx, phsyx = get_response(out.detach().cpu().numpy(),sig[:,:-1,:-1],\
                    self.dy.cpu().numpy(),self.dz.cpu().numpy(),self.ry,self.yn,freq)
                # phsyx = phsyx - 180.0
                d_rho, d_phs = rhoyx-y[...,2],phsyx-y[...,3]
                loss = np.mean((np.abs(d_rho)+np.abs(d_phs))**2)
                if loss_func is not None:
                    # relative error
                    MT_pred = np.stack((np.log10(rhoyx),phsyx),-1)
                    MT_true = np.stack((np.log10(y[...,2]),y[...,3]),-1)
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

    def find_lr(self, device, init_value=1e-5, final_value=1., beta=0.98):
        # https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        num = len(self.train_loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.tf_optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for x, _,u_bc in self.train_loader:
            batch_num += 1
            #As before, get the loss for this mini-batch of inputs/outputs
            x,u_bc = x.to(device), u_bc.to(device)
            self.tf_optimizer.zero_grad()
            output = self.model(x)
            loss = self._loss_tf(output,x,u_bc)
            # loss = criterion(outputs, labels)
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) *loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(lr)
            #Do the SGD step
            loss.backward()
            self.tf_optimizer.step()
            #Update the lr for the next step
            lr *= mult
            self.tf_optimizer.param_groups[0]['lr'] = lr
        print('finished find lr')
        return log_lrs, losses   

def main(item):
    t0 = default_timer()

    with open( '../run/config_test.yml') as f:
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

    model = conv_TM_pde(train_file,test_file,n_train,n_test,batch_size,f_idx,device,\
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
        item = 'grid_1000'
    main(item)


