'''
fork from https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/fno_zongyi_2d.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# activattion type
act_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "idt": nn.Identity(),
    "gelu": nn.GELU()
}

# initiation method
init_dict={
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "uniform": nn.init.uniform_,
    "norm": nn.init.normal_
}

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2,init_func):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.   

        Parameters:
        -----------
        in_channels  : lifted dimension 
        out_channels : output dimension 
        modes1       : truncated modes in the first dimension of fourier domain 
        modes2       : truncated modes in the second dimension of fourier domain
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.linear = nn.Linear(in_channels,out_channels)
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_channels, out_channels, modes1, modes2, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)

        if init_func in init_dict.keys():
            print("init_func:", init_func)
            init_method = init_dict[init_func]
            for param in self.fourier_weight:
                init_method(param, gain=1/(in_channels * out_channels))
        else:
            print("init_func: kaiming normal")
            for param in self.fourier_weight:
                # nn.init.uniform_(param)
                nn.init.kaiming_normal_(param)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # res = self.linear(x)
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        # the result in last dimension is half for fft
        # and the result in  the second to last dimension is symmetric.
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], torch.complex(self.fourier_weight[0][...,0], self.fourier_weight[0][...,1]))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],torch.complex(self.fourier_weight[1][...,0], self.fourier_weight[1][...,1])) # because of  symmetry?

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        # if self.residual:
        #     x = self.act(x + res)

        return x

# branch net
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width,n_out,layer_num=4,last_size=128,act_func="gelu",init_func='xavier_normal',
                set_bn=False,residual=False):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        Parameters:
        -----------
            - modes1    : truncated modes in the first dimension of fourier domain
            - modes2    : truncated modes in the second dimension of fourier domain
            - width     : width of the lifted dimension
            - n_out     : output dimension, here is 4: rhoxy, phasexy, rhoyx, phaseyx
            - layer_num : number of fourier layers
            - last_size : width of projected dimension
            - act_func  : activation function, key must in act_dict
        """
        self.set_bn = set_bn
        self.residual = residual
        self.padding = 9 # pad the domain if input is non-periodic
        self.layer_num = layer_num
        if act_func in act_dict.keys():
            self.activation = act_dict[act_func]
        else:
            raise KeyError("act name not in act_dict")
        self.fc0 = nn.Linear(1, width) # input channel is 3: (a(x, y), x, y)

        self.fno = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.bn  = nn.ModuleList()
        for _ in range(layer_num):
            self.fno.append(SpectralConv2d(width, width, modes1, modes2,init_func))
            self.conv.append(nn.Conv2d(width, width, 1))
            if set_bn:
                self.bn.append(nn.BatchNorm2d(width))

        self.fc1 = nn.Linear(width, last_size)
        # notice!
        self.fc2 = nn.Linear(last_size, n_out)

    def forward(self, x):
        '''
        input  : (batch, x, y, 1)
        output : (batch, x, y, n_out)
        '''
        # lift to high dimension
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)# batch_size, width, n1,n2
        x = F.pad(x, [0,self.padding, 0,self.padding])# pad for last 2 dimensions (n1,n2)
        # number of fourier layers
        for i in range(self.layer_num):
            res = x
            x1 = self.fno[i](x)
            x2 = self.conv[i](x)
            x  = x1 + x2
            x = self.activation(x)
            x = res + x if self.residual else x
            x = self.bn[i](x) if self.set_bn else x


        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        # 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x
