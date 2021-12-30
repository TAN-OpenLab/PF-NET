import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
# from matplotlib.pyplot import MultipleLocator
# import matplotlib.pyplot as plt
from torch.autograd import Variable
import math

def ERBSpace(lowFreq = 100.0, highFreq = 44100.0/4.0, N=100.0):
    earQ = 9.26449
    minBW = 24.7
    low = float(lowFreq)
    high = float(highFreq)
    N=float(N)
    cf = -(earQ * minBW) + np.exp((np.arange(N+1)[1:]) * (-np.log(high + earQ * minBW) + np.log(low + earQ * minBW)) / (N)) * (high + earQ * minBW)
    cf=cf.astype(np.float32)
    return cf

def GT(f, t, o = 0):
    n = 2
    b = 24.7+0.108 * f
    a=(1.019*b)**n
    Gamma = (a @ (t**(n - 1))) * torch.cos(2 * math.pi * f @ t + o)/torch.exp(2 * math.pi * b @ t)
    return Gamma

def flip(x, dim):  #反转
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dims
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def linear(f1, f2, f3, singal):
    out1_left = f2 * torch.cos(f2 @ singal) - torch.sin(f2 @ singal)/singal - f1 * torch.sin(f2 @ singal)/singal
    out1_right = f1 * torch.cos(f1 @ singal) - torch.sin(f1 @ singal)/singal - f1 * torch.sin(f1 @ singal)/singal
    out2_left = f3 * torch.cos(f3 @ singal) - torch.sin(f3 @ singal)/singal - f3 * torch.sin(f3 @ singal)/singal
    out2_right = f2 * torch.cos(f2 @ singal) - torch.sin(f2 @ singal)/singal - f3 * torch.sin(f2 @ singal)/singal
    return (out1_left - out1_right)/(f2 - f1) + (out2_left - out2_right)/(f2 - f3)

def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)),y_right])  #拼接

    return y

def linold(f1, f2, h1, h2, signal):
    # if f1 == f2:
    #    return 0
    delta1 = (h2 - h1)/(f2 - f1)
    delta2 = h1 - (f1 + 1) * delta1
    out2 = delta1 * f2 * torch.cos(f2 @ signal) + delta2 * torch.sin(f2 @ signal)/signal
    out1 = delta1 * f1 * torch.cos(f1 @ signal) + delta2 * torch.sin(f1 @ signal)/signal
    out = out2 - out1
    return out

def lin(f1, f2, h1, h2, signal):
    # if f1 == f2:
    #    return 0
    delta = (h2 - h1)/(f2 - f1)
    # delta2 = h1 - (f1 + 1) * delta1
    out2 = 1 * delta * torch.cos(f2 @ signal)/signal/signal + 1 * h2 * torch.sin(f2 @ signal)/signal
    out1 = 1 * delta * torch.cos(f1 @ signal)/signal/signal + 1 * h1 * torch.sin(f1 @ signal)/signal

    # out2 =4* delta * torch.cos(f2 @ signal)/signal/signal
    # out1 =4* delta * torch.cos(f1 @ signal)/signal/signal

    out = out2 - out1
    return out
def lin2(f1, f2, h1, h2, signal):
    # if f1 == f2:
    #    return 0
    delta = (h2 - h1)/(f2 - f1)
    # delta2 = h1 - (f1 + 1) * delta1
    out2 = 1 * delta * torch.cos(f2 * signal)/signal/signal + 1 * h2 * torch.sin(f2 * signal)/signal
    out1 = 1 * delta * torch.cos(f1 * signal)/signal/signal + 1 * h1 * torch.sin(f1 * signal)/signal

    # out2 =4* delta * torch.cos(f2 @ signal)/signal/signal
    # out1 =4* delta * torch.cos(f1 @ signal)/signal/signal

    out = out2 - out1
    return out

def linspace(tensor, tensor2, steps=2):
    out = torch.zeros(steps)
    out[0] = tensor
    count = torch.zeros(1)
    band = (tensor2- tensor)/(steps-1)
    for i in range(steps-1):
        out[i+1] = count + band
    return out








class pfConv_fast(nn.Module):


    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50, breakpoints=5):

        super(pfConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'pfConv only support one inpzz channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "pfConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)


        self.breakpoints = breakpoints
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:      #奇数核
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('pfConv does not support bias.')
        if groups > 1:
            raise ValueError('pfConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)



        self.low_hz_ = torch.Tensor(hz[:-1]).view(-1, 1)

        self.band_hz_ = torch.Tensor(np.diff(hz)).view(-1, 1)

        self.low = self.min_low_hz + torch.abs(self.low_hz_)

        self.high = torch.clamp(self.low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)  #拉直，能保证全部取到

        self.band = (self.high-self.low)[:,0]

        self.points = nn.Parameter((torch.rand(out_channels, breakpoints)-0.5))
        # self.points = nn.Parameter((torch.rand(out_channels, breakpoints) - 0.5)*5)

        #self.pmin = torch.zeros(out_channels).view(-1,1).cuda()
        #self.pmax = torch.zeros(out_channels).view(-1,1).cuda()


        self.hz = torch.Tensor([])
        # self.band_0 = self.band_hz_.view(-1)
        # self.band_0 = torch.cat([self.band_0[1:],torch.zeros(1)],dim=0)
        # print(self.band_0)


        for i in range(out_channels):
            self.hz = torch.cat([self.hz,
                                 self.to_hz(torch.linspace(self.to_mel(self.low[i, 0] - self.min_low_hz), self.to_mel(self.high[i, 0] - self.min_low_hz - self.min_band_hz), steps=self.breakpoints).view(1, -1))
                                + torch.linspace(self.min_low_hz,self.min_low_hz+self.min_band_hz, steps=self.breakpoints).view(1, -1)]
                               , dim=0)


        # self.min = self.hz[:,0].view(-1,1).cuda()
        # self.max = self.hz[:,-1].view(-1,1).cuda()

        self.hz = nn.Parameter(self.hz)
        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # 需要考虑pi的问题




    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)     #在waveform的设备上运行

        self.window_ = self.window_.to(waveforms.device)

        self.hz = self.hz.to(waveforms.device)

        band_pass_left = torch.zeros(self.out_channels, int(self.kernel_size/2)).to(waveforms.device)
        #for i in range(80):
            #if self.hz[i,0]>self.hz[i,1]:
        #print(self.points[0,:],self.points[0,1],self.points[0,0])
        #print("done")


       # hz = self.hz*(self.max-self.min)+self.min

        # points = torch.sigmoid(self.points-2.5)+torch.sigmoid(self.points+2.5)
        # max,dim =torch.max(torch.abs(self.points),dim=1)
        #min,dim =torch.min(self.points,dim=1)
        # value = max
        # print(self.points[0])
        points = torch.abs(0.5*self.points+1.)

        # print(self.points[23])
        # self.band = (self.hz[:,-1]-self.hz[:,0])*(self.hz[:,-1]+self.hz[:,0])
        # print(self.points)



        self.band = torch.zeros(1)
        self.band =self.band.to(waveforms.device)
        for j in range(self.breakpoints - 1):
            # if (j==0):
            #
            #     band_pass_left = band_pass_left +lin(self.min.view(-1,1), self.hz[:,j+1].view(-1,1),
            #                                          points[:,j].view(-1,1), points[:,j+1].view(-1,1), self.n_)
            #     self.band = self.band + 0.5 * (points[:, j + 1].view(-1, 1) + points[:, j].view(-1, 1)) * (
            #                 self.hz[:, j + 1].view(-1, 1) - self.min.view(-1,1))
            #     break
            # if (self.breakpoints - 2):
            #     band_pass_left = band_pass_left +lin(self.hz[:,j].view(-1,1), self.max.view(-1,1),
            #                                          points[:,j].view(-1,1), points[:,j+1].view(-1,1), self.n_)
            #     self.band = self.band + (0.5*(self.max.view(-1,1)*points[:,j].view(-1,1) - self.hz[:,j].view(-1,1)*points[:,j+1].view(-1,1)) +
            #                   0.5*(self.max.view(-1,1)*points[:,j+1].view(-1,1) - self.hz[:,j].view(-1,1)*points[:,j].view(-1,1)))
            #     break

            band_pass_left = band_pass_left +lin(self.hz[:,j].view(-1,1), self.hz[:,j+1].view(-1,1),
                                                 points[:,j].view(-1,1), points[:,j+1].view(-1,1), self.n_)
            # self.band = self.band + (0.5*(self.hz[:,j+1].view(-1,1)*points[:,j].view(-1,1) - self.hz[:,j].view(-1,1)*points[:,j+1].view(-1,1)) +
            #               0.5*(self.hz[:,j+1].view(-1,1)*points[:,j+1].view(-1,1) - self.hz[:,j].view(-1,1)*points[:,j].view(-1,1)))
            self.band = self.band + 0.5*(points[:,j+1].view(-1,1)+points[:,j].view(-1,1))*(self.hz[:,j+1].view(-1,1)-self.hz[:,j].view(-1,1))
        #print(self.band[66])

        #print(band_pass_left[70,120:123])
        band_pass_left=band_pass_left*self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = self.band.view(-1,1)
        band_pass_right = torch.flip(band_pass_left,dims=[1])


        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        #print(band_pass.size())
        #print(band_pass[7,16]/self.band[7,0])

        band_pass = band_pass / (self.band.view(-1,1))
        # draw_fre(torch.sum(band_pass, dim=0), 251)
        # maxh,dimh =torch.max(torch.abs(self.h),dim=1)
        #
        #
        # poh = 0.0001*self.h/maxh.view(-1,1)+1.
        #
        # #torch.set_printoptions(precision=7)
        # # print(band_pass[0,125:127])
        # #print(band_pass[0,0:10])
        #
        # #print(band_pass[66,120:130])
        #
        # band_pass = band_pass*poh
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)




        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


class pfNet(nn.Module):  # 导入继承nn.module方法

    def __init__(self, options):
        super(pfNet, self).__init__()  # 继承其属性

        self.cnn_N_filt = options['cnn_N_filt']
        self.cnn_len_filt = options['cnn_len_filt']
        self.cnn_max_pool_len = options['cnn_max_pool_len']

        self.cnn_act = options['cnn_act']
        self.cnn_drop = options['cnn_drop']

        self.cnn_use_laynorm = options['cnn_use_laynorm']
        self.cnn_use_batchnorm = options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp = options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp = options['cnn_use_batchnorm_inp']

        self.input_dim = int(options['input_dim'])  # 16000*200/1000.0，采样率，16khz，每秒从连续信号中提取并组成离散信号的采样个数

        self.fs = options['fs']

        self.N_cnn_lay = len(options['cnn_N_filt'])
        self.conv = nn.ModuleList([])  # 写一个module然后就写foreword函数很麻烦
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)  # 标准化层，中心极限定理

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)  # batch标准化，不使用

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):  # 三个滤波层

            N_filt = int(self.cnn_N_filt[i])  # 80，60，60滤波器个数
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))  # 0.0

            # activation
            self.act.append(act_fun(self.cnn_act[i]))  # leaky_relu（0.2）

            # layer norm initialization         
            self.ln.append(LayerNorm(
                [N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))  # 最后为最大值池化长度

            self.bn.append(
                nn.BatchNorm1d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]),
                               momentum=0.05))  # 最后滑动参数，为测试而准备

            if i == 0:
                self.conv.append(pfConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))

            else:
                self.conv.append(
                    nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))  # 输入维度，输出维度，滤波长度

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)  # reshape

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = self.drop[i](
                        self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))
                else:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
            # print(x.size())

        x = x.view(batch, -1)
        # print(0)
        # print(x.size())
        return x

def act_fun(act_type):

 if act_type=="relu":
    return nn.ReLU()
            
 if act_type=="tanh":
    return nn.Tanh()
            
 if act_type=="sigmoid":
    return nn.Sigmoid()
           
 if act_type=="leaky_relu":
    return nn.LeakyReLU(0.2)
            
 if act_type=="elu":
    return nn.ELU()
                     
 if act_type=="softmax":
    return nn.LogSoftmax(dim=1)
        
 if act_type=="linear":
    return nn.LeakyReLU(1) # initializzed like this, but not used in forward!

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()

        self.input_dim = int(options['input_dim'])
        self.fc_lay = options['fc_lay']
        self.fc_drop = options['fc_drop']
        self.fc_use_batchnorm = options['fc_use_batchnorm']
        self.fc_use_laynorm = options['fc_use_laynorm']
        self.fc_use_laynorm_inp = options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp = options['fc_use_batchnorm_inp']
        self.fc_act = options['fc_act']

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.fc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.fc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        self.N_fc_lay = len(self.fc_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_fc_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.fc_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))

            if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.fc_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.fc_lay[i], current_input).uniform_(-np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                                                                     np.sqrt(0.01 / (current_input + self.fc_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.fc_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.fc_use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_fc_lay):

            if self.fc_act[i] != 'linear':

                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                    x = self.drop[i](self.act[i](self.wx[i](x)))

            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                    x = self.drop[i](self.wx[i](x))

        return x




