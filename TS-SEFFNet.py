# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:50:38 2021

@author: phantom

code for TS-SEFFNet in the paper 
"A Temporal-Spectral-based Squeeze-and-Excitation Feature Fusion Network for Motor Imagery EEG Decoding"
"""

import scipy.io as io
import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu

class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__
            + "("
            + "expression="
            + str(expression_str)
            + ")"
        )

def identity(x):
    """
    No activation function
    """
    return x

def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

def _transpose1(x):
    return x.permute(0, 3, 2, 1)

def _transpose2(x):
    return x.permute(0, 1, 3, 2)

def self_padding(x):
    """
    pariodic padding after the wavelet convolution, defined by formula (3) in the paper

    Parameters
    ----------
    x : input feature
    """
    return torch.cat((x[:, :, :, -3:], x, x[:, :, :, 0:3]), 3)

class SELayer(nn.Module):
    """
    the Squeeze and Excitation layer, defined by formula (4)(5) in the paper

    Parameters
    ----------
    channel: the input channel number
    reduction: the reduction ratio r
    """
    def __init__(self, channel, reduction = 8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace  = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class WaveletTransform(nn.Module): 
    """
    the wavelet convolution layer, defined by formula (1) in the paper

    Parameters
    ----------
    channel: the input channel number
    params_path: the path of the file saving db4 wavelet kernel
    """
    def __init__(self, channel, params_path='./TS-SEFFNet/scaling_filter.mat'):

        super(WaveletTransform, self).__init__()    
        self.conv = nn.Conv2d(in_channels = channel, out_channels = channel*2, kernel_size = (1, 8), stride = (1, 2), padding = 0, groups = channel, bias = False)        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = io.loadmat(params_path)
                Lo_D, Hi_D = np.flip(f['Lo_D'], axis = 1).astype('float32'), np.flip(f['Hi_D'], axis = 1).astype('float32')
                #(1,8) (1,8)
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis = 0)).unsqueeze(1).unsqueeze(1).repeat(channel, 1, 1, 1)    
                #([18, 1, 1, 8])
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        out = self.conv(self_padding(x)) 
        return out[:, 0::2,:, :], out[:, 1::2, :, :]

class TS_SEFFNet(nn.Module):
    """
    the model of the TS-SEFFNet proposed in the paper

    Parameters
    ----------
    in_chans: the input channel number
    n_classes: the output class number
    reduction_ratio: the reduction ratio r
    conv_stride: the convolution stride
    pool_stride: the pooling stride
    batch_norm: whether to use batch normalization after convolution, default True
    batch_norm_alpha: the momentum value in the nn.BatchNorm2d()
    drop_prob: the dropout probability
    
    """
    def __init__(self,in_chans=22,
                 n_classes=4,
                 reduction_ratio=8,
                 conv_stride=1,
                 pool_stride=3,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5,
                 ):
        super(TS_SEFFNet, self).__init__()
        
        self.in_chans=in_chans
        self.n_classes=n_classes
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.batch_norm=batch_norm
        self.batch_norm_alpha=batch_norm_alpha
        self.drop_prob=drop_prob
        self.reduction_ratio=reduction_ratio
        
        #the Spatio-Temporal Block
        self.transpose1=Expression(_transpose1)
        self.conv_time=nn.Conv2d(1,25,(11, 1),stride=1)
        self.conv_spatial=nn.Conv2d(25,25,(1, self.in_chans),stride=1,bias=not self.batch_norm)
        self.bn0=nn.BatchNorm2d(25, momentum=self.batch_norm_alpha, affine=True)
        self.conv_nonlinear=Expression(elu) #return x*x
        self.first_pool=nn.MaxPool2d(kernel_size=(3, 1),stride=(pool_stride, 1))
        self.pool_nonlinear=Expression(identity) #th.log(th.clamp(x, min=eps))
        
        #the 1-st Temporal Conv Unit
        self.drop1=nn.Dropout(p=self.drop_prob)
        self.conv1=nn.Conv2d(25,100,(11, 1),stride=(conv_stride, 1),bias=not self.batch_norm)
        self.bn1=nn.BatchNorm2d(100,momentum=self.batch_norm_alpha,affine=True,eps=1e-5)
        self.conv_nonlinear1=Expression(elu)
        self.pool1=nn.MaxPool2d(kernel_size=(3, 1),stride=(pool_stride, 1))
        self.pool_nonlinear1=Expression(identity)
        
        #the 2-nd Temporal Conv Unit
        self.drop2=nn.Dropout(p=self.drop_prob)
        self.conv2=nn.Conv2d(100,100,(11, 1),stride=(conv_stride, 1),bias=not self.batch_norm)
        self.bn2=nn.BatchNorm2d(100,momentum=self.batch_norm_alpha,affine=True,eps=1e-5)
        self.conv_nonlinear2=Expression(elu)
        self.pool2=nn.MaxPool2d(kernel_size=(3, 1),stride=(pool_stride, 1))
        self.pool_nonlinear2=Expression(identity)
        
        #the 3-rd Temporal Conv Unit
        self.drop3=nn.Dropout(p=self.drop_prob)
        self.conv3=nn.Conv2d(100,100,(11, 1),stride=(conv_stride, 1),bias=not self.batch_norm)
        self.bn3=nn.BatchNorm2d(100,momentum=self.batch_norm_alpha,affine=True,eps=1e-5)
        self.conv_nonlinear3=Expression(elu)
        self.pool3=nn.MaxPool2d(kernel_size=(3, 1),stride=(pool_stride, 1))
        self.pool_nonlinear3=Expression(identity)
        
        #the Multi-Spectral Convolution Block
        self.conv_spectral=nn.Conv2d(25,10,(1, 1),stride=1)
        self.WaveletTransform = WaveletTransform(channel = 10) #inc=30    
        self.transpose2=Expression(_transpose2)
        self.reshape1 = nn.AdaptiveAvgPool2d((1, 69))
        self.reshape2 = nn.AdaptiveAvgPool2d((1))
        
        #The SEC Unit for Multi-Spectral features
        self.SElayer1=SELayer(50,self.reduction_ratio)
        self.SEconv1 = nn.Conv2d(in_channels = 50, out_channels = 100, kernel_size = (1, 7), stride = (1, 1))
        self.SEbn1 = nn.BatchNorm2d(100)        
        self.SEpooling1 = nn.MaxPool2d(kernel_size = (1, 3),stride = (1, 3))
        
        #The SEC Unit for Deep-Temporal features
        self.SElayer2=SELayer(100,self.reduction_ratio)
        self.SEconv2 = nn.Conv2d(in_channels = 100, out_channels = 100, kernel_size = (1, 3), stride = (1, 1), padding = (0, 3//2), 
                                  groups = 1, bias = True)
        self.SEbn2 = nn.BatchNorm2d(100)       
        self.SEpooling2 = nn.MaxPool2d(kernel_size = (1, 3),stride = (1, 3))
        
        self.elu = nn.ELU(inplace = True)
        
        #the Classifier
        self.conv_classifier=nn.Conv2d(100,self.n_classes,(9, 1),bias=True)
        self.softmax=nn.LogSoftmax(dim=1)
        self.squeeze_output=Expression(_squeeze_final_output)
        
        #initializing parameters
        self.initialize()
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        #the Spatio-Temporal Block
        x=self.transpose1(x)
        x=self.conv_time(x)
        x=self.conv_spatial(x)
        x=self.bn0(x)
        x=self.conv_nonlinear(x)
        
        #the Multi-Spectral Convolution Block
        out=self.conv_spectral(x)
        out, gamma = self.WaveletTransform(self.transpose2(out))
        out, beta = self.WaveletTransform(out)
        out, alpha = self.WaveletTransform(out)
        delta,  theta= self.WaveletTransform(out)
        x_freq_feature=torch.cat((delta, theta, self.reshape1(alpha), self.reshape1(beta), self.reshape1(gamma)),1)
        
        #The SEC Unit for Multi-Spectral features
        x_freq_feature=self.SElayer1(x_freq_feature)
        x_freq_feature = self.elu(self.SEbn1(self.SEconv1(x_freq_feature)))
        x_freq_feature = self.SEpooling1(x_freq_feature)
        x_freq_feature=self.reshape2(x_freq_feature)
        
        #the Deep-Temporal Convolution Block
        x=self.first_pool(x)
        x=self.pool_nonlinear(x)
        #the 1-st Temporal Conv Unit
        x=self.conv_nonlinear1(self.bn1(self.conv1(self.drop1(x))))
        x=self.pool_nonlinear1(self.pool1(x))
        #the 2-nd Temporal Conv Unit
        x=self.conv_nonlinear2(self.bn2(self.conv2(self.drop2(x))))
        x=self.pool_nonlinear2(self.pool2(x))
        #the 3-rd Temporal Conv Unit
        x=self.conv_nonlinear3(self.bn3(self.conv3(self.drop3(x))))
        x=self.pool3(x)
        
        #The SEC Unit for Deep-Temporal features
        x=self.SElayer2(x)
        x = self.elu(self.SEbn2(self.SEconv2(x)))
        # x = self.SEpooling2(x)
        
        #the Classifier
        x=torch.cat((x,x_freq_feature),2)
        x=self.conv_classifier(x)
        x=self.softmax(x)
        x=self.squeeze_output(x)
        return x

if __name__ == "__main__":
    x=torch.rand(58, 22, 1125, 1).cuda()
    model=TS_SEFFNet(22,4).cuda()
    output=model(x)
    # print(output.shape)
