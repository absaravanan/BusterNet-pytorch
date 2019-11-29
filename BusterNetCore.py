"""
This file defines all BusterNet related custom layers
"""


import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
vgg16 = models.vgg16()

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super(Conv2dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

# def std_norm_along_chs(x) :
#     '''Data normalization along the channle axis
#     Input:
#         x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
#     Output:
#         xn = tensor4d, same shape as x, normalized version of x
#     '''
#     avg = K.mean(x, axis=-1, keepdims=True)
#     std = K.maximum(1e-4, K.std(x, axis=-1, keepdims=True))
#     return (x - avg) / std

# def std_norm_along_chs(x):
#     '''
#     Applies the Sigmoid Linear Unit (SiLU) function element-wise:
#         SiLU(x) = x * sigmoid(x)
#     '''

#     avg = torch.mean(x, dim=-1, keepdims=True )
#     std = torch.max()
#     return input * torch.sigmoid(input)



# class SiLU(nn.Module):
#     '''
#     Applies the Sigmoid Linear Unit (SiLU) function element-wise:
#         SiLU(x) = x * sigmoid(x)
#     Shape:
#         - Input: (N, *) where * means, any number of additional
#           dimensions
#         - Output: (N, *), same shape as the input
#     References:
#         -  Related paper:
#         https://arxiv.org/pdf/1606.08415.pdf
#     Examples:
#         >>> m = silu()
#         >>> input = torch.randn(2)
#         >>> output = m(input)
#     '''
#     def __init__(self):
#         '''
#         Init method.
#         '''
#         super().__init__() # init the base class

#     def forward(self, input):
#         '''
#         Forward pass of the function.
#         '''
#         return std_norm_along_chs(input) # simply apply already implemented SiLU

class SelfCorrelationPercPooling(nn.Module):
    def __init__(self,  nb_pools=256, **kwargs):
        self.nb_pools = nb_pools
        super(SelfCorrelationPercPooling, self).__init__()

    def forward( self, x, mask=None ) :
        # parse input feature shape
        bsize, nb_rows, nb_cols, nb_feats = list(x.size())
        nb_maps = nb_rows * nb_cols
        # self correlation
        x_3d = x.view( torch.stack( [ -1, nb_maps, nb_feats ] ) )
        x_corr_3d = torch.matmul( x_3d, torch.t(x_3d) ) / nb_feats
        x_corr = x_corr_3d.view(torch.stack( [ -1, nb_rows, nb_cols, nb_maps ] ) )
        # argsort response maps along the translaton dimension
        if ( self.nb_pools is not None ) :        
            ranks =  torch.round(torch.linspace( 1., nb_maps - 1, self.nb_pools ) )
            ranks = ranks.type(torch.int32)
        else :
            ranks = torch.arange( 1, nb_maps)       
        x_sort, _ = torch.topk( x_corr, nb_maps, sorted = True )

        x_f1st_sort = x_sort.permute( 3, 0, 1, 2 )
        x_f1st_pool = torch.gather( x_f1st_sort, 1 ,ranks)
        x_pool = x_f1st_pool.permute( 1, 2, 3, 0 )
        return x_pool

    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = self.nb_pools if ( self.nb_pools is not None ) else ( nb_rows * nb_cols - 1 )
        return tuple( [ bsize, nb_rows, nb_cols, nb_pools ] )



class BusterNet(torch.nn.Module):
    '''Create the similarity branch for copy-move forgery detection
    '''
    def __init__(self):
        super(BusterNet, self).__init__()
        self.conv1 = Conv2dSame(3, 64, 3)
        self.conv2 = Conv2dSame(64, 128, 3)
        self.conv3 = Conv2dSame(128, 256, 3)
        self.conv4 = Conv2dSame(256, 256, 3)

        self.pool = nn.MaxPool2d(2, 2)

        # self.layerNorm = nn.LayerNorm(-1)

        # self.instSelfCorrelationPercPooling = SelfCorrelationPercPooling()

        # self.batchNorm1 = nn.BatchNorm2d()

    def forward(self, x):
        x = self.conv1(x)
        print (1)
        print (x.size())
        x = self.conv1(x)
        print (2)
        print (x.size())
        x = self.pool(x)

        x = self.conv2(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.pool(x)

        return x


if __name__ == "__main__":
    thisNet = BusterNet()
    # print (thisNet)
    summary(thisNet, input_size=(3, 256, 256))
    # for n, p in thisNet.named_parameters():
    #     print(n, p.shape)

