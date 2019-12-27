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


def std_norm_along_chs(x):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''

    avg = torch.mean(x, dim=-1, keepdims=True )
    std = torch.max()
    return input * torch.sigmoid(input)


class SelfCorrelationPercPooling(nn.Module):
    def __init__(self,  nb_pools=256, **kwargs):
        self.nb_pools = nb_pools
        super(SelfCorrelationPercPooling, self).__init__()

        self.dummpPool = nn.Conv2d(512, 256, 1) 

    def forward( self, x, mask=None ) :
        # parse input feature shape
        bsize, nb_feats, nb_rows, nb_cols = list(x.size())
        nb_maps = nb_rows * nb_cols
        # print ("nbmaps=",nb_maps)
        # print ("nbfeats=",nb_feats)
        # self correlation
        # stacked = torch.stack( [ -1, nb_maps, nb_feats ] )
        x_3d = x.view([-1, nb_maps, nb_feats ])
        # print ("x_shape", x.size())

        # print ("x_3d=",x_3d.size())

        t_x_3d = x_3d.permute(0,2,1)

        # print ("x_3d=",x_3d.size())

        x_corr_3d = torch.matmul( x_3d, t_x_3d ) / nb_feats
        
        x_corr = x_corr_3d.view( [ -1, nb_rows, nb_cols, nb_maps ])
        # argsort response maps along the translaton dimension

        if ( self.nb_pools is not None ) :        
            ranks =  torch.round(torch.linspace( 1., nb_maps - 1, self.nb_pools ) )
            ranks = ranks.long()
        else :
            ranks = torch.arange( 1, nb_maps)       
        
        x_sort, _ = torch.topk( x_corr, nb_maps, sorted = True )

        # print ("x_sort=", x_sort.size())
        x_f1st_sort = x_sort.permute( 3, 0, 1, 2 )
        # print ("x_f1st_sort=", x_f1st_sort.size())
        # print ("ranks=", ranks.size())
        # print (ranks)

        # ranks= ranks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print (ranks)
        # ranks = ranks.view(-1,1)
        # print ("ranks1=", ranks.size())
        # print (ranks)


        # x_f1st_pool = torch.gather( x_f1st_sort, 0 , ranks)
        x_pool = x_f1st_sort[ranks]
        x_pool = x_pool.permute( 1, 0, 2, 3 )
        # print ("x_pool=", ranks.size())
        # print ("x_pooled=", x_pool.size())

        # x = self.dummpPool(x)

        return x_pool

    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = self.nb_pools if ( self.nb_pools is not None ) else ( nb_rows * nb_cols - 1 )
        return tuple([ bsize, nb_rows, nb_cols, nb_pools ])


class BnInception(nn.Module):
    def __init__(self,  input_dims, src_output_dims, patch_list=[1,3,5], padd =[0,1,2]):
        super(BnInception, self).__init__()
        # print (src_output_dims)
        self.conv_c0 = nn.Conv2d(input_dims, src_output_dims, patch_list[0], padding=padd[0])
        self.conv_c1 = nn.Conv2d(input_dims, src_output_dims, patch_list[1], padding=padd[1])
        self.conv_c2 = nn.Conv2d(input_dims, src_output_dims, patch_list[2], padding=padd[2])
        self.batchNorm1 = nn.BatchNorm2d(int(src_output_dims*3))
        self.relu = nn.ReLU()


    def forward(self, x):
        x0 = self.conv_c0(x)
        # print (1)
        # print (x0.size())
        x1 = self.conv_c1(x)
        # print (2)
        # print (x1.size())
        x2 = self.conv_c2(x)
        # print (3)
        # print (x2.size())
        x = torch.cat((x0,x1,x2), 1)
        # print (4)
        # print (x.size())
        x = self.batchNorm1(x)
        x = self.relu(x)

        return x


class SimiNet(torch.nn.Module):
    '''Create the similarity branch for copy-move forgery detection
    '''
    def __init__(self):
        super(SimiNet, self).__init__()
        # self.conv1 = Conv2dSame(3, 64, 3)
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3c = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4c = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu =  nn.ReLU()

        self.layerNorm = nn.LayerNorm([16,16])

        self.instSelfCorrelationPercPooling = SelfCorrelationPercPooling()

        self.batchNorm1 = nn.BatchNorm2d(256)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.bnInception1 = BnInception(256, 8)
        self.bnInception2 = BnInception(24, 6)
        self.bnInception3 = BnInception(42, 4)
        self.bnInception4 = BnInception(36,2)
        self.bnInception5 = BnInception(30,2)

        self.bnInception6 = BnInception(30,2, patch_list=[5,7,11], padd=[2,3,5])
        self.conv2d_mask = nn.Conv2d(6,2,3, padding=1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        x = self.conv1a(x)
        x = self.relu(x)
        # print (1)
        # print (x.size())
        x = self.conv1b(x)
        x = self.relu(x)
        # print (2)
        # print (x.size())
        x = self.pool(x)
        # print (3)
        # print (x.size())
        x = self.conv2a(x)
        x = self.relu(x)
        # print (4)
        # print (x.size())        
        x = self.conv2b(x)
        x = self.relu(x)
        # print (5)
        # print (x.size()) 
        x = self.pool(x)
        # print (6)
        # print (x.size()) 
        x = self.conv3a(x)
        x = self.relu(x)
        # print (6)
        # print (x.size()) 

        x = self.conv3b(x)
        x = self.relu(x)
        # print (7)
        # print (x.size()) 
        x = self.conv3c(x)
        x = self.relu(x)
        # print (8)
        # print (x.size()) 
        x = self.pool(x)
        # print (9)
        # print (x.size()) 

        x = self.conv4a(x)
        x = self.relu(x)
        # print (10)
        # print (x.size()) 
        x = self.conv4b(x)
        x = self.relu(x)
        # print (11)
        # print (x.size()) 
        x = self.conv4c(x)
        x = self.relu(x)
        # print (12)
        # print (x.size()) 
        x = self.pool(x)
        # print (13)
        # print (x.size()) 

        x = self.layerNorm(x)
        # print (14)
        # print (x.size()) 
        x = self.instSelfCorrelationPercPooling(x)
        # print (15)
        # print (x.size())

        x_batchNorm1 = self.batchNorm1(x)

        f16 = self.bnInception1(x_batchNorm1)

        f32 = self.upsample1(f16)
        dx32 = self.bnInception2(f32)

        f64a = self.upsample1(f32)
        f64b = self.upsample1(dx32)
        # print (f64a.size())
        # print (f64b.size())

        f64 = torch.cat((f64a, f64b),1)
        # print (f64.size())
        dx64 = self.bnInception3(f64)
        # print (dx64.size())

        f128a = self.upsample1(f64a)
        f128b = self.upsample1(dx64)

        f128 = torch.cat((f128a, f128b),1)
        # print (f128.size())
        dx128 = self.bnInception4(f128)
        # print (dx128.size())


        f256a = self.upsample1(f128a)
        f256b = self.upsample1(dx128)
        f256 = torch.cat((f256a, f256b),1)
        # print (f256.size())
        dx256 = self.bnInception5(f256)
        # print (dx256.size())

        fm256 = torch.cat((f256a, dx256),1)
        # print (fm256.size())

        # print ("________________________________")

        masks = self.bnInception6(fm256)
        # print (masks.size())

        pred_mask = self.conv2d_mask(masks)
        # pred_mask =  self.sigmoid(pred_mask)
        # print (pred_mask.size())

        return pred_mask

class ManiNet(torch.nn.Module):
    '''Create the Manipulation branch for copy-move forgery detection
    '''
    def __init__(self):
        super(ManiNet, self).__init__()
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3c = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4c = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu =  nn.ReLU()

        self.layerNorm = nn.LayerNorm([16,16])

        self.instSelfCorrelationPercPooling = SelfCorrelationPercPooling()

        self.batchNorm1 = nn.BatchNorm2d(256)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.bnInception1 = BnInception(512, 8)
        self.bnInception2 = BnInception(24, 6)
        self.bnInception3 = BnInception(18, 4)
        self.bnInception4 = BnInception(12,2)
        self.bnInception5 = BnInception(6,2)

        self.bnInception6 = BnInception(30,2, patch_list=[5,7,11], padd=[2,3,5])
        self.conv2d_mask = nn.Conv2d(6,2,3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1a(x)
        x = self.relu(x)
        # print (1)
        # print (x.size())
        x = self.conv1b(x)
        x = self.relu(x)
        # print (2)
        # print (x.size())
        x = self.pool(x)
        # print (3)
        # print (x.size())
        x = self.conv2a(x)
        x = self.relu(x)
        # print (4)
        # print (x.size())        
        x = self.conv2b(x)
        x = self.relu(x)
        # print (5)
        # print (x.size()) 
        x = self.pool(x)
        # print (6)
        # print (x.size()) 
        x = self.conv3a(x)
        x = self.relu(x)
        # print (6)
        # print (x.size()) 

        x = self.conv3b(x)
        x = self.relu(x)
        # print (7)
        # print (x.size()) 
        x = self.conv3c(x)
        x = self.relu(x)
        # print (8)
        # print (x.size()) 
        x = self.pool(x)
        # print (9)
        # print (x.size()) 

        x = self.conv4a(x)
        x = self.relu(x)
        # print (10)
        # print (x.size())
        x = self.conv4b(x)
        x = self.relu(x)
        # print (11)
        # print (x.size()) 
        x = self.conv4c(x)
        x = self.relu(x)
        # print (12)
        # print (x.size()) 
        x = self.pool(x)
        # print (13)
        # print (x.size())


        f16 = self.bnInception1(x)
        # print (14)
        # print (f16.size())

        f32 = self.upsample1(f16)
        # print (15)
        # print (f32.size())

        dx32 = self.bnInception2(f32)
        # print (dx32.size())

        f64 = self.upsample1(dx32)
        # print (16)
        # print (f64.size())

        dx64 = self.bnInception3(f64)
        # print (dx64.size())

        f128 = self.upsample1(dx64)
        # print (17)

        # print (f128.size())
        dx128 = self.bnInception4(f128)
        # print (dx128.size())


        f256 = self.upsample1(dx128)
        # print (18)

        # print (f256.size())
        dx256 = self.bnInception5(f256)
        # print (dx256.size())

        pred_mask = self.conv2d_mask(dx256)
        # print (pred_mask.size())

        # pred_mask =  self.sigmoid(pred_mask)
        # print (pred_mask.size())

        return pred_mask


class ManiBottom(nn.Module):
    def __init__(self, original_model):
        super(ManiBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        # print (self.features)
        
    def forward(self, x):
        x = self.features(x)
        # print (x)
        return x


class BiseNet(nn.Module):
    '''Create the Bisnet branch for copy-move forgery detection
    '''
    def __init__(self):
        super(BiseNet, self).__init__()
        self.BiseManinet = ManiBottom(ManiNet())
        # self.siminet = SimiBottom(SimiNet())
        # self.bnInception1 = BnInception(256, 12)

    def forward(self, x):
        
        mani_x = self.BiseManinet(x)

        # print ('mani')
        # print (mani_x.size())
        # simi_x = self.siminet(x)['BnInception-71']
        # print ('semi')
        # print (simi_x.size())



        # x = torch.cat((simi_x, mani_x), 1)
        # print ('cat')
        # print (x.size())

        # x = self.bnInception1(x)
        # print ('inception')
        # print (x.size())
        return mani_x



if __name__ == "__main__":
    # org_model = ManiNet()
    # thisNet = ManiNet()
    thisNet = SimiNet()
    # print (thisNet)
    summary(thisNet, input_size=(3, 256, 256))
    # for n, p in thisNet.named_parameters():
        # print(n, p.shape)