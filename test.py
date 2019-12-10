from __future__ import print_function

import tensorflow as tf
import torch
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Layer, Input, Lambda
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import tensorflow as tf
import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
vgg16 = models.vgg16()

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# a = tf.stack([ -1, 390, 30])
# print (a.shape)

# torch.stack([-1, 30,30])

# t = torch.tensor([[1,2],[3,4]])

# output = torch.gather(t, 0, torch.tensor([[0,0],[0,0]]))

# t = tf.Variable([[1,2],[3,4]])
# output = tf.gather(t, tf.Variable([[0,0],[1,0]]))

# x_f1st_pool = tf.gather( [[1,2],[3,4]], 1 )
# print (x_f1st_pool.shape)

# print (output)


class BnInception(nn.Module):
    def __init__(self):
        super(BnInception, self).__init__()
        self.conv_c0 = nn.Conv2d(256, 8, 1)
        self.conv_c1 = nn.Conv2d(256, 8, 3, padding=1)
        self.conv_c2 = nn.Conv2d(256, 8, 5, padding=2)

    def forward(self, x):
        x0 = self.conv_c0(x)
        print (1)
        print (x0.size())
        x1 = self.conv_c1(x)
        print (2)
        print (x1.size())
        x2 = self.conv_c2(x)
        print (3)
        print (x2.size())

        x = torch.cat((x0,x1,x2), 1)
        print (4)
        print (x.size())


if __name__ == "__main__":
    thisNet = BnInception()
    # print (thisNet)
    summary(thisNet, input_size=(256, 16, 16))
    # for n, p in thisNet.named_parameters():
    #     print(n, p.shape)