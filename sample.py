import torch.nn as nn
import torch.nn.functional as F
import torch

# class Conv2dSame(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
#         super().__init__()
#         ka = kernel_size // 2
#         kb = ka - 1 if kernel_size % 2 == 0 else ka
#         self.net = torch.nn.Sequential(
#             padding_layer((ka,kb,ka,kb)),
#             torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
#         )
#     def forward(self, x):
#         return self.net(x)


# net = Conv2dSame(3, 64, 3)
# print (net)



#calculate same convolution

# o = output
# p = padding
# k = kernel_size
# s = stride
# d = dilation

i = 16
o = 16
# p = 0
k = 3
s = 1
d = 1

p = ((s * (o-1)) -i +(d*(k-1)) + 1)/2

print (o)
print (p)   