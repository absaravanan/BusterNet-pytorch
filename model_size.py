import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.build_BiSeNet import BiSeNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""


model = BiSeNet(2, "resnet18")
model = torch.nn.DataParallel(model)

# load pretrained model if exists
model.module.load_state_dict(torch.load("checkpoints_18_sgd/latest_dice_loss22.pth", map_location=torch.device('cpu')))
model.eval()


# Estimate Size
from pytorch_modelsize import SizeEstimator

se = SizeEstimator(model, input_size=(1,1,960,1280))
print(se.estimate_size())

# Returns
# (size in megabytes, size in bits)
# (408.2833251953125, 3424928768)

print(se.param_bits) # bits taken up by parameters
print(se.forward_backward_bits) # bits stored for forward and backward
print(se.input_bits) # bits for input