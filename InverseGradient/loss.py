import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import pdb
def loss_function_train(y, t, pos_lab, neg_lab): # loss_function(prediction, label, drop_rate_schedule(count))
    pos_loss_norm = F.binary_cross_entropy_with_logits(y, pos_lab, reduce = False).detach()
    pos_loss = F.binary_cross_entropy_with_logits(y, pos_lab, reduce = False)
    neg_loss_norm = F.binary_cross_entropy_with_logits(y, neg_lab, reduce = False).detach()
    neg_loss = F.binary_cross_entropy_with_logits(y, neg_lab, reduce = False)
    alpha = torch.tensor(1e-9)
    weightPos = ((neg_loss_norm / (pos_loss_norm + alpha)))
    weightNeg = ((pos_loss_norm / (neg_loss_norm + alpha)))
    weightSum = weightPos + weightNeg
    weightPos = weightPos / weightSum
    weightNeg = weightNeg / weightSum
    dual_loss = (weightPos) * pos_loss + (weightNeg) * neg_loss
    return  (dual_loss).mean([0]), weightPos, weightNeg

def loss_function_test(y, t): # loss_function(prediction, label, drop_rate_schedule(count))
    loss = F.binary_cross_entropy_with_logits(y, t)


    return  loss

def loss_function_true(y, t, vague, genuine, pos_lab, neg_lab): # loss_function(prediction, label, drop_rate_schedule(count))

    loss = F.binary_cross_entropy_with_logits(y, t)


    return  loss
