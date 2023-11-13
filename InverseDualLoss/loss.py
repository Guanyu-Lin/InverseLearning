import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import pdb
def loss_function(y, t, vague, genuine, pos_lab, neg_lab, epoch): # loss_function(prediction, label, drop_rate_schedule(count))
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    genuine_loss = genuine * loss
    # print(genuine)
    # if epoch < 5:
    # return genuine_loss.mean([0])
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

    # pdb.set_trace()
    # print("pos", weightPos)
    # print("neg", weightNeg)

    dual_loss = (weightPos) * pos_loss + (weightNeg) * neg_loss
    # dual_loss = pos_loss + neg_loss
    noisy_loss = vague * dual_loss

    # loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    # genuine_loss = genuine * loss

    return  (genuine_loss + noisy_loss).mean([0])


def loss_function_true(y, t, vague, genuine, pos_lab, neg_lab): # loss_function(prediction, label, drop_rate_schedule(count))

    loss = F.binary_cross_entropy_with_logits(y, t)


    return  loss
