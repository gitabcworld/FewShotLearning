import torch
from torch import nn
from torch.autograd import Variable

P = torch.FloatTensor(1).fill_(10)
expP = torch.exp(P)
negExpP = torch.exp(-P)

def preProc1(x):
    # Access the global variables
    global P,expP,negExpP
    P = P.type_as(x.data)
    expP = expP.type_as(x.data)
    negExpP = negExpP.type_as(x.data)

    # Create a variable filled with -1. Second part of the condition
    z = torch.zeros(x.size()).fill_(-1).type_as(x.data)
    absX = torch.abs(x.data)
    cond1 = torch.gt(absX, negExpP)
    if torch.sum(cond1) > 0:
        x1 = torch.log(torch.abs(x[cond1])).data/P
        z[cond1] = x1
    return z

def preProc2(x):
    # Access the global variables
    global P, expP, negExpP
    P = P.type_as(x.data)
    expP = expP.type_as(x.data)
    negExpP = negExpP.type_as(x.data)

    # Create a variable filled with -1. Second part of the condition
    z = torch.zeros(x.size()).type_as(x.data)
    absX = torch.abs(x.data)
    cond1 = torch.gt(absX, negExpP)
    cond2 = torch.le(absX, negExpP)
    if torch.sum(cond1) > 0:
        x1 = torch.sign(x[cond1].data)
        z[cond1] = x1
    if torch.sum(cond2) > 0:
        x2 = x[cond2].data*expP
        z[cond2] = x2
    return z

def preprocess(grad,loss):

    preGrad = Variable(grad.data.new(grad.data.size()[0], 1, 2).zero_())
    preGrad[:, :, 0] = preProc1(grad)
    preGrad[:, :, 1] = preProc2(grad)

    lossT = Variable(loss.data.new(1,1,1).zero_())
    lossT[0] = loss
    preLoss = Variable(loss.data.new(1,1,2).zero_())
    preLoss[:, :, 0] = preProc1(lossT)
    preLoss[:, :, 1] = preProc2(lossT)
    return preGrad,preLoss
