import torch
from torch import nn
from torch.autograd import Variable

P = Variable(torch.FloatTensor(1).fill_(10))
expP = Variable(torch.exp(P.data))
negExpP = Variable(torch.exp(-P.data))

def preProc1(x):
    # Access the global variables
    global P,expP,negExpP
    P = P.type_as(x)
    expP = expP.type_as(x)
    negExpP = negExpP.type_as(x)

    # Create a variable filled with -1. Second part of the condition
    z = Variable(torch.zeros(x.size()).fill_(-1)).type_as(x)
    absX = torch.abs(x)
    cond1 = torch.gt(absX, negExpP)
    if (torch.sum(cond1) > 0).data.all():
        x1 = torch.log(torch.abs(x[cond1]))/P
        z[cond1] = x1
    return z

def preProc2(x):
    # Access the global variables
    global P, expP, negExpP
    P = P.type_as(x)
    expP = expP.type_as(x)
    negExpP = negExpP.type_as(x)

    # Create a variable filled with -1. Second part of the condition
    z = Variable(torch.zeros(x.size())).type_as(x)
    absX = torch.abs(x)
    cond1 = torch.gt(absX, negExpP)
    cond2 = torch.le(absX, negExpP)
    if (torch.sum(cond1) > 0).data.all():
        x1 = torch.sign(x[cond1])
        z[cond1] = x1
    if (torch.sum(cond2) > 0).data.all():
        x2 = x[cond2]*expP
        z[cond2] = x2
    return z

def preprocess(grad,loss):

    #preGrad = Variable(grad.data.new(grad.data.size()[0], 1, 2).zero_())
    #preGrad = grad.expand(grad.data.size()[0], 1, 2)
    preGrad = grad.clone().expand(grad.data.size()[0], 1, 2)
    preGrad[:, :, 0] = preProc1(grad)
    preGrad[:, :, 1] = preProc2(grad)

    #lossT = Variable(loss.data.new(1,1,1).zero_())
    #lossT[0] = loss
    #preLoss = Variable(loss.data.new(1,1,2).zero_())
    #preLoss = loss.expand(1, 1, 2)
    preLoss = loss.clone().expand(1, 1, 2)
    preLoss[:, :, 0] = preProc1(loss)
    preLoss[:, :, 1] = preProc2(loss)
    return preGrad,preLoss
