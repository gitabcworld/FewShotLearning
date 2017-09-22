import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
import numpy as np

class Learner(nn.Module):
    def __init__(self, opt):
        super(Learner, self).__init__()

        # Note: we are using two networks to simulate learner where one network
        # is used for backward pass on test set and the other is used simply to get
        # gradients which serve as input to meta-learner.
        # This is a simple way to make computation graph work
        # so that it doesn't include gradients of learner

        # Create another network with only shared 'running_mean' and 'running_var'
        # this weights can be found in BatchNormalization layers (or InstanceNormalization)
        # In torch with the instruction: model.net:clone('running_mean', 'running_var')
        # it is already done but with Pytorch we need to copy those parameters with
        # state_dict and load_state_dict every time we want to use one of the shared
        # networks.

        # Add dimension filters for the cnn
        opt['nFilters'] = 32
        # Create 4 layers with batch norm. Share layers between self.model and self.modelF
        self.bn_layers = []
        for i in range(4):
            if 'BN_momentum' in opt.keys():
                self.bn_layers.append(nn.BatchNorm2d(opt['nFilters'],
                                                momentum=opt['BN_momentum']))
            else:
                self.bn_layers.append(nn.BatchNorm2d(opt['nFilters']))
        opt['bnorm2d'] = self.bn_layers

        # local embedding model
        self.model = importlib.import_module(opt['learner']).build(opt)
        self.modelF = importlib.import_module(opt['learner']).build(opt)
        self.nParams = self.modelF.nParams
        self.params = {param[0]: param[1] for param in self.modelF.net.named_parameters()}

    def unflattenParams_net(self,flatParams):
        flatParams = flatParams.squeeze()
        indx = 0
        for param in self.model.net.parameters():
            lengthParam = param.view(-1).size()[0]
            param = flatParams[indx:lengthParam].view_as(param).clone()

    def forward(self, inputs, targets ):

        output = self.modelF.net(inputs)
        loss = self.modelF.criterion(output, targets)
        return output, loss

    def feval(self, inputs, targets):
        # reset gradients
        self.model.net.zero_grad()
        # evaluate function for complete mini batch
        outputs = self.model.net(inputs)
        loss = self.model.criterion(outputs, targets)
        loss.backward()
        grads = torch.cat([param.grad.view(-1) for param in self.model.net.parameters()], 0)
        return grads,loss

    def reset(self):
        self.model.net.reset()
        self.modelF.net.reset()

    # Set training or evaluation mode
    def set(self,mode):
        if mode == 'training':
            self.model.net.train()
            self.modelF.net.train()
        elif mode == 'evaluate':
            self.model.net.eval()
            self.modelF.net.eval()
        else:
            print('model.set: undefined mode - %s' % (mode))

    def setCuda(self, value = True):
        # If value is a string then use self.opt
        # If it is not a string then it should be True or False
        if value == True:
            self.model.net.cuda()
            self.modelF.net.cuda()
        else:
            self.model.net.cpu()
            self.modelF.net.cpu()