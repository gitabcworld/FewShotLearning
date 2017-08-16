##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torch
import torch.nn as nn
import importlib
import pickle
from itertools import chain

class SimpleEmbedding():
    def __init__(self, opt):
        self.opt = opt # Store the parameters
        self.buildModels(self.opt)
        self.setCuda()

    # Build F and G models
    def buildModels(self,opt):
        model1 = importlib.import_module(opt['learner']).build(opt)
        self.embedNet1 = model1.net # F function
        model2 = importlib.import_module(opt['learner']).build(opt)
        self.embedNet2 = model2.net # G function

    # Build list of parameters for optim
    def parameters(self):
        # TODO: why in the original code creates a dictionary with the same
        # parameters. model.params = {f=paramsG, g=paramsG}
        return list(self.embedNet2.parameters()) + list(self.embedNet2.parameters())

    # Set training or evaluation mode
    def set(self,mode):
        if mode == 'training':
            self.embedNet1.train()
            self.embedNet2.train()
        elif mode == 'evaluate':
            self.embedNet1.eval()
            self.embedNet2.eval()
        else:
            print('model.set: undefined mode - %s' % (mode))

    def isTraining(self):
        return self.embedNet1.training

    def default(self, dfDefault):
        self.df = dfDefault

    def embedX(self, input, gS = [], K = []):
        return self.embedNet1(input)

    def embedS(self, input):
        return self.embedNet2(input)

    def save(self, path = './data'):
        # Save the opt parameters
        optParametersFile = open(os.path.join(path,'SimpleEmbedding_opt.pkl'), 'wb')
        pickle.dump(self.opt, optParametersFile)
        optParametersFile.close()
        # Clean not needed data of the models
        self.embedNet1.clearState()
        self.embedNet2.clearState()
        torch.save(self.embedNet1.state_dict(), os.path.join(path,'embedNet1.pth.tar'))
        torch.save(self.embedNet2.state_dict(), os.path.join(path, 'embedNet2.pth.tar'))

    def load(self, pathParams, pathModelF, pathModelG):
        # Load opt parameters 'SimpleEmbedding_opt.pkl'
        optParametersFile = open(pathParams, 'rb')
        self.opt = pickle.load(optParametersFile)
        optParametersFile.close()
        # build the models
        self.buildModels(self.opt)
        # Load the weights and biases of F and G
        checkpoint = torch.load(pathModelF)
        self.embedNet1.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(pathModelG)
        self.embedNet2.load_state_dict(checkpoint['state_dict'])
        # Set cuda
        self.setCuda()

    def setCuda(self, value = 'default'):
        # If value is a string then use self.opt
        # If it is not a string then it should be True or False
        if type(value) == str:
            value = self.opt['useCUDA']
        else:
            assert(type(value)==bool)

        if value == True:
            print('Check CUDA')
            self.embedNet1.cuda()
            self.embedNet2.cuda()
        else:
            self.embedNet1.cpu()
            self.embedNet2.cpu()

def build(opt):
    model = SimpleEmbedding(opt)
    return model


