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

class SimpleEmbedding():
    def __init__(self, opt):
        self.opt = opt # Store the parameters
        self.buildModels(self.opt)
        self.setCuda()

    # Build F and G models
    def buildModels(self,opt):
        modelF = importlib.import_module(opt['learner']).build(opt)
        self.embedNetF = modelF.net # F function
        modelG = importlib.import_module(opt['learner']).build(opt)
        self.embedNetG = modelG.net # G function

    # Build list of parameters for optim
    def parameters(self):
        # TODO: why in the original code creates a dictionary with the same
        # parameters. model.params = {f=paramsG, g=paramsG}
        return list(self.embedNetG.parameters()) + list(self.embedNetG.parameters())

    # Set training or evaluation mode
    def set(self,mode):
        if mode == 'training':
            self.embedNetF.train()
            self.embedNetG.train()
        elif mode == 'evaluate':
            self.embedNetF.eval()
            self.embedNetG.eval()
        else:
            print('model.set: undefined mode - %s' % (mode))

    def isTraining(self):
        return self.embedNetF.training

    def default(self, dfDefault):
        self.df = dfDefault

    def embedF(self, input, g = [], K = []):
        return self.embedNetF(input)

    def embedG(self, input):
        return self.embedNetG(input)

    def save(self, path = './data'):
        # Save the opt parameters
        optParametersFile = open(os.path.join(path,'SimpleEmbedding_opt.pkl'), 'wb')
        pickle.dump(self.opt, optParametersFile)
        optParametersFile.close()
        # Clean not needed data of the models
        self.embedNetF.clearState()
        self.embedNetG.clearState()
        torch.save(self.embedNetF.state_dict(), os.path.join(path,'embedNetF.pth.tar'))
        torch.save(self.embedNetG.state_dict(), os.path.join(path, 'embedNetG.pth.tar'))

    def load(self, pathParams, pathModelF, pathModelG):
        # Load opt parameters 'SimpleEmbedding_opt.pkl'
        optParametersFile = open(pathParams, 'rb')
        self.opt = pickle.load(optParametersFile)
        optParametersFile.close()
        # build the models
        self.buildModels(self.opt)
        # Load the weights and biases of F and G
        checkpoint = torch.load(pathModelF)
        self.embedNetF.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(pathModelG)
        self.embedNetG.load_state_dict(checkpoint['state_dict'])
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
            self.embedNetF.cuda()
            self.embedNetG.cuda()
        else:
            self.embedNetF.cpu()
            self.embedNetG.cpu()

def build(opt):
    model = SimpleEmbedding(opt)
    return model


