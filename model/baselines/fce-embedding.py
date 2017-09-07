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
import numpy as np
#from model.lstm.bnlstm import RecurrentLSTMNetwork

class FceEmbedding():
    def __init__(self, opt):
        self.opt = opt # Store the parameters
        self.maxGradNorm = opt['maxGradNorm'] if ['maxGradNorm'] in opt.keys() else 0.25
        self.numLayersAttLstm = opt['numLayersAttLstm'] if ['numLayersAttLstm'] in opt.keys() else 1
        self.numLayersBiLstm = opt['numLayersBiLstm'] if ['numLayersBiLstm'] in opt.keys() else 1
        self.buildModels(self.opt)
        self.setCuda()

    # Build F and G models
    def buildModels(self,opt):
        # F function
        modelF = importlib.import_module(opt['learner']).build(opt)
        self.embedNetF = modelF.net
        # G function
        modelG = importlib.import_module(opt['learner']).build(opt)
        self.embedNetG = modelG.net

        '''
        # Build LSTM for attention model.
        self.attLSTM = RecurrentLSTMNetwork({
            'inputFeatures': self.embedNetF.outSize + self.embedNetG.outSize,
            'hiddenFeatures': self.embedNetF.outSize,
            'outputType': 'all'
        })

        self.biLSTMForward = RecurrentLSTMNetwork({
            'inputFeatures': self.embedNetG.outSize,
            'hiddenFeatures': self.embedNetG.outSize,
            'outputType': 'all'
        })

        self.biLSTMBackward = RecurrentLSTMNetwork({
            'inputFeatures': self.embedNetG.outSize,
            'hiddenFeatures': self.embedNetG.outSize,
            'outputType': 'all'
        })
        '''

        self.attLSTM = nn.LSTM(input_size=self.embedNetF.outSize + self.embedNetG.outSize,
                                hidden_size=self.embedNetF.outSize,
                                num_layers = self.numLayersAttLstm)
        # Build bidirectional LSTM
        self.biLSTM =  nn.LSTM(input_size=self.embedNetG.outSize,
                               hidden_size=self.embedNetG.outSize,
                               num_layers=self.numLayersBiLstm,
                               bidirectional=True)

        self.softmax = nn.Softmax()

    # Build list of parameters for optim
    def parameters(self):
        # TODO: why in the original code creates a dictionary with the same
        # parameters. model.params = {f=paramsG, g=paramsG, attLST, biLSTM}
        return list(self.embedNetG.parameters()) + \
                list(self.embedNetG.parameters()) + \
                    list(self.attLSTM.parameters()) + \
                        list(self.biLSTM.parameters())

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

    def attLSTM_forward(self,gS,fX, K):

        r = gS.mean(0).expand_as(fX)
        for i in np.arange(K):
            x = torch.cat((fX, r), 1)
            x = x.unsqueeze(0)
            if i == 0:
                #dim: [sequence = 1, batch_size, num_features * 2]
                output, (h, c) = self.attLSTM(x)
            else:
                output, (h, c) = self.attLSTM(x,(h,c))
            h = fX.squeeze(0) + output

            embed = None
            # Iterate over batch size
            for j in np.arange(h.size(1)):
                hInd = h[0,i, :].expand_as(gS)
                weight = (gS*hInd).sum(1).unsqueeze(1)
                embed_tmp = (self.softmax(weight).expand_as(gS) * gS).sum(0).unsqueeze(0)
                if embed is None:
                    embed = embed_tmp
                else:
                    embed = torch.cat([embed,embed_tmp],0)
        # output dim: [batch, num_features]
        return h.squeeze(0)

    def biLSTM_forward(self, input):
        gX = input
        # Expected input dimension of the form [sequence_length, batch_size, num_features]
        gX = gX.unsqueeze(1)
        output, (hn, cn) = self.biLSTM(gX)
        # output dim: [sequence, batch_size, num_features * 2]
        output = output[:, :, :self.embedNetG.outSize] + output[:, :, self.embedNetG.outSize:]
        output = output.squeeze(1)
        # output dim: [sequence, num_features]
        return output

    def embedG(self, input):
        g = self.embedNetG(input)
        return self.biLSTM_forward(g)

    def embedF(self, input, g, K):
        f = self.embedNetF(input)
        return self.attLSTM_forward(g,f,K)

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
            self.attLSTM.cuda()
            self.biLSTM.cuda()
        else:
            self.embedNetF.cpu()
            self.embedNetG.cpu()
            self.attLSTM.cpu()
            self.biLSTM.cpu()

def build(opt):
    model = FceEmbedding(opt)
    return model


