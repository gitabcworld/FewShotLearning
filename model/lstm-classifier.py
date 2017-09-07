##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math

def convLayer(opt, layer_pos, nInput, nOutput, k ):
    "3x3 convolution with padding"
    #if 'BN_momentum' in opt.keys():
    #    batchNorm = nn.BatchNorm2d(nOutput,momentum=opt['BN_momentum'])
    #else:
    #    batchNorm = nn.BatchNorm2d(nOutput)
        
    seq = nn.Sequential(
        nn.Conv2d(nInput, nOutput, kernel_size=k,
                  stride=1, padding=1, bias=True),
        #batchNorm,
        opt['bnorm2d'][layer_pos],
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    if opt['useDropout']: # Add dropout module
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)
    return seq

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()

        finalSize = int(math.floor(opt['nIn'] / (2 * 2 * 2 * 2)))

        self.layer1 = convLayer(opt, 0, opt['nDepth'], opt['nFilters'], 3)
        self.layer2 = convLayer(opt, 1, opt['nFilters'], opt['nFilters'], 3)
        self.layer3 = convLayer(opt, 2, opt['nFilters'], opt['nFilters'], 3)
        self.layer4 = convLayer(opt, 3, opt['nFilters'], opt['nFilters'], 3)

        self.outSize = opt['nFilters']*finalSize*finalSize
        self.classify = opt['classify']
        if self.classify:
            self.layer5 = nn.Linear(opt['nFilters']*finalSize*finalSize, opt['nClasses']['train'])
        self.outSize = opt['nClasses']['train']

        # Initialize layers
        self.reset()

    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def reset(self):
        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

    def forward(self, x):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if self.classify:
            x = self.layer5(x)
        return x


class MatchingNetClassifier():
    def __init__(self, opt):

        self.net = Classifier(opt)
        if opt['classify']:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = []
        self.nParams = sum([i.view(-1).size()[0] for i in self.net.parameters()])
        self.outSize = self.net.outSize

def build(opt):

    model = MatchingNetClassifier(opt)
    print('created net:')
    print(model.net)
    return model
