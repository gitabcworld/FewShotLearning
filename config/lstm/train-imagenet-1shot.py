##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def params(opt):
    opt['learner'] = 'model.lstm-classifier'
    opt['metaLearner'] = 'model.lstm.train-lstm'


    opt['BN_momentum'] = 0.9
    opt['optimMethod'] = 'adam'
    opt['lr'] = 1e-03
    opt['lr_decay'] = 1e-6
    opt['weight_decay'] = 1e-4
    opt['maxGradNorm'] = 0.25

    opt['batchSize'] = {1: 5, 5: 5}
    opt['nEpochs'] = {1: 12, 5: 5}

    opt['nEpisode'] = 7500
    opt['nValidationEpisode'] = 100
    opt['printPer'] = 1000
    opt['useCUDA'] = True
    return opt

