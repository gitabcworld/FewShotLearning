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
    opt['learner'] = 'model.matching-net-classifier'
    opt['metaLearner'] = 'model.baselines.conv-nearest-neighbor'

    opt['trainFull'] = True
    opt['nClasses.train'] = 64
    opt['learningRate'] = 0.001
    opt['trainBatchSize'] = 64
    opt['nEpochs'] = 30000
    opt['nValidationEpisode'] = 100
    opt['printPer'] = 1000
    opt['useCUDA'] = True
    return opt

