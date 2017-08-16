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
    opt['metaLearner'] = 'model.baselines.pre-trained-SGD'


    opt['trainFull'] = True
    opt['nClasses.train'] = 64

    opt['learningRate'] = 0.001
    opt['trainBatchSize'] = 64

    opt['learningRates'] = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    opt['learningRateDecays'] = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]
    opt['nUpdates'] = [15]

    opt['nEpochs'] = 30000
    opt['nValidationEpisode'] = 100
    opt['printPer'] = 1000
    opt['useCUDA'] = True
    return opt

