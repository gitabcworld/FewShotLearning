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
    opt['metaLearner'] = 'model.baselines.pixel-nearest-neighbor'

    opt['trainFull'] = True
    opt['nClasses.train'] =  64 - (-20) - (-16)
    opt['nAllClasses'] = 64 - (-4112)
    opt['useCUDA'] = False
    return opt

