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
    opt['metaLearner'] = 'model.baselines.matching-net'

    # simple or FCE - embedding model?
    # opt['embedModel'] = 'model.baselines.simple-embedding'
    opt['embedModel'] = 'model.baselines.fce-embedding'

    opt['steps'] = 3
    opt['classify'] = False
    opt['useDropout'] = True
    opt['optimMethod'] = 'adam'
    opt['lr'] = 1e-03
    opt['lr_decay'] = 1e-6
    opt['weight_decay'] = 1e-4
    opt['batchSize'] = opt['nClasses']['train'] * opt['nEval']
    opt['nEpisode'] = 75000
    opt['nValidationEpisode'] = 100
    opt['printPer'] = 1000
    opt['useCUDA'] = True
    opt['ngpu'] = 2
    return opt
