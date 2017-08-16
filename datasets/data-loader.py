##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import importlib
import numpy as np

def getData(opt):
    # set up meta-train, meta-validation & meta-test datasets
    dataTrain = importlib.import_module(opt['dataName']).DatasetLoader(dataroot=opt['rawDataDir'],
                                                                       type='train',
                                                                       nEpisodes=opt['nEpisode'],
                                                                       classes_per_set=opt['nClasses']['train'],
                                                                       samples_per_class=opt['nTrainShot'])

    dataVal = importlib.import_module(opt['dataName']).DatasetLoader(dataroot=opt['rawDataDir'],
                                                                     type='val',
                                                                     nEpisodes=opt['nValidationEpisode'],
                                                                     classes_per_set=opt['nClasses']['val'],
                                                                     samples_per_class=opt['nEval'])
    dataTest = []
    for nTest in opt['nTest']:
        dataTest.append(importlib.import_module(opt['dataName']).DatasetLoader(dataroot=opt['rawDataDir'],
                                                                               type='test',
                                                                               nEpisodes=np.sum(opt['nTest']),
                                                                               classes_per_set=opt['nClasses']['test'],
                                                                               samples_per_class=np.max(
                                                                                   opt['nTestShot'])))
    data = {'train': dataTrain, 'validation': dataVal, 'test': dataTest}
    return data