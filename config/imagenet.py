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
    opt['nExamples'] = 20
    opt['nDepth'] = 3
    opt['nIn'] = 84

    opt['rawDataDir'] = '/home/aberenguel/Dataset/miniImagenet'
    opt['dataName'] = 'datasets.miniImagenet'
    opt['dataLoader'] = 'datasets.data-loader'
    opt['episodeSamplerKind'] = 'permutation'

    return opt

