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
    opt['nEpisode'] = 0
    opt['paramsFile'] = 'saved_params/matching-net-FCE/matching-net_params_snapshot.th'
    opt['networkFile'] = 'saved_params/matching-net-FCE/matching-net-models.th'
    return opt
