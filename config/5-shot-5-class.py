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
    opt['nClasses'] = {'train':5, 'val':5, 'test':5}
    opt['nTrainShot'] = 5
    opt['nEval'] = 15

    opt['nTest'] = [100, 250, 600]
    opt['nTestShot'] = [1, 5]
    return opt