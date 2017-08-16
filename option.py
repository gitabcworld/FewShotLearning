##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Few-Shot Learning')
        parser.add_argument('--task', type=str, default='config.5-shot-5-class',
                            help='path to config file for task')
        parser.add_argument('--data', type=str, default='config.imagenet',
                            help='path to config file for data')
        parser.add_argument('--model', type=str, default='config.baselines.train-matching-net',
                            help='path to config file for model')
        parser.add_argument('--test', type=str, default='-',
                            help='path to config file for test details')
        parser.add_argument('--log-dir', default='./logs',
                            help='folder to output model checkpoints')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
