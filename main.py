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
from option import Options
from logger import Logger
import numpy as np

# Import params from Config
# Parse other options
args = Options().parse()

# load config info for task, data, and model
opt = {}
opt = importlib.import_module(args.task).params(opt)
opt = importlib.import_module(args.data).params(opt)
opt = importlib.import_module(args.model).params(opt)
if not args.test == '-':
    opt = importlib.import_module(args.test).params(opt)
LOG_DIR = args.log_dir + '/task_{}_data_{}_model_{}' \
    .format(args.task,args.data,args.model)
# create logger
logger = Logger(LOG_DIR)

# Print options
print('Training with options:')
for key in sorted(opt.iterkeys()):
    print "%s: %s" % (key, opt[key])

# set up meta-train, meta-validation and meta-test datasets
data = importlib.import_module(opt['dataLoader']).getData(opt)
# Run the training, validation and test.
results = importlib.import_module(opt['metaLearner']).run(opt,data)
print('Task: %s. Data: %s. Model: %s' % (args.task,args.data,args.model) )


