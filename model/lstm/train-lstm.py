##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
import numpy as np
import time
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from utils import util
from learner import Learner
from metaLearner import MetaLearner

def create_optimizer(opt, params):
    if opt['optimMethod'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opt['lr'],
                              momentum=0.9, dampening=0.9,
                              weight_decay=opt['weight_decay'])
    elif opt['optimMethod']:
        optimizer = torch.optim.Adam(params, lr=opt['lr'],
                               weight_decay=opt['weight_decay'])
    else:
        raise Exception('Not supported optimizer: {0}'.format(opt['optimMethod']))
    return optimizer

def adjust_learning_rate(opt,optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = opt['lr'] / (1 + group['step'] * opt['lr_decay'])

    return optimizer

def run(opt,data):

    # learner
    learner = Learner(opt)
    print('Learner nParams: %d' % (learner.nParams))

    # meta-learner
    params_dict = {'learnerParams': learner.params,
                   'nParams': learner.nParams}
    for param in ['debug','homePath','nHidden','BN1','BN2']:
        if param in opt.keys():
            params_dict[param] = opt[param]
    metaLearner = MetaLearner(params_dict)
    # set cuda
    metaLearner.setCuda(opt['useCUDA'])
    learner.setCuda(opt['useCUDA'])

    # Keep track of errors
    trainConf_pred = []
    trainConf_gt = []
    valConf_pred = {}
    valConf_gt = {}
    testConf_pred = {}
    testConf_gt = {}
    for i in opt['nTestShot']:
        valConf_pred[i] = []
        valConf_gt[i] = []
        testConf_pred[i] = []
        testConf_gt[i] = []

    cost = 0
    timer = time.time()

    #################################################################
    ############ Meta-training
    #################################################################

    # Init optimizer
    #optimizer = create_optimizer(opt, metaLearner.params.values())
    #optimizer = create_optimizer(opt, learner.modelF.net.parameters())
    optimizer = create_optimizer(opt, list(metaLearner.lstm.parameters()) + list(metaLearner.lstm2.parameters()))

    # train episode loop
    for episodeTrain,(x_support_set, y_support_set, x_target, target_y) in enumerate(data['train']):

        # Re-arange the Target vectors between [0..nClasses_train]
        dictLabels, dictLabelsInverse = util.createDictLabels(y_support_set)
        y_support_set = util.fitLabelsToRange(dictLabels, y_support_set)
        target_y = util.fitLabelsToRange(dictLabels, target_y)

        # Convert them in Variables
        input = {}
        trainInput = Variable(x_support_set).float()
        trainTarget = Variable(y_support_set,requires_grad=False).long()
        testInput = Variable(x_target).float()
        testTarget = Variable(target_y,requires_grad=False).long()

        # Convert to GPU if needed
        if opt['useCUDA']:
            trainInput = trainInput.cuda()
            trainTarget = trainTarget.cuda()
            testInput = testInput.cuda()
            testTarget = testTarget.cuda()

        # learner-optimizer with learner.model.net


        # forward metalearner
        output, loss = metaLearner(learner, trainInput, trainTarget,
                                            testInput, testTarget,
                                            opt['nEpochs'][opt['nTrainShot']],
                                            opt['batchSize'][opt['nTrainShot']])
        optimizer.zero_grad()
        loss.backward()
        metaLearner.gradNorm(loss)
        optimizer.step()

        # Adjust learning rate
        optimizer = adjust_learning_rate(opt, optimizer)

        cost = cost + loss

        # update stats
        values_pred, indices_pred = torch.max(output, 1)
        target_y = util.fitLabelsToRange(dictLabelsInverse, target_y)
        indices_pred = util.fitLabelsToRange(dictLabelsInverse, indices_pred.cpu().data)
        trainConf_pred.append(indices_pred.numpy())
        trainConf_gt.append(target_y.numpy())

        print(
            'Training Episode: [{}/{} ({:.0f}%)]\tLoss: {:.3f}. Elapsed: {:.4f} s'.format(
                episodeTrain, len(data['train']), 100. * episodeTrain / len(data['train']),
                loss.data[0], time.time() - timer))


        if episodeTrain % opt['printPer'] == 0:
            trainConf_pred = np.concatenate(trainConf_pred, axis=0)
            trainConf_gt = np.concatenate(trainConf_gt, axis=0)
            target_names = [str(i) for i in np.unique(trainConf_gt)]

            print(
                'Training Episode: [{}/{} ({:.0f}%)]\tCost: {:.3f}. Elapsed: {:.4f} s'.format(
                    episodeTrain, len(data['train']), 100. * episodeTrain / len(data['train']),
                    (cost.cpu().data.numpy() / opt['printPer'])[0],time.time() - timer))
            print(classification_report(trainConf_gt, trainConf_pred,
                                        target_names=target_names))
            # Set to 0
            trainConf_pred = []
            trainConf_gt = []

        cost = 0
        timer = time.time()

