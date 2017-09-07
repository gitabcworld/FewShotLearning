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

class MatchingNet(nn.Module):
    def __init__(self, opt):
        super(MatchingNet, self).__init__()

        # function cosine-similarity layer
        self.cosineSim = nn.CosineSimilarity()

        # local embedding model (simple or FCE)
        self.embedModel = importlib.import_module(opt['embedModel']).build(opt)
        # set Cuda
        self.embedModel.setCuda(opt['useCUDA'])

        # load loss. Why does not load with the model
        self.lossF = nn.CrossEntropyLoss()

    # Set training or evaluation mode
    def set(self, mode):
        self.embedModel.set(mode)

    def forward(self, opt, input ):

        trainInput = input['trainInput']
        trainTarget = input['trainTarget']
        testInput = input['testInput']
        testTarget = input['testTarget']

        # Create one-hot vector
        trainTarget = trainTarget.view(-1,1)
        y_one_hot = trainTarget.clone()
        y_one_hot = y_one_hot.expand(
            trainTarget.size()[0], opt['nClasses']['train'])
        y_one_hot.data.zero_()
        y_one_hot = y_one_hot.float().scatter_(1, trainTarget, 1)

        # embed support set & test items using g and f respectively
        gS = self.embedModel.embedG(trainInput)
        fX = self.embedModel.embedF(testInput, gS, opt['steps'])

        # repeat tensors so that can get cosine sims in one call
        repeatgS = gS.repeat(fX.size(0),1)
        repeatfX = fX.repeat(1, gS.size(0)).view(fX.size(0)*gS.size(0),fX.size(1))

        # weights are num_test x num_train (weigths per test item)
        weights = self.cosineSim(repeatgS, repeatfX).view(fX.size(0), gS.size(0),1)

        # one-hot matrix of train labels is expanded to num_train x num_test x num_labels
        expandOneHot = y_one_hot.view(1,y_one_hot.size(0),y_one_hot.size(1)).expand(
            fX.size(0),y_one_hot.size(0),y_one_hot.size(1))

        # weights are expanded to match one-hot matrix
        expandWeights = weights.expand_as(expandOneHot)

        # cmul one-hot matrix by weights and sum along rows to get weight per label
        # final size: num_train x num_labels
        out = expandOneHot.mul(expandWeights).sum(1)

        # calculate NLL
        if self.embedModel.isTraining():
            loss = self.lossF(out,testTarget)
            return out, loss
        else:
            return out

def create_optimizer(opt, model):
    if opt['optimMethod'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt['lr'],
                              momentum=0.9, dampening=0.9,
                              weight_decay=opt['weight_decay'])
    elif opt['optimMethod']:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'],
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

    # Set the model
    network = MatchingNet(opt)

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

    # load params from file
    # paramsFile format: <path>/SimpleEmbedding_opt.pkl
    # pathModelF format: <path>/embedNet_F.pth.tar
    # pathModelF format: <path>/embedNet_G.pth.tar
    if np.all([key in opt.keys() for key in ['paramsFile','pathModelF','pathModelG']]):

        if (os.path.isfile(opt['paramsFile']) and \
             os.path.isfile(opt['pathModelF']) and \
             os.path.isfile(opt['pathModelG'])):
            print('loading from params: %s' % (opt['paramsFile']))
            print('loading model F: %s' % (opt['pathModelF']))
            print('loading model G: %s' % (opt['pathModelG']))
            network.embedModel.load(opt['paramsFile'],
                                    opt['pathModelF'],
                                    opt['pathModelG'])

    cost = 0
    timer = time.time()

    #################################################################
    ############ Meta-training
    #################################################################

    # Init optimizer
    optimizer = create_optimizer(opt, network.embedModel)

    # set net for training
    network.set('training')

    # train episode loop
    for episodeTrain,(x_support_set, y_support_set, x_target, target_y) in enumerate(data['train']):

        # Re-arange the Target vectors between [0..nClasses_train]
        dictLabels, dictLabelsInverse = util.createDictLabels(y_support_set)
        y_support_set = util.fitLabelsToRange(dictLabels, y_support_set)
        target_y = util.fitLabelsToRange(dictLabels, target_y)

        # Convert them in Variables
        input = {}
        input['trainInput'] = Variable(x_support_set).float()
        input['trainTarget'] = Variable(y_support_set,requires_grad=False).long()
        input['testInput'] = Variable(x_target).float()
        input['testTarget'] = Variable(target_y,requires_grad=False).long()

        # Convert to GPU if needed
        if opt['useCUDA']:
            input['trainInput'] = input['trainInput'].cuda()
            input['trainTarget'] = input['trainTarget'].cuda()
            input['testInput'] = input['testInput'].cuda()
            input['testTarget'] = input['testTarget'].cuda()

        output, loss = network(opt,input)
        optimizer.zero_grad()
        loss.backward()
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

        if episodeTrain % opt['printPer'] == 0:
            trainConf_pred = np.concatenate(trainConf_pred, axis=0)
            trainConf_gt = np.concatenate(trainConf_gt, axis=0)
            target_names = [str(i) for i in np.unique(trainConf_gt)]
            print(
                'Training Episode: [{}/{} ({:.0f}%)]\tLoss: {:.3f}. Elapsed: {:.4f} s'.format(
                    episodeTrain, len(data['train']), 100. * episodeTrain / len(data['train']),
                    (cost.cpu().data.numpy() / opt['printPer'])[0],time.time() - timer))
            print(classification_report(trainConf_gt, trainConf_pred,
                                        target_names=target_names))
            # Set to 0
            trainConf_pred = []
            trainConf_gt = []

            #################################################################
            ############ Meta-evaluation
            #################################################################

            timerEval = time.time()

            # evaluate validation set
            network.set('evaluate')
            # validation episode loop
            for episodeValidation, (x_support_set, y_support_set, x_target, target_y) in enumerate(data['validation']):

                # Re-arange the Target vectors between [0..nClasses_train]
                dictLabels, dictLabelsInverse = util.createDictLabels(y_support_set)
                y_support_set = util.fitLabelsToRange(dictLabels, y_support_set)
                target_y = util.fitLabelsToRange(dictLabels, target_y)
                unique_labels = dictLabels.keys()

                # k-shot loop
                for k in opt['nTestShot']:

                    # Select k samples from each class from x_support_set and
                    indexes_selected = []
                    for k_selected in unique_labels:
                        selected = np.random.choice(np.squeeze(np.where(y_support_set.numpy() == dictLabels[k_selected]))
                                         ,k, False)
                        indexes_selected.append(selected)

                    # Select the k-shot examples from the Tensors
                    x_support_set_k = x_support_set[torch.from_numpy(np.squeeze(indexes_selected).flatten())]
                    y_support_set_k = y_support_set[torch.from_numpy(np.squeeze(indexes_selected).flatten())]

                    # Convert them in Variables
                    input = {}
                    input['trainInput'] = Variable(x_support_set_k).float()
                    input['trainTarget'] = Variable(y_support_set_k, requires_grad=False).long()
                    input['testInput'] = Variable(x_target).float()
                    input['testTarget'] = Variable(target_y, requires_grad=False).long()

                    # Convert to GPU if needed
                    if opt['useCUDA']:
                        input['trainInput'] = input['trainInput'].cuda()
                        input['trainTarget'] = input['trainTarget'].cuda()
                        input['testInput'] = input['testInput'].cuda()
                        input['testTarget'] = input['testTarget'].cuda()

                    output = network(opt, input)

                    # update stats validation
                    values_pred, indices_pred = torch.max(output, 1)
                    target_y = util.fitLabelsToRange(dictLabelsInverse, target_y)
                    indices_pred = util.fitLabelsToRange(dictLabelsInverse, indices_pred.cpu().data)
                    valConf_pred[k].append(indices_pred.numpy())
                    valConf_gt[k].append(target_y.numpy())

            for k in opt['nTestShot']:
                valConf_pred[k] = np.concatenate(valConf_pred[k], axis=0)
                valConf_gt[k] = np.concatenate(valConf_gt[k], axis=0)
                print('Validation: {}-shot Acc: {:.3f}. Elapsed: {:.4f} s.'.format(
                        k,accuracy_score(valConf_gt[k],valConf_pred[k]),time.time() - timerEval))
                target_names = [str(i) for i in np.unique(valConf_gt[k])]
                print(classification_report(valConf_gt[k], valConf_pred[k],
                                            target_names=target_names))
                valConf_pred[k] = []
                valConf_gt[k] = []

        cost = 0
        timer = time.time()
        network.set('training')

    #################################################################
    ############ Meta-testing
    #################################################################
    # set net for testing
    network.set('evaluate')

    results = []
    for n in np.arange(len(opt['nTest'])):
        # validation episode loop
        for episodeTest, (x_support_set, y_support_set, x_target, target_y) in enumerate(data['test'][n]):

            # Re-arange the Target vectors between [0..nClasses_train]
            dictLabels, dictLabelsInverse = util.createDictLabels(y_support_set)
            y_support_set = util.fitLabelsToRange(dictLabels, y_support_set)
            target_y = util.fitLabelsToRange(dictLabels, target_y)
            unique_labels = dictLabels.keys()

            # k-shot loop
            for k in opt['nTestShot']:

                # Select k samples from each class from x_support_set and
                indexes_selected = []
                for k_selected in unique_labels:
                    selected = np.random.choice(np.squeeze(np.where((y_support_set.numpy() == dictLabels[k_selected])))
                                                , k, False)
                    indexes_selected.append(selected)

                # Select the k-shot examples from the Tensors
                x_support_set_k = x_support_set[torch.from_numpy(np.squeeze(indexes_selected).flatten())]
                y_support_set_k = y_support_set[torch.from_numpy(np.squeeze(indexes_selected).flatten())]

                # Convert them in Variables
                input = {}
                input['trainInput'] = Variable(x_support_set_k).float()
                input['trainTarget'] = Variable(y_support_set_k, requires_grad=False).long()
                input['testInput'] = Variable(x_target).float()
                input['testTarget'] = Variable(target_y, requires_grad=False).long()

                # Convert to GPU if needed
                if opt['useCUDA']:
                    input['trainInput'] = input['trainInput'].cuda()
                    input['trainTarget'] = input['trainTarget'].cuda()
                    input['testInput'] = input['testInput'].cuda()
                    input['testTarget'] = input['testTarget'].cuda()

                output = network(opt, input)

                # update stats test
                values_pred, indices_pred = torch.max(output, 1)
                target_y = util.fitLabelsToRange(dictLabelsInverse, target_y)
                indices_pred = util.fitLabelsToRange(dictLabelsInverse, indices_pred.cpu().data)
                testConf_pred[k].append(indices_pred.numpy())
                testConf_gt[k].append(target_y.numpy())

        for k in opt['nTestShot']:
            acc = []
            for i in np.arange(len(testConf_gt[k])):
                acc.append(accuracy_score(testConf_gt[k][i],testConf_pred[k][i]))
            low = np.mean(acc) - 1.96*(np.std(acc)/math.sqrt(len(acc)))
            high = np.mean(acc) + 1.96 * (np.std(acc) / math.sqrt(len(acc)))
            print('Test: nTest: {}. {}-shot. mAcc: {:.3f}. low mAcc: {:.3f}. high mAcc: {:.3f}.'.format(
                opt['nTest'][n],k,np.mean(acc),low, high))
            testConf_pred[k] = []
            testConf_gt[k] = []
            results.append((opt['nTest'][n],k,np.mean(acc),low, high))

    return results