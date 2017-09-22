import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
import model.lstm.bnlstm as bnlstm
import model.lstm.metalstm as metalstm
from model.lstm.recurrentLSTMNetwork import RecurrentLSTMNetwork
from model.lstm.lstmhelper import preprocess
from utils import util
from visualize.visualize import make_dot

class MetaLearner(nn.Module):
    def __init__(self, opt):
        super(MetaLearner, self).__init__()

        self.nHidden = opt['nHidden'] if 'nHidden' in opt.keys() else 20
        self.maxGradNorm = opt['maxGradNorm'] if 'maxGradNorm' in opt.keys() else 0.25

        #inputFeatures = 4 #loss(2) + preGrad(2) = 4
        inputFeatures = 2  # loss(2) + preGrad(2) = 4
        batchNormalization1 = opt['BN1'] if 'BN1' in opt.keys() else False
        maxBatchNormalizationLayers = opt['steps'] if 'steps' in opt.keys() else 1
        batchNormalization1 = False
        if batchNormalization1:
            self.lstm = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=inputFeatures,
                         hidden_size=self.nHidden, batch_first=True,
                         max_length=maxBatchNormalizationLayers)
        else:
            self.lstm = nn.LSTM(input_size=inputFeatures,
                                 hidden_size=self.nHidden,
                                 batch_first=True,
                                 num_layers=maxBatchNormalizationLayers)

        # set initial hidden layer and cell state
        # num_layers * num_directions, batch, hidden_size
        batch_size = 1
        self.lstm_h0_c0 = None

        #self.lstm_c0 = Variable(torch.rand(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
        #                        requires_grad=False).cuda()
        #self.lstm_h0 = Variable(torch.rand(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
        #                        requires_grad=False).cuda()

        # Meta-learner LSTM
        # TODO: BatchNormalization in MetaLSTM
        batchNormalization2 = opt['BN2'] if 'BN2' in opt.keys() else False
        self.lstm2 = metalstm.MetaLSTM(input_size = opt['nParams'],
                             hidden_size = self.nHidden,
                             batch_first=True,
                             num_layers=maxBatchNormalizationLayers)

        # set initial c0 and h0 states for lstm2
        batch_size = 1
        self.lstm2_fS_iS_cS_deltaS = None

        # Join parameters as input for optimizer
        self.params = lambda: list(self.lstm.named_parameters()) + list(self.lstm2.named_parameters())
        self.params = { param[0]:param[1] for param in self.params()}

        # initialize weights learner
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(self.lstm, name)
                weight.data.uniform_(-0.01, 0.01)

        # initialize weights meta-learner for all layers.
        for params in self.lstm2.named_parameters():
            if 'WF' in names[0] or names[0] in names[0] or 'cI' in params[0]:
                params[1].data.uniform_(-0.01, 0.01)

        # want initial forget value to be high and input value
        # to be low so that model starts with gradient descent
        for params in self.lstm2.named_parameters():
            if "cell_0.bF" in names[0]:
                params[0].data.uniform_(4, 5)
            if "cell_0.bI" in names[0]:
                params[0].data.uniform_(-4, -5)

        # Set initial cell state = learner's initial parameters
        initialParams = torch.cat([value.view(-1) for key,value in opt['learnerParams'].items()], 0)
        #self.lstm2.cells[0].cI = initialParams.unsqueeze(1).clone()

        #torch.nn.Parameter(initial_param['weight'])
        for params in self.lstm2.named_parameters():
            if "cell_0.cI" in params[0]:
                params[1].data = initialParams.data.clone()
        # self.lstm2.cells[0].cI.data = initialParams.view_as(self.lstm2.cells[0].cI).clone()
        a = 0

        #for params in self.lstm2.parameters():
        #    params.retain_grad()

        #for params in self.lstm.parameters():
        #    params.retain_grad()


    def forward(self, learner, trainInput, trainTarget, testInput, testTarget
                , steps, batchSize, evaluate = False ):

        trainSize = trainInput.size(0)

        # reset parameters for each dataset
        # Modules with learnable parameters have a reset(). This function
        # allows to re-initialize parameters. It's also used for weight
        # initialization.
        learner.reset()
        learner.set('training')

        # Set learner's initial parameters = initial cell state
        util.unflattenParams(learner.model, self.lstm2.cells[0].cI)

        #for params in self.lstm2.named_parameters():
        #    if "cell_0.cI" in params[0]:
        #        util.unflattenParams(learner.model, params[1])
        #util.unflattenParams(learner.model, self.lstm2.cells[0].cI)

        idx = 0
        for s in range(steps):
            for i in range(0,trainSize,batchSize):
                # get image input & label
                x = trainInput[i:batchSize,:]
                y = trainTarget[i:batchSize]

                #if idx > 0:
                #    # break computational graph
                #    learnerParams = output.detach()
                #    # Unflatten params and copy parameters to learner network
                #    util.unflattenParams(learner.model,learnerParams)

                # get gradient and loss w/r/t learnerParams for input+label
                grad_model, loss_model = learner.feval(x,y)
                grad_model = grad_model.view(grad_model.size()[0], 1, 1)

                # Delete albert
                inputs = torch.cat((grad_model, loss_model.expand_as(grad_model)), 2)
                '''
                # preprocess grad & loss by DeepMind "Learning to learn"
                preGrad, preLoss = preprocess(grad_model,loss_model)
                # use meta-learner to get learner's next parameters
                lossExpand = preLoss.expand_as(preGrad)
                inputs = torch.cat((lossExpand,preGrad),2)
                '''
                output, self.lstm_h0_c0 = self.lstm(inputs, self.lstm_h0_c0)
                self.lstm2_fS_iS_cS_deltaS = self.lstm2((output,grad_model),
                                                                self.lstm2_fS_iS_cS_deltaS)

                ## Delete
                util.unflattenParams(learner.modelF, self.lstm2_fS_iS_cS_deltaS[2])
                output, loss = learner(testInput, testTarget)
                #g = make_dot(loss)
                #g.render('/home/aberenguel/tmp/g.gv', view=True)
                #torch.autograd.grad(loss, list(self.lstm2.parameters()) + list(self.lstm.parameters()))
                ######

                # get the internal cell state
                output = self.lstm2_fS_iS_cS_deltaS[2]

                # Unflatten params and copy parameters to learner network
                util.unflattenParams(learner.model, output)

                idx = idx + 1

        # Unflatten params and copy parameters to learner network
        util.unflattenParams(learner.modelF, output)

        ## get loss + predictions from learner.
        ## use batch-stats when meta-training; otherwise, use running-stats
        if evaluate:
            learner.set('evaluate')
        # Do a dummy forward / backward pass to get the correct grads into learner
        output, loss = learner(testInput, testTarget)

        # replace the gradients with the lstm2
        torch.autograd.grad(loss, self.lstm2.parameters())

        return output, loss


    def gradNorm(self, loss):

        print('Grads lstm + lstm2:')
        for params in self.lstm.parameters():
            print(params.grad)

        for params in self.lstm2.parameters():
            print(params.grad)
        a = 0

    def setCuda(self, value = True):
        if value:
            self.lstm.cuda()
            self.lstm2.cuda()
        else:
            self.lstm.cpu()
            self.lstm2.cpu()