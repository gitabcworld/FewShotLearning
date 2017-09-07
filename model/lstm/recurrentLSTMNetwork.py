import torch
from torch import nn
from torch.autograd import Variable

class RecurrentLSTMNetwork(nn.Module):
    def __init__(self, opt):
        super(RecurrentLSTMNetwork, self).__init__()

        self.inputFeatures = opt['inputFeatures'] if 'inputFeatures' in opt.keys() else 10
        self.hiddenFeatures = opt['hiddenFeatures'] if 'hiddenFeatures' in opt.keys() else 100
        self.outputType = opt['outputType'] if 'outputType' in opt.keys() else 'last' # 'last' or 'all'
        self.batchNormalization = opt['batchNormalization'] if 'batchNormalization' in opt.keys() else False
        self.maxBatchNormalizationLayers = opt['maxBatchNormalizationLayers'] if 'batchNormalization' in opt.keys() else 10

        # parameters
        self.p = {}
        self.p['W'] = Variable(torch.zeros(self.inputFeatures+self.hiddenFeatures,4 * self.hiddenFeatures),
                               requires_grad = True)
        self.params = [self.p['W']]

        #TODO: delete this line. only for debugging
        self.batchNormalization = True

        if self.batchNormalization:
            # TODO: check if nn.BatchNorm1d or torch.legacy.nn.BatchNormalization
            # translation and scaling parameters are shared across time.
            lstm_bn = nn.BatchNorm1d(4*self.hiddenFeatures)
            cell_bn = nn.BatchNorm1d(self.hiddenFeatures)
            self.layers = {'lstm_bn':[lstm_bn],'cell_bn':[cell_bn]}

            for i in range(2,self.maxBatchNormalizationLayers):
                lstm_bn = nn.BatchNorm1d(4*self.hiddenFeatures)
                cell_bn = nn.BatchNorm1d(self.hiddenFeatures)
                self.layers['lstm_bn'].append(lstm_bn)
                self.layers['cell_bn'].append(cell_bn)

            # Initializing scaling to <1 is recommended for LSTM batch norm
            # TODO: why only the first are initialized??
            self.layers['lstm_bn'][0].weight.data.fill_(0.1)
            self.layers['lstm_bn'][0].bias.data.zero_()
            self.layers['cell_bn'][0].weight.data.fill_(0.1)
            self.layers['cell_bn'][0].bias.data.zero_()

            self.params = self.params + \
                              list(self.layers['lstm_bn'][0].parameters()) + \
                              list(self.layers['lstm_bn'][0].parameters())
        else:
            self.p['b'] = Variable(torch.zeros(1, 4*self.hiddenFeatures),
                                   require_grad = True)
            self.params = self.params + [self.p['b']]
            self.layers = {}

    def setCuda(self, value = True):
        # If value is a string then use self.opt
        # If it is not a string then it should be True or False
        if value == True:
            for key in self.p.keys():
                self.p[key].cuda()
            for key in self.layers.keys():
                for i in range(len(self.layers[key])):
                    self.layers[key][i].cuda()
        else:
            for key in self.p.keys():
                self.p[key].cpu()
            for key in self.layers.keys():
                for i in range(len(self.layers[key])):
                    self.layers[key][i].cpu()

    def forward(self, x, prevState = None ):

        # dimensions
        if len(x.size()) == 2: x = x.unsqueeze(0)
        batch = x.size(0)
        steps = x.size(1)

        if prevState == None: prevState = {}
        hs = {}
        cs = {}
        for t in range(steps):
            # xt
            xt = x[:,t,:]
            # prev h and pre c
            hp = hs[t-1] or prevState.h or torch.zeros()
        a = 0
