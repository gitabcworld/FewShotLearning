import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init

class MetaLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(MetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.WF = nn.Parameter(torch.FloatTensor(hidden_size + 2, 1))
        self.WI = nn.Parameter(torch.FloatTensor(hidden_size + 2, 1))
        # initial cell state is a param
        self.cI = nn.Parameter(torch.FloatTensor(input_size, 1))
        self.bI = nn.Parameter(torch.FloatTensor(1, 1))
        self.bF = nn.Parameter(torch.FloatTensor(1, 1))
        self.m = nn.Parameter(torch.FloatTensor(1))

        '''
        self.WF = Variable(torch.FloatTensor(hidden_size + 2, 1), requires_grad=True)
        self.WI = Variable(torch.FloatTensor(hidden_size + 2, 1), requires_grad=True)
        # initial cell state is a param
        self.cI = Variable(torch.FloatTensor(input_size, 1), requires_grad=True)
        self.bI = Variable(torch.FloatTensor(1, 1), requires_grad=True)
        self.bF = Variable(torch.FloatTensor(1, 1), requires_grad=True)
        self.m = Variable(torch.FloatTensor(1), requires_grad=True)
        '''

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters 
        """
        self.WF.data.uniform_(-0.01, 0.01)
        self.WI.data.uniform_(-0.01, 0.01)
        self.cI.data.uniform_(-0.01, 0.01)
        self.bI.data.zero_()
        self.bF.data.zero_()
        self.m.data.zero_()

    def forward(self, input_, grads_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        # next forget, input gate
        (fS, iS, cS, deltaS) = hx
        fS = torch.cat((cS, fS), 1)
        iS = torch.cat((cS, iS), 1)

        fS = torch.mm(torch.cat((input_,fS), 1),self.WF)
        fS += self.bF.expand_as(fS)

        iS = torch.mm(torch.cat((input_,iS), 1),self.WI)
        iS += self.bI.expand_as(iS)

        # next delta
        deltaS = self.m * deltaS - nn.Sigmoid()(iS).mul(grads_)

        # next cell/params
        cS = nn.Sigmoid()(fS).mul(cS) + deltaS

        return fS, iS, cS, deltaS

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MetaLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size,
                    batch_first = False, num_layers=1):
        super(MetaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = MetaLSTMCell(input_size=layer_input_size,
                              hidden_size=hidden_size)
            self.cells.append(cell)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, grads_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            hx = cell(input_=input_[time],grads_=grads_[time], hx=hx)
            #mask = (time < length).float().unsqueeze(1).expand_as(h_next[0])
            #fS_next = h_next[0] * mask + hx[0] * (1 - mask)
            #iS_next = h_next[1] * mask + hx[1] * (1 - mask)
            #cS_next = h_next[2] * mask + hx[2] * (1 - mask)
            #deltaS_next = h_next[3] * mask + hx[3] * (1 - mask)
            #hx_next = (fS_next, iS_next, cS_next, deltaS_next)
            #output.append(h_next)
            #hx = hx_next
        #output = torch.stack(output, 0)
        #return output,hx
        #return hx[2],hx
        return hx

    def forward(self, input_, length=None, hx=None):

        x_input = input_[0] # output from lstm
        grad_input = input_[1] # gradients from learner
        if self.batch_first:
            x_input = x_input.transpose(0, 1)
            grad_input = grad_input.transpose(0, 1)
        max_time, batch_size, _ = x_input.data.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if x_input.is_cuda:
                length = length.cuda()
        # hidden variables. Here we have fS, iS and cS.
        if hx is None:
            fS = Variable(grad_input.data.new(batch_size, 1).zero_())
            iS = Variable(grad_input.data.new(batch_size, 1).zero_())
            cS = (self.cells[0].cI).unsqueeze(1)
            deltaS = Variable(grad_input.data.new(batch_size, 1).zero_())
            hx = (fS, iS, cS, deltaS)

        fS_n = []
        iS_n = []
        cS_n = []
        deltaS_n = []
        for layer in range(self.num_layers):
            hx_new = MetaLSTM._forward_rnn(
                cell=self.cells[layer], input_=x_input,
                grads_= grad_input, length=length, hx=hx)
            fS_n.append(hx_new[0])
            iS_n.append(hx_new[1])
            cS_n.append(hx_new[2])
            deltaS_n.append(hx_new[3])
        fS_n = torch.stack(fS_n, 0)
        iS_n = torch.stack(iS_n, 0)
        cS_n = torch.stack(cS_n, 0)
        fS_n = torch.stack(fS_n, 0)
        deltaS_n = torch.stack(deltaS_n, 0)
        # return cS and the actual state
        return (fS_n, iS_n, cS_n, deltaS_n)

