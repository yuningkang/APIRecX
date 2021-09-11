import torch
import torch.nn as nn
class CommonRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 layer_num,
                 rnn_type="lstm",
                 batch_first = True,
                 bidirectional=True,


                 ):
        super(CommonRNN,self).__init__()
        self.rnn =['GRU','LSTM','RNN']
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.rnn_type = rnn_type.upper()
        self.bidirection = bidirectional
        self.drection_num = 2 if self.bidirection else 1
        self.batch_first = batch_first
        assert self.rnn_type in self.rnn

        if self.rnn_type == "LSTM":
            self.rnn_cell = nn.LSTMCell
        elif self.rnn_type == "GRU":
            self.rnn_cell = nn.GRUCell
        elif self.rnn_type == "RNN":
            self.rnn_cell =nn.RNNCell
        self.fw_cell_list = nn.ModuleList()
        self.bw_cell_list = nn.ModuleList()
        for i in range(self.layer_num):
            self.input_layer_num = self.input_size if i == 0 else self.hidden_size * self.drection_num
            self.fw_cell_list.append(self.rnn_cell(self.input_layer_num,self.hidden_size))
            if self.bidirection:
                self.bw_cell_list.append(self.rnn_cell(self.input_layer_num,self.hidden_size))

    def _forward(self,cell,inputs,mask,hidden=None):
        '''

        :param
        :param inputs:[seq_len,batch_size,input_size]
        :param hidden: [batch_size,hidden_size]
        :return:
        '''
        outputs = []
        seq_len = inputs.shape[0]
        fw_next = hidden
        for i in  range(seq_len):
            if self.rnn_type == 'LSTM':
                h1,c1 = cell(inputs[i],fw_next)
                h1 = h1 * mask[i]
                c1 = c1 * mask[i]
                fw_next = (h1,c1)
                outputs.append(h1)
            else:
                h1 = cell(inputs[i],fw_next)
                h1 = h1 * mask[i]
                fw_next = h1
                outputs.append(fw_next)
        #return size is[seq_len,batch_size,hidden_size] and fw_next:[batch_size,hidden_size]
        return torch.stack(tuple(outputs),dim=0),fw_next
    def _backward(self,cell,inputs,mask,hidden=None):
        bw_outputs = []
        seq_len = inputs.shape[0]

        bw_next = hidden
        for i in reversed(range(seq_len)):
            if self.rnn_type == 'LSTM':
                h1, c1 = cell(inputs[i], bw_next)
                h1 = h1 * mask[i]
                c1 = c1 * mask[i]
                bw_next = (h1, c1)
                bw_outputs.append(h1)
            else:
                h1 = cell(inputs[i], bw_next)
                h1 = h1 * mask[i]
                bw_next = h1
                bw_outputs.append(bw_next)

        bw_outputs.reverse()
        return torch.stack(tuple(bw_outputs), dim=0), bw_next

    def forward(self, inputs,mask,hidden=None):
        '''

        :param inputs:
        :return:
        '''
        hn = []
        cn = []
        if self.batch_first:
            inputs = inputs.transpose(0,1)
            mask = mask.transpose(0,1)
        batch_size = inputs.shape[1]
        mask = mask.unsqueeze(dim=2).expand(-1,-1,self.hidden_size)
        if hidden == None:
            hidden = self.init_hidden(batch_size,inputs.device)
        for i in range(self.layer_num):
            fw_outputs ,fw_hn =self._forward(self.fw_cell_list[i],inputs,mask,hidden)
            bw_outputs,bw_hn = None,None
            if self.bidirection:
                bw_outputs,bw_hn = self._backward(self.bw_cell_list[i],inputs,mask,hidden)
            if self.rnn_type == "LSTM":
                hn.append(torch.cat((fw_hn[0],bw_hn[0]),dim=1) if self.bidirection else fw_hn[0])
                cn.append(torch.cat((fw_hn[1],bw_hn[1]),dim=1) if self.bidirection else fw_hn[1])
            else:
                hn.append(torch.cat((fw_hn,bw_hn),dim=1) if self.bidirection else fw_hn)

            inputs = torch.cat((fw_outputs,bw_outputs),dim=2) if self.bidirection else fw_outputs
        hn = torch.stack(tuple(hn),dim=0)
        if self.rnn_type == "LSTM":
            cn = torch.stack(tuple(cn),dim=0)
        hidden = (hn,cn)
        outputs = inputs.transpose(0,1)
        return outputs,hidden
    def init_hidden(self,batch_size,device):
        hn = torch.zeros(batch_size,self.hidden_size,device = device)
        if self.rnn_type == "LSTM":
            init_hidden = hn,hn
        else:
            init_hidden = hn

        return init_hidden




