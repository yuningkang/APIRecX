import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnCommon(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 layer_num,
                 batch_first,
                 drop_out = 0.0,
                 biderction=True,
                 rnn_type = "LSTM"
                 ):
        super(RnnCommon,self).__init__()

        self._rnn_type =rnn_type
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._layer_num = layer_num
        self._bidirection = biderction
        self._batch_first = batch_first
        self._num_direction == 2 if self._bidirection else 1

        if self._rnn_type == "LSTM":
            self._rnn_cell = nn.LSTMCell
        elif self._rnn_type == "RNN":
            self._rnn_cell = nn.RNNCell
        elif self._rnn_type == "GRU":
            self._rnn_cell = nn.GRUCell

       #定义前向和后向的cell实现多层
        self._fw_cell = nn.ModuleList()
        self._bw_cell = nn.ModuleList()

        for i in range(self._layer_num):
            layer_input_size = self._input_size if i == 0 else self._hidden_size*self._num_direction
            self._fw_cell.append(self._rnn_cell(layer_input_size,self._hidden_size))
            if self._bidirection:
                self._bw_cell.append(self._rnn_cell(layer_input_size,self._hidden_size))



    def _forward(self,cell,input,init_hidden,mask):
        if self._batch_first:
            input = input.transpose(0,1)
        seq_len = input.shape[1]
        output = []
        for i in range(seq_len):
            if self._rnn_type == "LSTM":
                h1,c1 = cell(input[i],init_hidden)
                h1 = h1 * mask[i]
                c1 = c1 * mask[i]
                output.append(h1)
                init_hidden =(h1,c1)
            else:
                h1 = cell(input[i],init_hidden)
                h1 = h1 * mask[i]
                output.append(h1)
                init_hidden = h1
        output = torch.stack(output,dim=0)

        return output,init_hidden

    def _backward(self,cell,input,init_hidden,mask):
        if self._batch_first:
            input = input.transpose(0,1)
        seq_len = input.shape[0]
        output = []
        for i in reversed(range(seq_len)):
            if self._rnn_type == "LSTM":
                h1,c1= cell(input[i],init_hidden)
                h1 = h1 * mask[i]
                c1 = c1 * mask[i]
                output.append(h1)
                init_hidden = (h1,c1)
            else:
                output.append(h1)
                init_hidden = h1

        output = torch.stack(output,dim=0)
        reversed()
        return output,init_hidden




    def forward(self, inputs,mask,init_hidden = None):
        '''

        :param inputs: [batch,seq,input_size] if batch_first
        :param init_hideen:
        :param mask :[batch,seq]
        :return:
        '''
        hn = []
        cn = []
        inputs = inputs.transpose(0,1) if self._batch_first else inputs
        mask = mask.transpose(0,1)
        mask= mask.unsuqueeze(dim=2).expand((-1,-1,self._hidden_size))
        if init_hidden == None:
            init_hidden =init_hidden(inputs.shape[1])
        for i in range(self._layer_num):
            #fw_output,bw_output of shape [seq_len,batch,hidden_size]
            #fw_hn of shape [batch_size,hidden_size]
            fw_output,fw_hidden =self._forward(self._fw_cell[i],inputs,init_hidden,mask)
            if self._bidirection:
                bw_output,bw_hidden = self._backward(self._bw_cell[i],inputs,init_hidden,mask)

            if self._rnn_type == "LSTM":
                hn.append(torch.cat((fw_hidden[0],bw_hidden[0]),dim=1) if self._bidirection else fw_hidden[0])
                cn.append(torch.cat((fw_hidden[1],bw_hidden[1]),dim=1) if self._bidirection else fw_hidden[1])
            else:
                hn.append(torch.cat((fw_hidden,bw_hidden),dim=1) if self._bidirection else fw_hidden)


            inputs = torch.cat((fw_output,bw_output),dim=2)

        output = inputs.transpose(0,1)
        hn = torch.stack(hn,dim=0)
        if self._rnn_type =="LSTM":
            cn = torch.stack(cn,dim=0)
            hidden = (hn,cn)
        else:
            hidden = hn

        return output,hidden


    def init_hidden(self,batch_size):
        h0 = torch.zeros(batch_size,self._hidden_size)
        if self._rnn_type == "LSTM":
            return h0,h0

        else :
            return h0
