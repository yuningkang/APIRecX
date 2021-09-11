from model.CommonRNN import *
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from  dataloader.DataloaderApi import simple_elementwise_apply
class BiLSTM_P(nn.Module):
    def __init__(self,args,vocab,embedding_weight):
        super(BiLSTM_P,self).__init__()
        self.word_emb_new = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weight))
        # weight = np.random.uniform(-0.05, 0.05, (len(vocab.keys()), 50)).astype(np.float32)
        # self.word_emb_new = nn.Embedding.from_pretrained(torch.from_numpy(weight))
        embedding_dim = embedding_weight.shape[1]
        # self.lstm = CommonRNN(input_size=embedding_dim,
        #                  hidden_size=args.hidden_size,
        #                  layer_num = args.num_layers,
        #                  batch_first=True,
        #                  bidirectional=True,
        #                  rnn_type='lstm'
        #                  )
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=True,
                            bidirectional=False
                            )
        self.bidirectional = False
        self.num_direction = 2 if self.bidirectional else 1
        self.emb_dropout = nn.Dropout(args.dropout_emb)
        self.linear_dropout = nn.Dropout(args.dropout_linear)

        self.linear_new = nn.Linear(in_features=args.hidden_size*self.num_direction,
                                 out_features=len(vocab.keys()))


    # def self_attention(self,encoder_output,hidden):
    #     '''
    #
    #     :param encoder_output:[seq_len,batch_size,hidden_size*num_direction]
    #     :param hidden: [batch_size,hidden_size*num_direction]
    #     :return:
    #     '''
    #     hidden = hidden.unsqueeze(dim=1)
    #     simulation = torch.bmm(encoder_output, hidden.transpose(1, 2))
    #     simulation = simulation.squeeze(dim=2)
    #     # simlation of shape [batch_size,seq_len]
    #     att_weight = F.softmax(simulation, dim=1)
    #     # att_weight of shape [batch_size,seq_len]
    #     output = torch.bmm(att_weight.unsqueeze(dim=1), encoder_output).squeeze(dim=1)
    #     return  output
    #
    #
    #

    def forward(self,inputs,arg,mask,seq_lengths):
        '''

        :param inputs: [batch_size,seq_len]
        :param mask: [batch_size,seq_len]
        :return:
        '''
        if arg.mode == "train":
            word_emb = self.word_emb_new(inputs)
            if self.training:
               word_emb =  self.emb_dropout(word_emb)
            #[batch_size,seq_len,hidden_size*num_derection]
            #outputs:[batch_size,seq_len,hidden_size*num_direction]
            word_emb = pack_padded_sequence(word_emb, seq_lengths, batch_first=True, enforce_sorted=False)
            outputs,(hn,cn) = self.lstm(word_emb)
            # outputs = outputs.transpose(1,2)
            #[batch_size,hidden_size*num_derection,1]
            # outputs = F.max_pool1d(outputs,kernel_size = outputs.shape[-1]).squeeze(dim=2)
            #[batch_size,tag_size]
            # outputs =  self._linear_dropout(outputs)
            #选取target api出现的时间步的输出做预测
            # a = outputs[:, tag[:, 1:], :]
            # location = tag[:,1:]
            # pred = torch.FloatTensor(outputs.shape[0],1,outputs.shape[2]).zero_().to(arg.device)
            # for i in range(inputs.shape[0]):
            #     pred[i] = outputs[i,location[i],:]
            # pred = pred.squeeze(dim=1)
            #outputs = self.self_attention(outputs,hn[0])
            logit = self.linear_new(hn[1])
            #logit = F.softmax(logit,dim=1)
        elif arg.mode == "detect":
            word_emb = self.word_emb(inputs)
            # [batch_size,seq_len,hidden_size*num_derection]
            # outputs:[batch_size,seq_len,hidden_size*num_direction]
            outputs, _ = self.lstm(word_emb)
            #seq_predict_result:[batch_size,seq_len,tag_size]
            seq_predict_result = self.linear(outputs)
            seq_predict_result = seq_predict_result.squeeze(dim=0)
            logit =F.softmax(seq_predict_result,dim=1)

        return logit