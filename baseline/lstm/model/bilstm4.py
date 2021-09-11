from model.CommonRNN import *
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from  dataloader.DataloaderApi import simple_elementwise_apply
class BiLSTM(nn.Module):
    def __init__(self,args,vocab,vocab_method,embedding_weight):
        super(BiLSTM,self).__init__()
        # self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weight),freeze=False)
        self.word_emb = nn.Embedding(len(vocab),128)
        self.word_emb = nn.Embedding(len(vocab_method), 128)
        # embedding_dim = embedding_weight.shape[1]
        embedding_dim = 256
        print(embedding_dim)
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
        self.num_direction = 2 if  self.bidirectional else 1
        self.emb_dropout = nn.Dropout(args.dropout_emb)
        self.linear_dropout = nn.Dropout(args.dropout_linear)
        # self.linear_1 = nn.Linear(in_features=args.hidden_size * self.num_direction,
        #                         out_features=len(vocab))
        self.linear = nn.Linear(in_features=args.hidden_size*self.num_direction,
                                 out_features=len(vocab_method))


    def self_attention(self,encoder_output,hidden):
        '''

        :param encoder_output:[seq_len,batch_size,hidden_size*num_direction]
        :param hidden: [batch_size,hidden_size*num_direction]
        :return:
        '''
        hidden = hidden.unsqueeze(dim=1)
        simulation = torch.bmm(encoder_output, hidden.transpose(1, 2))
        simulation = simulation.squeeze(dim=2)
        # simlation of shape [batch_size,seq_len]
        att_weight = F.softmax(simulation, dim=1)
        # att_weight of shape [batch_size,seq_len]
        output = torch.bmm(att_weight.unsqueeze(dim=1), encoder_output).squeeze(dim=1)
        return  output




    def forward(self,inputs,inputs_1):
        '''

        :param inputs: [batch_size,seq_len]
        :param mask: [batch_size,seq_len]
        :return:
        '''
        # if arg.mode == "train":
            # word_emb_1 = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weight))
            # word_emb = word_emb_1(inputs)
        word_emb = self.word_emb(inputs)
        word_emb_1 = self.word_emb(inputs_1)
        whole_word_emb = torch.cat([word_emb, word_emb_1], dim=2)
        if self.training:
            whole_word_emb = self.emb_dropout(whole_word_emb)
        # print(word_emb.shape)
        #[batch_size,seq_len,hidden_size*num_derection]
        #outputs:[batch_size,seq_len,hidden_size*num_direction]
        # word_emb = pack_padded_sequence(word_emb, seq_lengths, batch_first=True, enforce_sorted=False)
        outputs,(hn,cn) = self.lstm(whole_word_emb)

        logit = self.linear(outputs)
        # logit = hn[1]
        #logit = F.softmax(logit,dim=1)
        # elif arg.mode == "detect":
        #     word_emb = self.word_emb(inputs)
        #     # [batch_size,seq_len,hidden_size*num_derection]
        #     # outputs:[batch_size,seq_len,hidden_size*num_direction]
        #     outputs, _ = self.lstm(word_emb,mask)
        #     #seq_predict_result:[batch_size,seq_len,tag_size]
        #     seq_predict_result = self.linear(outputs)
        #     logit =F.softmax(seq_predict_result,dim=1)

        return logit