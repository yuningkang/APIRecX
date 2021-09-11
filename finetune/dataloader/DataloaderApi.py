import numpy as np
import torch
import pickle
import random
from collections import defaultdict
from torch.autograd import Variable
from GPT.data_utils import *
from GPT.tokenization import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class train_data_seq:
    def __init__(self,vec,tag,num,cut_dot):
        self.vec = vec
        self.tag = tag
        self.line_num = num
        self.cut_dot = cut_dot
class Instance(object):
    def __init__(self,seq,tag,num,cut_dot):
        self.seq = seq
        self.tag = tag
        self.line_num = num
        self.cut_dot = cut_dot
class MInstance(object):
    def __init__(self, seq, tag):
        self.seq = seq
        self.tag = tag
def get_batch(dataset,batch_size,arg,batch_len,shuffle = False):
    # if arg.mode == "train":
    #     batch_num = int(np.ceil(len(dataset) / batch_size))
    #     if shuffle:
    #        random.shuffle(dataset)
    #     for i in range(batch_num):
    #
    #         yield  dataset[i*batch_size:(i+1)*batch_size]
    if arg.mode == "train":
        # for i in range(int((len(batch_len) / 2)) -1 ):
        #     # print(i)
        #     yield dataset[batch_len[i*2]:batch_len[i*2+1]]
        # line_num_list = list(batch_len.values())
        random.shuffle(batch_len)
        # print(line_num_list)
        for list_index in batch_len:
            # print(line_num)
            # print(list_index[0],list_index[-1])
            yield dataset[list_index[0]:list_index[-1]]
        # batch_num = int(np.ceil(len(dataset) / batch_size))
        # if shuffle:
        #    random.shuffle(dataset)
        # for i in range(batch_num):
        #
        #     yield  dataset[i*batch_size:(i+1)*batch_size]
    elif arg.mode == "detect":
        batch_size = 1
        batch_num = int(np.ceil(len(dataset) / batch_size))
        for i in range(batch_num):
            yield dataset[i * batch_size:(i + 1) * batch_size]
def get_batch_train(dataset,batch_size,arg,batch_len,shuffle = True):
    # if arg.mode == "train":
    #     batch_num = int(np.ceil(len(dataset) / batch_size))
    #     if shuffle:
    #        random.shuffle(dataset)
    #     for i in range(batch_num):
    #
    #         yield  dataset[i*batch_size:(i+1)*batch_size]
    if arg.mode == "train" or arg.mode == "test":
        # for i in range(int((len(batch_len) / 2)) -1 ):
        #     # print(i)
        #     yield dataset[batch_len[i*2]:batch_len[i*2+1]]
        # line_num_list = list(batch_len.values())
        # random.shuffle(batch_len)
        # # print(line_num_list)
        # for list_index in batch_len:
        #     # print(line_num)
        #     # print(list_index[0],list_index[-1])
        #     yield dataset[list_index[0]:list_index[-1]]
        batch_num = int(np.ceil(len(dataset) / batch_size))
        if shuffle:
           random.shuffle(dataset)
        for i in range(batch_num):

            yield  dataset[i*batch_size:(i+1)*batch_size]
    elif arg.mode == "detect":
        batch_size = 1
        batch_num = int(np.ceil(len(dataset) / batch_size))
        for i in range(batch_num):
            yield dataset[i * batch_size:(i + 1) * batch_size]
# def get_minibatch(dataset,batch_size,arg,batch_len,cut_dot,shuffle = False):
#     # if arg.mode == "train":
#     #     batch_num = int(np.ceil(len(dataset) / batch_size))
#     #     if shuffle:
#     #        random.shuffle(dataset)
#     pre_num = 0
#     for i,dot in enumerate(cut_dot):
# 
#         yield  dataset[pre_num:dot+1]

        # for i in range(int((len(batch_len) / 2)) -1 ):
        #     # print(i)
        #     yield dataset[batch_len[i*2]:batch_len[i*2+1]]
        # line_num_list = list(batch_len.values())
    # random.shuffle(batch_len)
    # print(line_num_list)
    # for list_index in batch_len:
    #     # print(line_num)
    #     # print(list_index[0],list_index[-1])
    #     yield dataset[list_index[0]:list_index[-1]]


def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)
def train_data_loader(file,file1=None):
    train_data = []
    seq=[]
    tag =[]
    a = list()
    sentence = set()
    with open(file,"rb")as  f:
        train_data_dict = pickle.load(f)
    # with open(file,"r",encoding="utf-8")as  f:
    #     for line in f.readlines():
    #         # line = line.strip().split(" ")
    #         # sequence = [int(i) for i in line]
    #         # print(sequence)
    #         seq.append(line.strip())
    # with open(file1, "r", encoding="utf-8")as  f1:
    #     for line in f1.readlines():
    #         tag.append(line.strip())
    # for i,vec in enumerate(seq):
    #     s = vec + " " +tag[i]
    #     print(s)
    #     sentence.add(s)
    # for data in sentence:
    #     lenth = len(data)
    #     lable = data.strip().split(" ")[-1]
    #     vec = data[:lenth]
    #     train_data.append(Instance(vec,lable))
        


        
    # for k,seq in train_data_dict.items():
    #         for v in seq:
    #             train_data.append( Instance(v,k))
    batch_len = defaultdict(list)
    for i ,inst in  enumerate(train_data_dict):
        # print(inst.line_num)
        batch_len[inst.line_num].append(i)
        train_data.append(Instance(inst.vec, inst.tag,inst.line_num,inst.cut_dot))
        # train_data.append(Instance(inst.vec, inst.tag))
    # print(batch_len)

    return train_data,batch_len
def test_data_loader(file,vocab):
    test_data = []
    with open(file , "r", encoding="utf-8") as f1:
       for line in f1.readlines():
          seq = line.strip().split("##")[0].split(" ")
          loc = line.strip().split("##")[1]
          if isinstance(seq, list):
              seq_index = [vocab.get(api)for api in seq]
          test_data.append(Instance(seq_index,loc))

    return test_data
def batch_numberize(batch,device,arg):
    if arg.mode == "train" or arg.mode == "test":
        batch_size = len(batch)
        seq_lengths = []
        # length = max([len(batch[i].seq) for i in range(batch_size)])
        length = arg.max_seq_len
        #apiseq_idx = torch.LongTensor(batch_size,length).zero_().to(device)
        #tag_idx = torch.LongTensor(batch_size).zero_().to(device)
        apiseq_idx = torch.LongTensor(batch_size,length).zero_().to(device)
        tag_idx = torch.LongTensor(batch_size).zero_().to(device)
        # word_idx = torch.zero_((batch_size,length),dtype =torch.long).to(device)
        # tag_idx = torch.zero_(batch_size,dtype=torch.long).to(device)
        # mask = torch.ByteTensor(batch_size,length).zero_().to(device)
        mask = torch.zeros(batch_size, length).to(device)
        for i,inst in enumerate(batch):
            seq_lengths.append(len(inst.input_ids))
            seq_len = len(inst.input_ids)
            apiseq_idx[i, :seq_len] = torch.tensor(np.asarray(inst.input_ids))
            # tag_idx[i] = torch.tensor(np.asarray(inst.tag))
        mask = torch.eq(apiseq_idx,0)
        
        
    elif arg.mode == "detect":
        batch_size = len(batch)
        seq_lengths = []
        # length = max([len(batch[i].seq) for i in range(batch_size)])
        length = arg.max_seq_len
        # apiseq_idx = torch.LongTensor(batch_size,length).zero_().to(device)
        # tag_idx = torch.LongTensor(batch_size).zero_().to(device)
        apiseq_idx = torch.LongTensor(batch_size, length).zero_().to(device)
        tag_idx = torch.LongTensor(batch_size).zero_().to(device)
        # word_idx = torch.zero_((batch_size,length),dtype =torch.long).to(device)
        # tag_idx = torch.zero_(batch_size,dtype=torch.long).to(device)
        # mask = torch.ByteTensor(batch_size,length).zero_().to(device)
        mask = torch.zeros(batch_size, length).to(device)
        for i, inst in enumerate(batch):
            seq_lengths.append(len(inst.input_ids))
            seq_len = len(inst.input_ids)
            apiseq_idx[i, :seq_len] = torch.tensor(np.asarray(inst.input_ids))
            # tag_idx[i] = torch.tensor(np.asarray(inst.tag))
        mask = torch.eq(apiseq_idx, 0)


    return apiseq_idx,tag_idx,mask,seq_lengths





