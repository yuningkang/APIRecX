import numpy as np
import torch
import pickle
import random
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class train_data_seq:
    def __init__(self,vec,tag):
        self.vec = vec
        self.tag = tag
class Instance(object):
    def __init__(self,seq,tag):
        self.seq = seq
        self.tag = tag
class MInstance(object):
    def __init__(self, seq, tag):
        self.seq = seq
        self.tag = tag
def get_batch(dataset,batch_size,arg,shuffle = True):
    if arg.mode == "train":
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
    for inst in  train_data_dict:
        train_data.append(Instance(inst.vec, inst.tag))
    return train_data
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
def batch_numberize(batch,vocab,device,arg):
    if arg.mode == "train":
        batch_size = len(batch)
        seq_lengths = []
        length = max([len(batch[i].seq) for i in range(batch_size)])
        #apiseq_idx = torch.LongTensor(batch_size,length).zero_().to(device)
        #tag_idx = torch.LongTensor(batch_size).zero_().to(device)
        apiseq_idx = torch.LongTensor(batch_size,length).zero_().to(device)
        tag_idx = torch.LongTensor(batch_size).zero_().to(device)
        # word_idx = torch.zero_((batch_size,length),dtype =torch.long).to(device)
        # tag_idx = torch.zero_(batch_size,dtype=torch.long).to(device)
        # mask = torch.ByteTensor(batch_size,length).zero_().to(device)
        mask = torch.zeros(batch_size, length).to(device)
        for i,inst in enumerate(batch):
            seq_lengths.append(len(inst.seq))
            seq_len = len(inst.seq)
            apiseq_idx[i, :seq_len] = torch.tensor(np.asarray(inst.seq))
            tag_idx[i] = torch.tensor(np.asarray(inst.tag))
            mask[i,:seq_len].fill_(1)
        
        
    elif arg.mode == "detect":
        batch_size = len(batch)
        length = max([len(batch[i].seq) for i in range(batch_size)])
        apiseq_idx = torch.LongTensor(batch_size, length).zero_().to(device)
        tag_idx = torch.LongTensor(batch_size, length).zero_().to(device)
        # word_idx = torch.zero_((batch_size,length),dtype =torch.long).to(device)
        # tag_idx = torch.zero_(batch_size,dtype=torch.long).to(device)
        # mask = torch.ByteTensor(batch_size,length).zero_().to(device)
        mask = torch.zeros(batch_size, length).to(device)
        for i, inst in enumerate(batch):
            seq_len = len(inst.seq)
            apiseq_idx[i, :seq_len] = torch.tensor(np.asarray(inst.seq))
            tag_idx[i, :seq_len] = torch.tensor(np.asarray(inst.seq))
            mask[i, :seq_len].fill_(1)


    return apiseq_idx,tag_idx,mask,seq_lengths





