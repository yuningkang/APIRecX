import numpy as np
import torch
import pickle
from torch.autograd import Variable
class Instance(object):
    def __init__(self,words,tag):
        self.words = words
        self.tag = tag
def Dataloader(file):
    alldata = []
    with open(file,'r',encoding='UTF-8') as file:
        for line in file:
            divide = line.strip().split("|||")
            words = divide[1].split(" ")
            tag = divide[0]
            data = Instance(words,tag)
            alldata.append(data)
    np.random.shuffle(alldata)
    return alldata

# def batch_numberize(batch,vocab,config):
#     words_num = len(batch)
#
#     a = 0
#     index = 0
#
#     length = len(batch[0].words)
#     for i in range(len(batch)):
#         if len(batch[i].words)>length:
#             length = len(batch[i].words)
#     words = Variable(torch.LongTensor(words_num,length).zero_())
#     tags = Variable(torch.LongTensor(words_num).zero_())
#
#     for inst in batch:
#         b = 0
#         for cur_word in inst.words:
#             words[a,b] = vocab.word2id(cur_word)
#             #tags[a] = vocab.tag2id(tag)
#             b += 1
#
#         #_tag2id为此种形式{'1 ': 0, '0 ': 1}
#         #_id2tag为这种形式{0: '1 ', 1: '0 '}
#         tags[index] = int(vocab.tag2id(inst.tag))
#     #tags[index] = vocab.tag2id(inst.tag)
#     index += 1
#     a += 1
#
#     return words,tags

# def dataslice(data,config):
#     batch_num = int(np.ceil(len(data) / 32))
#     for i in range(batch_num):
#         cur_batch_size = 32 if i < batch_num - 1 else len(data) - 32 * i
#         insts = [data[i*32+b]for b in range(cur_batch_size)]
#         np.random.shuffle(insts)
#         yield insts

def get_batch(dataset,batch_size,shuffle = True):
    batch_num = int(np.ceil(len(dataset) / batch_size))
    if shuffle:
        np.random.shuffle(dataset)
    for i in range(batch_num):

        yield  dataset[i*batch_size:(i+1)*batch_size]


def batch_numberize(batch,vocab,device):
    batch_size = len(batch)
    # length = len(batch[0].words)
    # for i in  range(len(batch)):
    #     if batch[i].words>length:
    #         length = len(batch[i].words)
    length = max([len(batch[i].words) for i in range(batch_size)])
    #print(length)
    word_idx = torch.LongTensor(batch_size,length).zero_().to(device)
    tag_idx = torch.LongTensor(batch_size).zero_().to(device)
    # word_idx = torch.zero_((batch_size,length),dtype =torch.long).to(device)
    # tag_idx = torch.zero_(batch_size,dtype=torch.long).to(device)
    # mask = torch.ByteTensor(batch_size,length).zero_().to(device)
    mask = torch.zeros(batch_size, length).to(device)
    for i,inst in enumerate(batch):
        seq_len = len(inst.words)
        #print(word_idx.shape)
        word_idx[i,:seq_len] = torch.tensor(vocab.extword2id(inst.words))
        tag_idx[i] = torch.tensor(vocab.tag2id(inst.tag))
        mask[i,:seq_len].fill_(1)
    return word_idx,tag_idx,mask





