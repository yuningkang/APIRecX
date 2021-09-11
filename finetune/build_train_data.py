import json
from config.config import data_path_config
import numpy as np
from collections import defaultdict
import pickle
import datetime
def build_vocab_emb(data_path):
    # opt = data_path_config(data_path)
    # data_path = opt["data"]["embedding_weight"]
    vec_tabs = {}
    vec_size = 0
    UNK = 0
    i = 0
    with open(data_path, 'r', encoding='UTF-8')as file:
        for line in file.readlines():

            tokens = line.strip().split(" ")
            extword = tokens[0]
            vec = tokens[1:]
            vec_size = len(vec)
            vec_tabs[extword] = np.asarray(vec, dtype=np.float32)


    api2id = {words: idx + 1 for idx, words in enumerate(vec_tabs.keys())}
    api2id['<UNK>'] = UNK
    id2api = {idx: word for word, idx in api2id.items()}
    vocab_size = len(api2id)
    embedding_weight = np.zeros((vocab_size, vec_size), dtype=np.float32)

    for word, i in api2id.items():
        if i != UNK:
            embedding_weight[i] = vec_tabs[word]
    # 给oov词附随机值embedding_weight[self.UNK] = np.random.uniform(-0.22,0.25,vec_size)
    embedding_weight[UNK] = np.mean(embedding_weight, 0) / np.std(embedding_weight)
    #print(type(embedding_weight))
    return embedding_weight,api2id,id2api

def build_train_data(data_path,vocab):
    opt = data_path_config(data_path)
    raw_data_path = opt["data"]["raw_seq_data"]
    UNK = 0
    seq2id = []
    train_data = defaultdict(list)
    with open(raw_data_path, "r", encoding="utf-8")as file:
        for inst in file.readlines():
            seq = inst.strip().split(" ")
            if isinstance(seq,list):
                seq2id.append([vocab.get(api,UNK) for api in seq])
            else:
                seq2id.append(vocab.get(seq,UNK))
    for k,v in vocab.items():
        for oneseq in seq2id:
            for i,id in enumerate(oneseq):
                if id == v and len(oneseq) < 50:
                    train_data[(v,i)].append(oneseq)
    datetime.datetime.now()
    with open("API/train_data","wb")as file:
        pickle.dump(train_data,file)
    print(111)


if __name__ == "__main__":
    data_path = "API/embedding_api_256dd.txt"
    embedding_weight,vocab,Rvocab = build_vocab_emb(data_path)
    data_path1= "../config/api_data_path.json"
    build_train_data(data_path1,vocab)
