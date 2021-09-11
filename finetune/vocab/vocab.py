from dataloader.Dataloader import *
from collections import Counter
import  numpy as np
import pickle
class vocab:
    #写 法 二(不用建立字典形式的id2words，列表形式的即可，因为本质上列表已经给每个元素编好了号)
    # def __init__(self,words,tag):
    #     self._id2words = []
    #     self._wordfreq = []
    #     self.min_occur_time == 2
    #     for words,time in words.most_common():
    #         if time> self.min_occur_time:
    #             self._id2words.append(words)
    #             self._wordfreq.append(time)
    #
    #     reverse = lambda x:dict(zip(x,range(len(x))))
    #     self._word2id = reverse( self._id2words)
    #    self._id2word = {idx:words for words,idx in self.word2id}
    def __init__(self, word_counter, tag_counter, min_occur_count=2):
        self.UNK = 0
        self.word2freq = {word: count for word, count in word_counter.most_common() if count > min_occur_count}
        # for word,freq in self.word2freq.items():
        #     print(word,freq)

        self._word2id = {word: idx+1 for idx , word in enumerate(self.word2freq.keys())}
        self._word2id['<UNK>'] = self.UNK
        self._id2word = {idx: word for word, idx in self._word2id.items()}

        self._tag2id = {tag: idx for idx, tag in enumerate(tag_counter.keys())}
        self._id2tag = {idx: tag for tag, idx in self._tag2id.items()}

    #获取预训练的词向量
    def get_emdedding_weight(self,path):
        self._vec_tabs = {}
        self.vec_size = 0
        with open(path,'r',encoding='UTF-8')as file:
            for line in file.readlines():
                tokens = line.strip().split(" ")
                extword = tokens[0]
                vec = tokens[1:]
                self.vec_size = len(vec)
                self._vec_tabs[extword] =np.asarray(vec, dtype=np.float32)


        self._extword2id = {words:idx+1 for idx,words in enumerate(self._vec_tabs.keys())}
        self._extword2id['<UNK>'] = self.UNK
        self._extid2word = {idx: word for word, idx in  self._extword2id.items()}
        vocab_size = len(self._extword2id)
        embedding_weight = np.zeros((vocab_size,self.vec_size), dtype=np.float32)

        for word ,i in self._extword2id.items():
            if i !=self.UNK:
                embedding_weight[i] = self._vec_tabs[word]
        #给oov词附随机值embedding_weight[self.UNK] = np.random.uniform(-0.22,0.25,vec_size)
        embedding_weight[self.UNK] = np.mean(embedding_weight,0) / np.std(embedding_weight)
        print(type(embedding_weight))
        return embedding_weight
    #计算oov概率
        # count = 0
        # for words in self._word2id.keys():
        #     if words not  in self._extword2id.keys():
        #         count += 1
        # rate = count /(len(self._extword2id))
        # print("OOV:%.5f"%rate)

    def word2id(self, words):
        # 输入是一个词还是很多词
        if isinstance(words, list):
            return [self._word2id.get(x, self.UNK) for x in words]
        return self._word2id.get(words, self.UNK)

    def id2word(self, idx):

        if isinstance(idx, list):
            return [self._id2word.get(x) for x in idx]
        return self._id2word.get(idx)
    def extword2id(self,words):
        if isinstance(words,list):
            return [self._extword2id.get(x,self.UNK) for x in words]
        return self._extword2id.get(words,self.UNK)
    def extid2word(self,idx):
        if isinstance(idx,list):
            return [self._extid2word.get(x) for x in idx]
        return self._extid2word.get(idx)



    def tag2id(self, tag):
        if isinstance(tag, list):
            return [self._tag2id.get(x, -1) for x in tag]
        return self._tag2id.get(tag)

    def id2tag(self, idx):

        if isinstance(idx, list):
            return [self._id2tag.get(x) for x in idx]
        return self._id2tag.get(idx)

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def extwords_size(self):
        return len(self._extword2id)

    @property
    def words_size(self):
        return len(self._word2id)

    # @property
    # def vec_size(self):
    #     return  self.vec_size

    def save(self,path):
        with open(path,'wb')as fw:
            pickle.dump(self,fw)



def creat_vocab(path):
        words = Counter()
        tag = Counter()
        insts = Dataloader(path)
        #print(insts)
        for inst in insts :
            for curword in inst.words:
                #print(curword)
                words[curword] += 1
            #print(inst.tag)
            tag[inst.tag] += 1

        return vocab(words,tag)

