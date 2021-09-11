import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.ngram_baseline import NGramLanguageModeler
import random
from config.config import *
import datetime
import pickle
from data_utils import *
from GPT.tokenization import *
from collections import defaultdict
import re
from collections import Counter
class Instance:
    def __init__(self, input_ids, word_index, project, tags):
        self.input_ids = input_ids
        self.word_index = word_index
        self.project = project
        self.tags = tags
class Candidate:
    def __init__(self, pre_ids, pro, is_complete):
        self.pre_ids = pre_ids
        self.pro = pro
        self.is_complete = is_complete


class BestToken:
    def __init__(self, pre_ids, pro):
        self.pre_ids = pre_ids
        self.pro = pro
def build_vocab_emb(data_path):
    # opt = data_path_config(data_path)
    # data_path = opt["data"]["embedding_weight"]
    vec_tabs = {}
    vec_size = 0
    UNK = 0
    i = 0
    with open(data_path, 'r', encoding='UTF-8')as file:
        for line in file.readlines():
            # if i != 0:
            tokens = line.strip().split(" ")
            extword = tokens[0]
            vec = tokens[1:]
            vec_size = len(vec)
            vec_tabs[extword] = np.asarray(vec, dtype=np.float32)
            # i += 1


    api2id = {words: idx for idx, words in enumerate(vec_tabs.keys())}
    # api2id['<UNK>'] = UNK
    id2api = {idx: word for word, idx in api2id.items()}
    vocab_size = len(api2id)
    embedding_weight = np.zeros((vocab_size, vec_size), dtype=np.float32)

    for word, i in api2id.items():
        # if i != UNK:
        embedding_weight[i] = vec_tabs[word]
    # 给oov词附随机值embedding_weight[.UNK] = np.random.uniform(-0.22,0.25,vec_size)
    # embedding_weight[UNK] = np.mean(embedding_weight, 0) / np.std(embedding_weight)
    #print(type(embedding_weight))
    return embedding_weight,api2id,id2api
def data_variable(batch,lable2id,args,context_size):
    batch_size = len(batch)
    vec_tensor = torch.LongTensor(batch_size,context_size).zero_().to(device=args.device)
    tag = torch.LongTensor(batch_size).zero_().to(device=args.device)
    for i, inst in enumerate(batch):
        # print("wwww")
        # print(len(inst[0]))
        vec_tensor[i,:] = torch.tensor([lable2id.get(voc,0) for voc in inst[0]],dtype=torch.long)
        tag[i] = torch.tensor(lable2id.get(inst[-1],0),dtype=torch.long)
    return vec_tensor,tag
def data_variable_1(batch,lable2id,args,context_size):
    batch_size = len(batch)
    vec_tensor = torch.LongTensor(batch_size,context_size).zero_().to(device=args.device)
    tag = torch.LongTensor(batch_size).zero_().to(device=args.device)
    for i, inst in enumerate(batch):
        # print("wwww")
        # print(len(inst[0]))
        vec_tensor[i,:] = torch.tensor([ idx for idx in inst[0]],dtype=torch.long)
        tag[i] = torch.tensor(inst[-1],dtype=torch.long)
    return vec_tensor,tag
def get_batch(data,batch_size):
    batch_num = int(np.ceil(len(data) /batch_size))
    for i in range(batch_num):
        yield data[i*batch_size:(i+1)*batch_size]
def compuate_acc(true_tags, logit):
    # select_index = []
    logit = F.softmax(logit, dim=1)
    correct_num = 0

    for i in range(logit.shape[0]):
        if true_tags[i] in torch.argsort(logit[i], descending=True)[: 2]:
            correct_num += 1
    # 返回正确的item的数目,eq是返回一个矩阵，sum之后返回总数

    # return torch.eq(torch.argmax(logit, dim=1), true_tags).sum().item(), true_tags.shape[0]
    return correct_num
def validate(validate_data,model,vocab,args):
    dev_acc_num = 0
    total_loss = 0
    # 循环context上下文，比如：['When', 'forty']
    # target，比如：winters
    batch_num = 0
    loss_function = nn.NLLLoss()
    model.eval()
    for trigrams in get_batch(validate_data, batch_size=args.batch_size):
        words, tag = data_variable_1(trigrams, vocab, args, 4)
        # 步骤3：运行网络的正向传播，获得 log 概率
        log_probs = model(words)
        # 步骤4：计算损失函数
        # 传入 target Variable
        loss = loss_function(log_probs, tag)
        dev_acc_num += compuate_acc(tag, log_probs)
        total_loss += loss.data.item()
    print("LOSS: {} ACC: {}".format(total_loss / len(validate_data), dev_acc_num / len(validate_data)))
    return dev_acc_num / len(validate_data) ,total_loss / len(validate_data)
def make2ngram(data):
    ngram_data = []
    for seq in data:
        trigrams = [([seq.input_ids[i],seq.input_ids[i + 1]], seq.input_ids[i + 2]) for i in range(len(seq.input_ids) - 2)]
        for trm in trigrams:
            ngram_data.append(trm)
    return ngram_data
def finetune(train_data_all,model,vocab,args,Rvocab,search_word_dict):
    loss_function = nn.CrossEntropyLoss()
    # 优化函数使用随机梯度下降算法，学习率设置为0.001
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    test_data = defaultdict(list)
    train_data = []
    validate_data =[]
    random_api_word_top1 = 0.0
    random_api_word_top3 = 0.0
    random_api_word_top5 = 0.0
    random_api_word_top10 = 0.0
    random_domain_api_word_top1 = 0.0
    random_domain_api_word_top3 = 0.0
    random_domain_api_word_top5 = 0.0
    random_domain_api_word_top10 = 0.0
    project_api_word_top1 = defaultdict(list)
    project_api_word_top3 = defaultdict(list)
    project_api_word_top5 = defaultdict(list)
    project_api_word_top10 = defaultdict(list)
    project_domain_api_word_top1 = defaultdict(list)
    project_domain_api_word_top3 = defaultdict(list)
    project_domain_api_word_top5 = defaultdict(list)
    project_domain_api_word_top10 = defaultdict(list)
    Perplexity = 0
    if args.domain == "jdbc":
        test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes", "openGauss-connector-jdbc",
                    "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]

        for i, project in enumerate(test_project):
            print(project)
            for test_seq in train_data_all[project]:
                test_data[project].append(test_seq)
    # test_project = ["StoreManager","SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
    elif args.domain == "swing":
        test_project = ["java-swing-tips", "openhospital-gui", "School-Management-System", "def-guide-to-java-swing", "pumpernickel",
                        "swingx",
                        "swingBasic", "beautyeye", "joxy-oxygen", "java2script"]
        for i, project in enumerate(test_project):
            print(project)
            for test_seq in train_data_all[project]:
                test_data[project].append(test_seq)
    elif args.domain == "io":
        test_project = []
        for j in range(args.k):
            K_train_project, K_val_project = get_k_fold_data(args.k, j, project_list)
            K_val_project_order = order_test_set(K_val_project, train_data_all)
            for i, project in enumerate(K_val_project_order):
                if i == 0:
                    print(project[0], project[1])
                    test_project.append(project[0])
                    for test_seq in train_data_all[project[0]]:
                        if 1 in test_seq.input_ids:
                            continue
                        test_data[project[0]].append(test_seq)
    appendControlNodesString = [
        "IF", "CONDITION", "THEN", "ELSE",
        "WHILE", "BODY",
        "TRY", "TRYBLOCK", "CATCH", "FINALLY",
        "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
        "FOREACH", "VARIABLE", "ITERABLE",
    ]




    for project_name in project_list:
        if project_name not in test_project:
            for train_seq in train_data_all[project_name]:
                train_data.append(train_seq)
    train_data_1 = random.sample(train_data, int(len(train_data) * 0.8))
    dev_data = list(set(train_data).difference(set(train_data_1)))
    select_dev_data = select_dev_data_domain(dev_data)
    print("train data all num:", len(train_data_1))
    # print(test_data)
    for i in range(1):
        api_word_top1 = 0.0
        api_word_top3 = 0.0
        api_word_top5 = 0.0
        api_word_top10 = 0.0
        domain_api_word_top1 = 0.0
        domain_api_word_top3 = 0.0
        domain_api_word_top5 = 0.0
        domain_api_word_top10 = 0.0
        select_train_data = select_train_data_domain(train_data_1, args.sample)
        ngram_train_data = make2ngram(select_train_data)
        ngram_validate_data = make2ngram(select_dev_data)


        for project, project_test_data in test_data.items():
            print (project, len(project_test_data))

            if project in test_project:
                test_word_acc_1, test_word_acc_3, test_word_acc_5, test_word_acc_10, test_num, domain_acc_list, domain_count = evluate(model,
                    project_test_data,args.device, args,vocab,Rvocab,search_word_dict)
                print(
                    "project:{} test_word_acc_top1:{} test_word_acc_top3:{} test_word_acc_top5:{} test_word_acc_top10:{}".format(
                        project,
                        test_word_acc_1, test_word_acc_3, test_word_acc_5, test_word_acc_10))
                api_word_top1 += test_word_acc_1
                api_word_top3 += test_word_acc_3
                api_word_top5 += test_word_acc_5
                api_word_top10 += test_word_acc_10
                domain_api_word_top1 += domain_acc_list[0]
                domain_api_word_top3 += domain_acc_list[1]
                domain_api_word_top5 += domain_acc_list[2]
                domain_api_word_top10 += domain_acc_list[3]
                project_api_word_top1[project].append(test_word_acc_1)
                project_api_word_top3[project].append(test_word_acc_3)
                project_api_word_top5[project].append(test_word_acc_5)
                project_api_word_top10[project].append(test_word_acc_10)
                project_domain_api_word_top1[project].append(domain_acc_list[0])
                project_domain_api_word_top3[project].append(domain_acc_list[1])
                project_domain_api_word_top5[project].append(domain_acc_list[2])
                project_domain_api_word_top10[project].append(domain_acc_list[3])
        print("Perplexity:", Perplexity / 10)
        print("********************************************************")
        print("avg_word_acc_top1:{} avg_word_acc_top3:{} avg_word_acc_top5:{} avg_word_acc_top10:{}".format(
            api_word_top1 / args.k, api_word_top3 / args.k, api_word_top5 / args.k, api_word_top10 / args.k))
        print("********************************************************")
        print("domain setting")
        print("********************************************************")
        print(
            "domain_avg_word_acc_top1:{} domain_avg_word_acc_top3:{} domain_avg_word_acc_top5:{} domain_avg_word_acc_top10:{}".format(
                domain_api_word_top1 / args.k,
                domain_api_word_top3 / args.k,
                domain_api_word_top5 / args.k,
                domain_api_word_top10 / args.k))
        random_api_word_top1 += (api_word_top1 / args.k)
        random_api_word_top3 += (api_word_top3 / args.k)
        random_api_word_top5 += (api_word_top5 / args.k)
        random_api_word_top10 += (api_word_top10 / args.k)
        random_domain_api_word_top1 += (domain_api_word_top1 / args.k)
        random_domain_api_word_top3 += (domain_api_word_top3 / args.k)
        random_domain_api_word_top5 += (domain_api_word_top5 / args.k)
        random_domain_api_word_top10 += (domain_api_word_top10 / args.k)
    print("********************************************************")
    print("project setting")
    print("********************************************************")

    for project in test_project:
        avg_project_api_word_top1 = mean(project_api_word_top1[project])
        avg_project_api_word_top3 = mean(project_api_word_top3[project])
        avg_project_api_word_top5 = mean(project_api_word_top5[project])
        avg_project_api_word_top10 = mean(project_api_word_top10[project])
        avg_project_domain_api_word_top1 = mean(project_domain_api_word_top1[project])
        avg_project_domain_api_word_top3 = mean(project_domain_api_word_top3[project])
        avg_project_domain_api_word_top5 = mean(project_domain_api_word_top5[project])
        avg_project_domain_api_word_top10 = mean(project_domain_api_word_top10[project])
        print("********************************************************")
        print("Project:{} avg_word_acc_top1:{} avg_word_acc_top3:{} avg_word_acc_top5:{} avg_word_acc_top10:{}".format(
            project, avg_project_api_word_top1, avg_project_api_word_top3, avg_project_api_word_top5,
            avg_project_api_word_top10))
        print("********************************************************")
        print("domain setting")
        print("********************************************************")
        print(
            "Project:{} domain_avg_word_acc_top1:{} domain_avg_word_acc_top3:{} domain_avg_word_acc_top5:{} domain_avg_word_acc_top10:{}".format(
                project, avg_project_domain_api_word_top1,
                avg_project_domain_api_word_top3,
                avg_project_domain_api_word_top5,
                avg_project_domain_api_word_top10))

    print("********************************************************")
    print(
        "random_avg_word_acc_top1:{} random_avg_word_acc_top3:{} random_avg_word_acc_top5:{} random_avg_word_acc_top10:{}".format(
            random_api_word_top1 / 1, random_api_word_top3 / 1, random_api_word_top5 / 1, random_api_word_top10 / 1))
    print("********************************************************")
    print("domain setting")
    print("********************************************************")
    print(
        "random_domain_avg_word_acc_top1:{} random_domain_avg_word_acc_top3:{} random_domain_avg_word_acc_top5:{} random_domain_avg_word_acc_top10:{}".format(
            random_domain_api_word_top1 / 1,
            random_domain_api_word_top3 / 1,
            random_domain_api_word_top5 / 1,
            random_domain_api_word_top10 / 1))
    print("********************************************************")
    print("********************************************************")
    print("avg_acc:{} 平均收敛轮次:{} 困惑度:{}".format((ACC / args.k) * 100, EP / args.k, Perplexity / args.k))



def select_train_data_domain(train_set,sample,per=0.1):
    is_ready = True
    select_train_data = []
    store_data = []
    for seq in train_set:
        if 1 in seq.tags:
            select_train_data.append(seq)
    train_num = int(np.ceil(len(select_train_data) * sample))
    print("domain_seq_num:",len(select_train_data))
    train_data = random.sample(select_train_data, train_num)
    print (len(train_data))
    print ("----------------------------------------")
    for data in train_data:
        store_data.append(PretrainInputFeatures(data.input_ids))
    torch.save(store_data,"sample_swing_{}_111".format(sample))
    print (len(store_data))
    print("data stored")
    domain_count = 0
    # for seq in train_data:
    #     if 1 in seq.tags:
    #        domain_count += 1
    # if (per+0.1)>(domain_count / len(train_data))>per:
    #     is_ready = False
    return train_data

def select_dev_data_domain(dev_set,per=0.1):
    is_ready = True
    select_dev_data = []
    # store_data = []
    for seq in dev_set:
        if 1 in seq.tags:
            select_dev_data.append(seq)

    print("dev_domain_seq_num:",len(select_dev_data))


    return select_dev_data

def order_test_set(test_set,all_set,per=0.1):
    K_val_project_order = {}
    for test_project in test_set:
        domain_count = 0
        for seq in all_set[test_project]:
            if 1 in seq.tags:
                domain_count += seq.tags.count(1)
                # domain_count += 1
        K_val_project_order[test_project] = domain_count
    K_val_project_order = sorted(K_val_project_order.items(), key=lambda d: d[1], reverse=True)
    # K_val_project_order = K_val_project_order[:10]
    return K_val_project_order
def get_k_fold_data(k, i, X):
    # 此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    # assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）

    X_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part = X[idx]
        if j == i:  ###第i折作valid
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            # X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            X_train = X_train + X_part

    # print(X_train.size(),X_valid.size())
    return X_train, X_valid


def convert_examples_to_features(examples,
                                 args,
                                 ):
    # Create features
    features = defaultdict(list)
    features_1 = []
    project_list = []
    eos_token = "[EOS]"
    bos_token = "[BOS]"
    pad_token = "[PAD]"
    unk = 0
    all_word_num = 0
    pad_num = 0
    for i, example in enumerate(examples):
        tags = []
        tokens = example.split(" ")[:-1]
        locate = example.split(" ")[-1]
        project_list.append(locate)

        tokens = [bos_token] + tokens[:args.max_seq_len - 2] + [eos_token]  # BOS, EOS
        tokens += [pad_token] * (args.max_seq_len - len(tokens))
        refine_tokens = []
        for loc, token_1 in enumerate(tokens):
            refine_tokens.append(token_1.replace("#domain", "").replace("#nondomain", ""))
            if token_1 != "[EOS]" and token_1 != "[BOS]" and token_1 != "[PAD]":
                # print(token_1)
                if token_1.split("#")[1] == "domain":
                    tags.append(1)
                elif token_1.split("#")[1] == "nondomain":
                    tags.append(0)


        features[locate].append(Instance(refine_tokens,None,locate,tags))

    project_list_1 = list(set(project_list))
    project_list_1.sort(key=project_list.index)
    return features, project_list_1
# def convert_examples_to_features_pre(examples,
#                                  args,
#                                  ):
#     # Create features
#     features = list()
#     eos_token = "[EOS]"
#     bos_token = "[BOS]"
#     pad_token = "[PAD]"
#     for i, example in enumerate(examples):
#         tokens = example.split(" ")
#         tokens = [bos_token] + tokens[:args.max_seq_len - 2] + [eos_token]  # BOS, EOS
#         tokens += [pad_token] * (args.max_seq_len - len(tokens))
#
#
#
#         features.append(Instance(tokens,None,None,None))
#
#
#     return features


def create_examples(train_corpus, args):
    # Load data features from cache or dataset file
    corpus_path = train_corpus
    with open(corpus_path, 'r', encoding='utf-8') as reader:
        corpus = reader.readlines()
    corpus = list(map(lambda x: x.strip(), corpus))
    corpus = list(filter(lambda x: len(x) > 0, corpus))
    examples = [text for text in corpus]
    features, project_list = convert_examples_to_features(examples, args)


    return features,project_list

# def create_examples_pre(train_corpus, args):
#     # Load data features from cache or dataset file
#
#     corpus_path = train_corpus
#     with open(corpus_path, 'r', encoding='utf-8') as reader:
#         corpus = reader.readlines()
#     corpus = list(map(lambda x: x.strip(), corpus))
#     corpus = list(filter(lambda x: len(x) > 0, corpus))
#     examples = [text for text in corpus]
#     features = convert_examples_to_features_pre(examples, args)
#     # print(len(features))
#
#
#     return features

def compuate_acc_2(logit, pre_info, k, reject_token, search_dict, class_name, control_lable, Rvocab):
    bestTokens = []
    pre_candidate = []
    logit = F.softmax(logit, dim=1)
    sort = torch.argsort(logit, dim=1, descending=True)

    for j in range(logit.shape[1]):
        if len(bestTokens) < k:
            method_name = Rvocab.get(sort[0][j].item())

            if class_name == "":
                if method_name in control_lable:
                    bestTokens.append(BestToken([sort[0][j].item()], logit[0][sort[0][j].item()].item()))
            else:

                if method_name in reject_token:
                    continue
                else:
                    if method_name.startswith(class_name):
                        bestTokens.append(BestToken([sort[0][j].item()], logit[0][sort[0][j].item()].item()))
        else:
            # print (len(bestTokens))
            # print (11111)
            # print (flag1)
            break

            # pre_candidate.append(sort[-1][j].item() % 4000)
    bestTokens = sorted(bestTokens, key=lambda x: x.pro, reverse=True)
    # print(lowest_pro)
    # print(len(bestTokens))
    return pre_candidate, bestTokens

def batch_numberize(batch,vocab,device,arg):
    # if arg.mode == "train":
    batch_size = len(batch)
    seq_lengths = []
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
        # print(inst.seq)
        # seq_lengths.append(len(inst.seq))
        seq_lengths.append(len(inst.input_ids))
        # seq_len = len(inst.seq)
        seq_len = len(inst.input_ids)
        # try:
        # apiseq_idx[i, :seq_len] = torch.tensor([vocab.get(i,1) for i in inst.seq])
        # print ([i for i in inst.input_ids])
        apiseq_idx[i, :seq_len] = torch.tensor([vocab.get(i,0) for i in inst.input_ids])
        # except:
        #     print([i for i in inst.seq])
        # tag_idx[i] = torch.tensor(np.asarray(inst.tag))
        mask[i,:seq_len].fill_(1)

        return apiseq_idx,mask

def evluate(model,dev_data, args_device, arg , vocab,Rvocab, search_word_dict):
    reject_token = ["[EOS]", "[BOS]", "[PAD]", "[UNK]"]
    appendControlNodesStrings = [
        "IF", "CONDITION", "THEN", "ELSE",
        "WHILE", "BODY",
        "TRY", "TRYBLOCK", "CATCH", "FINALLY",
        "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
        "FOREACH", "VARIABLE", "ITERABLE",
    ]

    control_node = 0
    k = 10
    beam_size = arg.boundary
    batch_num = 0
    dev_word_acc_top1 = 0
    dev_word_acc_top10 = 0
    dev_word_acc_top3 = 0
    dev_word_acc_top5 = 0
    dev_word_acc_class_1 = 0
    dev_word_acc_class_3 = 0
    dev_word_acc_class_5 = 0
    dev_word_acc_class_10 = 0
    # non-control-node-num
    num_1 = 1
    # control-node-num
    num_2 = 0
    # coreect_non-control-node-num
    num_3 = 0
    num_4 = 0
    num_5 = 0
    new_num = 0
    model.eval()
    dev_num = 1
    refine_num = 0
    no_refine_num = 0
    domain_count = 1
    distance = []
    for line_num, onebatch in enumerate(get_batch(dev_data, 1)):
        batch_num += 1
        print("*********new_seq***********")

        words,  mask = batch_numberize(onebatch, vocab ,args_device, arg)
        targets = words[:, 1:].contiguous()
        # true_seq = " ".join([Rvocab.get(s) for s in onebatch[0].input_ids]).replace("▁", "")
        # print (true_seq)
        pred_index_1 = 0
        pred_index = 0
        for word_loc, api in enumerate(onebatch[0].input_ids):
            if Rvocab.get(targets[0, word_loc:word_loc + 1].item()) == "[EOS]":
                break
            candidate_list = []
            bestToken_list = []
            beam_candidate_list = []
            tokensDone = 0
            iter = 0
            count = 0
            # if The probability of the best candidate is less than the worst current complete top-k tokens
            true_token = []
            if word_loc == 0:
                true_token1 = Rvocab.get(targets[0, :1].item())

                print("number one word", true_token1)
                continue
            if word_loc <= 1:
                continue
            true_token.append(Rvocab.get(targets[0, word_loc:word_loc + 1].item()))
            true_api = "".join(true_token)
            # print(true_api)
            if true_api.find(".new") != -1:
                if onebatch[0].tags[word_loc] == 1:
                    new_num += 1
                continue
            dev_num += 1
            # print (onebatch[0].tags)
            # print(word_loc)
            if true_api == "UNK":
                if onebatch[0].tags[word_loc] == 1:
                    domain_count += 1
                continue
            if onebatch[0].tags[word_loc] == 1:
                domain_count += 1
                if true_api == "UNK":
                    continue

            class_name = true_api.split(".")[0]
            currt_pred = model(words[:1, word_loc - 2:word_loc])
            init_candidate_list, init_bestTokens = compuate_acc_2(
                currt_pred, None, k, reject_token,
                search_word_dict, class_name, appendControlNodesStrings, Rvocab)

            bestToken_list = init_bestTokens
            final_result = []
            final_result_check = []
            final_result_nop = []
            final_class_result = []

            bestToken_list = sorted(bestToken_list, key=lambda x: x.pro, reverse=True)

            for best_token in bestToken_list[:10]:
                # print("".join([Rvocab.get(index) for index in best_token.pre_ids]).replace("▁", ""))
                final_result.append("".join(
                    [Rvocab.get(index) for index in best_token.pre_ids]))

            if true_api in final_result[:1]:
                dev_word_acc_top1 += 1
                if onebatch[0].tags[word_loc] == 1:
                    dev_word_acc_class_1 += 1
            # if true_api.replace("</t>","").split(".")[0] in final_class_result[:1]:
            #     dev_word_acc_class_1 += 1
            else:
                if true_api.replace("</t>", "") not in appendControlNodesStrings:
                    pass

                else:
                    # 未在top1的control node的数量
                    control_node += 1
                    pass
            if true_api.replace("</t>", "") in final_result[:3]:
                dev_word_acc_top3 += 1
                if onebatch[0].tags[word_loc] == 1:
                    dev_word_acc_class_3 += 1
            else:
                m3 = 0
                # if true_api.replace("</t>", "") not in appendControlNodesStrings:
                #     for i in range(3):
                #         m3 += len(bestToken_list[i].pre_ids)
                #     m3 /= 3
                #     top3_len[m3] += 1
                #     top3_ground_true_len[method_len] += 1
                # else:
                #     pass
            # if true_api.replace("</t>", "").split(".")[0] in final_class_result[:3]:
            #     dev_word_acc_class_3 += 1
            if true_api.replace("</t>", "") in final_result[:5]:
                dev_word_acc_top5 += 1
                if onebatch[0].tags[word_loc] == 1:
                    dev_word_acc_class_5 += 1
            # if true_api.replace("</t>", "").split(".")[0] in final_class_result[:5]:
            #     dev_word_acc_class_5 += 1
            if true_api.replace("</t>", "") in final_result:
                if true_api.replace("</t>", "") not in appendControlNodesStrings:
                    num_3 += 1
                else:
                    num_4 += 1
                dev_word_acc_top10 += 1
                if onebatch[0].tags[word_loc] == 1:
                    # print ("--------------------------")
                    # print ("domain:")
                    # print (true_api)
                    # # print (class_name_var)
                    # print (word_loc)
                    # print (Rvocab.get(words[:,word_loc:word_loc+1].item()))
                    # print (onebatch[0].tags)
                    # print (onebatch[0].input_ids)
                    # # print (Rvocab.get(pred[0, word_loc:word_loc + 1, :].item()))
                    # print ("--------------------------")
                    dev_word_acc_class_10 += 1
                distance.append(word_loc)
            else:
                if true_api.replace("</t>", "") not in appendControlNodesStrings:
                    num_1 += 1
                    

                else:
                    num_2 += 1
        
    # domain_count -= new_num
    word_acc_1 = dev_word_acc_top1 / dev_num
    word_acc_3 = dev_word_acc_top3 / dev_num
    word_acc_5 = dev_word_acc_top5 / dev_num
    word_acc_10 = dev_word_acc_top10 / dev_num
    dev_1 = dev_word_acc_class_1 / domain_count
    dev_3 = dev_word_acc_class_3 / domain_count
    dev_5 = dev_word_acc_class_5 / domain_count
    dev_10 = dev_word_acc_class_10 / domain_count
    print("非控制结构:", num_3, "控制结构", num_4)
    print("非参数错误：", num_5)
    print("非控制结构错误:", num_1, "控制结构错误", num_2)
    print("class acc: top1:{}   top3:{}    top5:{}    top10:{}".format(dev_1, dev_3, dev_5, dev_10))
    # print(top1_len)
    # print(top1_ground_true_len)
    # print(top3_len)
    # print(top3_ground_true_len)
    # print("------------------------------")
    # print("check reason")
    # print(length_pro)
    # print(length_pro_1)
    # print(length_pro_2)
    # print(dev_word_acc_top1 * dev_num - control_node)
    # print(control_node)
    # print(top1_len_info)
    print("------------------------------")
    print(dev_num)
    print(domain_count)
    print(sum(distance) / len(distance))
    return word_acc_1, word_acc_3, word_acc_5, word_acc_10, dev_num, [dev_1, dev_3, dev_5, dev_10], domain_count
def build_vocab(data):
    vec_tabs = Counter()
    for line in data:
        # if i!=0:
        for token in line:
            vec_tabs[token] += 1

    api2id = {words[0]: idx + 1 for idx, words in enumerate(vec_tabs.most_common(n=30000))}

    api2id["UNK"] = 0
    id2api = {idx: word for word, idx in api2id.items()}

    return api2id, id2api
def build_vocab_select(data):
    vec_tabs = Counter()
    for line in data:
        # if i!=0:
        for token in line.input_ids:
            vec_tabs[token] += 1

    api2id = {words[0]: idx + 1 for idx, words in enumerate(vec_tabs.most_common(n=30000))}

    api2id["UNK"] = 0
    id2api = {idx: word for word, idx in api2id.items()}
    return api2id, id2api
if __name__ == "__main__":
    opts = data_path_config('config/api_data_path.json')
    args = arg_config()
    '''
    ----------------------------------pretrain process---------------------------------------
    '''
    if args.mode == 'pretrain':
        seq_data = []
        test_data = defaultdict(list)
        train_data = []
        if args.domain == "swing":
            with open("data/API/data/all_remove_domain_clean_new.txt","r",encoding="utf-8")as f:
                for line in f.readlines():
                    seq_data.append(line.strip().split(" "))

            train_data_all, project_list = create_examples("data/API/data/swing_seq_clean_oneseq.txt", args)


            test_project = ["java-swing-tips", "openhospital-gui", "School-Management-System", "def-guide-to-java-swing", "pumpernickel",
                            "swingx",
                            "swingBasic", "beautyeye", "joxy-oxygen", "java2script"]
            for i, project in enumerate(test_project):
                print(project)
                for test_seq in train_data_all[project]:
                    test_data[project].append(test_seq)
        elif args.domain == "jdbc":
            with open("data/API/data/all_remove_domain_clean_new.txt", "r",
                      encoding="utf-8")as f:
                for line in f.readlines():
                    seq_data.append(line.strip().split(" "))

            train_data_all, project_list = create_examples("data/API/data/jdbc_clean_oneseq.txt", args)
            test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes", "openGauss-connector-jdbc",
                          "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
            for i, project in enumerate(test_project):
                print(project)
                for test_seq in train_data_all[project]:
                    test_data[project].append(test_seq)
        elif args.domain == "io":
            with open("data/API/data/all_remove_domain_io.txt", "r",
                      encoding="utf-8")as f:
                for line in f.readlines():
                    seq_data.append(line.strip().split(" "))
            train_data_all, project_list = create_examples("data/API/data/netio_seq.txt", args)
            test_project = []
            for j in range(args.k):
                K_train_project, K_val_project = get_k_fold_data(args.k, j, project_list)
                K_val_project_order = order_test_set(K_val_project, train_data_all)
                for i, project in enumerate(K_val_project_order):
                    if i == 0:
                        print(project[0], project[1])
                        test_project.append(project[0])
                        for test_seq in train_data_all[project[0]]:
                            if 1 in test_seq.input_ids:
                                continue
                            test_data[project[0]].append(test_seq)

        for project_name in project_list:
            if project_name not in test_project:
                for train_seq in train_data_all[project_name]:
                    train_data.append(train_seq)
        train_data_1 = random.sample(train_data, int(len(train_data) * 0.8))
        select_train_data = select_train_data_domain(train_data_1, args.sample)
        # select_train_data = random.sample(select_train_data, int(len(select_train_data) * args.sample))
        select_train_data_ngram = make2ngram(select_train_data)
        trigrams_seq = []
        #buildling voca------------------b
        vocab, Rvocab = build_vocab(seq_data)
        vocab_1, Rvocab_1 = build_vocab_select(select_train_data)
        # vocab.update(vocab_1)
        for k,v in vocab_1.items():
            if k in list(vocab.keys()):
                continue
            else:
                vocab.update({k:len(vocab.keys())})

        with open("vocab_{}_{}_60".format(args.domain,args.sample),"wb")as f1:
            pickle.dump(vocab,f1)
        # buildling voca------------------b

        for seq in seq_data:
            trigrams = [ ([seq[i], seq[i+1]], seq[i+2]) for i in range(len(seq) - 2) ]
            for trm in trigrams:
                trigrams_seq.append(trm)
        trigrams_seq = trigrams_seq+select_train_data_ngram
        # print(len(trigrams_seq[-1]))
        print("data_size:",len(trigrams_seq))
        print("data loaded")
        if torch.cuda.is_available():
            if args.device_index == -1:
                print("device:cpu")

            else:
                args.device = torch.device("cuda:{}".format(args.device_index))

        CONTEXT_SIZE = 2
        # 嵌入维度
        EMBEDDING_DIM = args.hidden_size
        losses = []

        loss_function = nn.CrossEntropyLoss()

        model = NGramLanguageModeler(len(vocab), CONTEXT_SIZE,args).to(args.device)

        print (len(vocab))
        print("model created")

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        # trigrams_seq = trigrams_seq[:20000]
        batch_size = args.batch_size
        init_loss = 10000
        for epoch in range(args.epoch):
            s_time = datetime.datetime.now()
            model.train()
            acc_num = 0
            total_loss = 0

            batch_num = 0
            for trigrams in get_batch(trigrams_seq,batch_size=batch_size):
                words, tag = data_variable(trigrams,vocab,args,CONTEXT_SIZE)
                model.zero_grad()
                # print(words.shape)
                # 步骤3：运行网络的正向传播，获得 log 概率
                # print(words.shape)
                log_probs = model(words)

                # 步骤4：计算损失函数
                # 传入 target Variable
                loss = loss_function(log_probs, tag)

                # 步骤5：进行反向传播并更新梯度
                loss.backward()
                optimizer.step()
                acc_num += compuate_acc(tag,log_probs)
                total_loss += loss.data.item()
            total_loss = total_loss / len(trigrams_seq)

            print("LOSS: {} ACC: {}  TIME：{}".format(total_loss ,acc_num /len(trigrams_seq),datetime.datetime.now()-s_time))
            #验证
            if total_loss < init_loss:
                init_loss = total_loss
            else:
                print("train complete")
                break
            losses.append(total_loss)
            torch.save(model,f="data/API/pretrain/ngram/{}/pretrain_ngram_train_model_{}_baseline_{}".format(args.domain,epoch,args.sample))
        print('pretrain process Finished')
        '''
            ----------------------------------pretrain process---------------------------------------
        '''
    elif args.mode == "fine_tune":
        '''
               ----------------------------------fine tune and recommendation process---------------------------------------
        '''
        if torch.cuda.is_available():
            if args.device_index == -1:
                print("device:cpu")

            else:
                args.device = torch.device("cuda:{}".format(args.device_index))
        # embedding_weight, vocab, Rvocab = build_vocab_emb(opts["data"]["embedding_weight"])
        with open("vocab_{}_{}_60".format(args.domain, args.sample), "rb")as f2:
            # pickle.dump(vocab, f2)
            vocab = pickle.load(f2)
        Rvocab = {idx: word for word, idx in vocab.items()}
        pretrained_model = torch.load("data/API/pretrain/ngram/{}/pretrain_ngram_train_model_15_baseline_{}".format(args.domain,args.sample)).to(args.device)
        with open("data/API/data/classmethod_search_list", "rb") as f3:
            search_word_dict = pickle.load(f3)
        # 将单词序列转化为数据元组列表，
        # 其中的每个元组格式为([ word_i-2, word_i-1 ], target word)
        if args.domain == "jdbc":
            train_data_all, project_list = create_examples("data/API/data/jdbc_clean_oneseq.txt", args)
        if args.domain == "swing":
            train_data_all, project_list = create_examples("data/API/data/swing_seq_clean_oneseq.txt", args)
        if args.domain == "io":
            train_data_all, project_list = create_examples("data/API/data/netio_seq.txt", args)
        # train_data_all_1, project_list_1 = create_examples(opts['data']['raw_seq_data_1'], tokenizer, args)
        finetune(train_data_all, model = pretrained_model,vocab=vocab,args=args,Rvocab=Rvocab,search_word_dict=search_word_dict)
