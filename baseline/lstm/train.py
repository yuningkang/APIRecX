from dataloader.DataloaderApi import *
from config.config import *
import torch
from vocab.vocab import *
from model.CNNmodel import *
from classifier import *
from build_train_data import *
from model.bilstm1 import *
import random
from torch.backends import cudnn
import numpy as np
import datetime
from collections import defaultdict
from data_utils import *
from GPT.tokenization import *
from collections import Counter
import os
import pickle
class Instance:
    def __init__(self, input_ids, word_index, project, tags):
        self.input_ids = input_ids
        self.word_index = word_index
        self.project = project
        self.tags = tags
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

    # for data in train_data:
    #     store_data.append(PretrainInputFeatures(data.input_ids))
    # torch.save(store_data,"sample_jdbc_{}".format(sample))
    # print("data stored")
    domain_count = 0
    # for seq in train_data:
    #     if 1 in seq.tags:
    #        domain_count += 1
    # if (per+0.1)>(domain_count / len(train_data))>per:
    #     is_ready = False
    return train_data



def order_test_set(test_set,all_set,per=0.1):
    K_val_project_order = {}
    for test_project in test_set:
        domain_count = 0
        for seq in all_set[test_project]:
            if 1 in seq.tags:
                domain_count += seq.tags.count(1)
        K_val_project_order[test_project] = domain_count
    K_val_project_order = sorted(K_val_project_order.items(), key=lambda d: d[1], reverse=True)
    # K_val_project_order = K_val_project_order[:10]
    return K_val_project_order
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
def convert_examples_to_features_pre(examples,
                                 args,
                                 ):
    # Create features
    features = list()
    eos_token = "[EOS]"
    bos_token = "[BOS]"
    pad_token = "[PAD]"
    for i, example in enumerate(examples):
        tokens = example.split(" ")
        tokens = [bos_token] + tokens[:args.max_seq_len - 2] + [eos_token]  # BOS, EOS
        tokens += [pad_token] * (args.max_seq_len - len(tokens))



        features.append(Instance(tokens,None,None,None))


    return features


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

def create_examples_pre(train_corpus, args):
    # Load data features from cache or dataset file

    corpus_path = train_corpus
    with open(corpus_path, 'r', encoding='utf-8') as reader:
        corpus = reader.readlines()
    corpus = list(map(lambda x: x.strip(), corpus))
    corpus = list(filter(lambda x: len(x) > 0, corpus))
    examples = [text for text in corpus]
    features = convert_examples_to_features_pre(examples, args)
    # print(len(features))


    return features


def get_k_fold_data(k, i, X):
    # 此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
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
def get_parameter_number(net):
    for p in net.parameters():
        print(p.numel())
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
def select_dev_data_domain(dev_set,per=0.1):
    is_ready = True
    select_dev_data = []
    # store_data = []
    for seq in dev_set:
        if 1 in seq.tags:
            select_dev_data.append(seq)

    print("dev_domain_seq_num:",len(select_dev_data))
    return select_dev_data
def build_vocab(all_data):
    vec_size = 0
    UNK = 0
    i = 0
    vec_tabs = Counter()
    for line in all_data:
        # if i!=0:
        for token in line.input_ids:
            vec_tabs[token] += 1


    api2id = {words[0]: idx+1 for idx, words in enumerate(vec_tabs.most_common(n=30000))}


    api2id["UNK"] = 0
    id2api = {idx: word for word, idx in api2id.items()}
    return api2id,id2api

if __name__ == '__main__':
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    dev_data = []
    # dev_index = []
    # train_data = []
    ACC = 0
    Perplexity = 0
    EP = 0
    train_data = []
    test_data = []
    # torch.cuda.maual_seed(666)
    opts = data_path_config('config/api_data_path.json')
    args = arg_config()

    train_data_all, project_list = create_examples(opts['data']['train_file'], args)

    pretrain_data= create_examples_pre(opts['data']['raw_seq_data'],args)

    print("-----------------")
    print(args.lr)
    print(args.batch_size)
    print(args.hidden_size)
    print("-----------------")
    with open("data/API/data/classmethod_search_list","rb") as f3:
      search_word_dict = pickle.load(f3)
    if torch.cuda.is_available():
        if args.device_index == -1:
            print("device:cpu")

        else:
            args.device = torch.device("cuda:{}".format(args.device_index))
    test_data = defaultdict(list)
    dev_data = []
    train_data = []
    # test_project = []
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
    # test_project = []
    # for j in range(args.k):
    #     K_train_project, K_val_project = get_k_fold_data(args.k, j, project_list)
    #     K_val_project_order = order_test_set(K_val_project, train_data_all)
    #     for i, project in enumerate(K_val_project_order):
    #         if i == 0:
    #             print(project[0], project[1])
    #             test_project.append(project[0])
    #             for test_seq in train_data_all[project[0]]:
    #                 test_data[project[0]].append(test_seq)
    test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes", "openGauss-connector-jdbc",
                  "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]

    # test_project = ["java-swing-tips", "openhospital-gui", "School-Management-System", "def-guide-to-java-swing", "pumpernickel",
    #                 "swingx",
    #                 "swingBasic", "beautyeye", "joxy-oxygen", "java2script"]
    for i, project in enumerate(test_project):
        print(project)
        for test_seq in train_data_all[project]:
            test_data[project].append(test_seq)
    for project_name in project_list:
        if project_name not in test_project:
            for train_seq in train_data_all[project_name]:
                train_data.append(train_seq)
    # for project_name in project_list_1:
    #     if project_name not in test_project:
    #         for train_seq in train_data_all_1[project_name]:
    #             train_data.append(train_seq)
    # train_data_1 = random.sample(train_data, int(len(train_data) * 0.8))
    # dev_data = list(set(train_data).difference(set(train_data_1)))
    train_data_1 = random.sample(train_data, int(len(train_data) * 0.8))
    # print(train_data_1[0].input_ids)
    dev_data = list(set(train_data).difference(set(train_data_1)))
    # print(dev_data[0].input_ids)
    # select_dev_data = select_dev_data_domain(dev_data)
    print("train data all num:", len(train_data_1))
    select_train_data = select_train_data_domain(train_data_1, args.sample)
    # select_train_data = random.sample(train_data_1,int(len(train_data_1)*args.sample))
    all_data = pretrain_data + select_train_data
    # vocab, Rvocab = build_vocab(pretrain_data)
    # # if os.path.exists("vocab_sample_{}".format(args.sample)):
    # #     with open("vocab_sample_{}".format(args.sample), "rb")as f4:
    # #         vocab = pickle.load(f4)
    # # else:
    # #
    # #     with open("vocab_sample_{}".format(args.sample),"wb")as f4:
    # #         pickle.dump(vocab,f4)
    # # if os.path.exists("Rvocab_sample_{}".format(args.sample)):
    # #     with open("Rvocab_sample_{}".format(args.sample), "rb")as f5:
    # #         Rvocab = pickle.load(f5)
    # # else:
    # #     vocab, Rvocab = build_vocab(pretrain_data)
    # #     with open("Rvocab_sample_{}".format(args.sample),"wb")as f5:
    # #         pickle.dump(Rvocab,f5)
    # vocab_1, Rvocab_1 = build_vocab(select_train_data)
    # # vocab.update(vocab_1)
    # for k,v in vocab_1.items():
    #     if k in list(vocab.keys()):
    #         continue
    #     else:
    #         vocab.update({k:len(vocab.keys())})
    # # print(len(vocab))
    # # # # print("-------")
    # Rvocab = {idx:word for word,idx in vocab.items()}
    with open("vocab_jdbc_0.01_60","rb")as f:
        vocab = pickle.load(f)
    Rvocab = {idx:word for word,idx in vocab.items()}
    # # # with open("Rocab_io_1","wb")as f1:
    #     Rvocab = pickle.load(f1)
    # with open("vocab_jdbc_{}_60".format(args.sample),"wb")as f:
    #     pickle.dump(vocab,f)
    # with open("Rocab_swing_1","wb")as f1:
    #     pickle.dump(Rvocab,f1)
    # print(len(vocab))
    print(len(Rvocab))
    # print(vocab.get("[PAD]"))
    pad_id = vocab.get("[PAD]")
    print(pad_id)
    if args.mode == "train":
        start_time = datetime.datetime.now()
        for i in range(1):
            api_word_top1 = 0.0
            api_word_top3 = 0.0
            api_word_top5 = 0.0
            api_word_top10 = 0.0
            domain_api_word_top1 = 0.0
            domain_api_word_top3 = 0.0
            domain_api_word_top5 = 0.0
            domain_api_word_top10 = 0.0

            # select_train_data = select_train_data_domain(train_data_1, args.sample)
            # select_train_data = random.sample(train_data_1,int(len(train_data_1)*args.sample))
            # train_sample_domain_count = 0
            # train_sample_count = 1
            # test_sample_domain_count = 0
            # test_sample_count = 1
            # print("train data num:", len(select_train_data))
            # for seq in select_train_data:
            #     # print(seq.input_ids)
            #     train_sample_domain_count += seq.tags.count(1)
            #     train_sample_count += len(seq.input_ids)

            # for project, seq in test_data.items():
            #     for one_seq in seq:
            #         test_sample_domain_count += one_seq.tags.count(1)
            #         test_sample_count += len(one_seq.input_ids)
            # print(test_sample_domain_count)
            # print("test_domain_percent:", test_sample_domain_count / test_sample_count)
            # model_lstm = None
            # classifier = Classifier(ft_model, model_lstm, args, tokenizer.vocab, word_vocab, None, tokenizer)
            # model_lstm = BiLSTM(args, vocab, embedding_weight).to(args.device)
            # # model_lstm
            # print(model_lstm)
            # print(get_parameter_number(model_lstm))
            model_lstm = torch.load("data/API/pretrain/jdbc/trained_model_14_jdk_128_word_sample_0.01_1e-3_2layer")

            model_lstm.to(args.device)
            print(model_lstm)
            classifier = Classifier(model_lstm, args, vocab, pad_id, Rvocab)
            print("train data num:", len(select_train_data))
            # acc, perplexitys, ep = classifier.train(select_train_data, select_dev_data, test_data, args.device, args, i,
            #                                        search_word_dict)
            # ACC += acc
            # # Perplexity += perplexity
            # EP += ep
            # sorted(test_data.items(), key=lambda d: len(d[1]), reverse=False)
            # escap_list = ["java-swing-tips", "openhospital-gui", "j2se_for_android", "weblaf", "pumpernickel", "swingx",
            #               "swingBasic", "beautyeye", "littleluck", "java2script"]

            for project, project_test_data in test_data.items():
                print (project, len(project_test_data))
                if project in test_project:
                    dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity = classifier.validate(project_test_data,
                                                                                                       args.device,
                                                                                                       args,
                                                                                                       None,
                                                                                                       True)
                    print(project,perplexity)
                    Perplexity += perplexity
                    test_word_acc_1, test_word_acc_3, test_word_acc_5, test_word_acc_10, test_num, domain_acc_list, domain_count = classifier.evluate(
                        project_test_data, args.device, args, None, True, search_word_dict)
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
            print("Perplexity:",Perplexity  / 10)
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
            print(
                "Project:{} avg_word_acc_top1:{} avg_word_acc_top3:{} avg_word_acc_top5:{} avg_word_acc_top10:{}".format(
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
                random_api_word_top1 / 1, random_api_word_top3 / 1, random_api_word_top5 / 1,
                random_api_word_top10 / 1))
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
        print("********************************************************")

    elif args.mode == "pretrain":
        print(len(vocab))
        model_lstm = BiLSTM(args, vocab, None).to(args.device)
        # model_lstm = torch.load("data/API/pretrain/trained_model_9_jdk_300_word_sample_0.05_5e_128").to(args.device)
        print(model_lstm)
        classifier = Classifier(model_lstm, args, vocab, pad_id, Rvocab)
        classifier.pretrain(all_data,args.device, args)













