from dataloader.DataloaderApi import *
from config.config import *
import torch
from vocab.vocab import *
from Classifier6 import *
from data.build_train_data import *
from data_utils import *
from GPT.tokenization import *
import random
from lstm.bilstm1 import *
from torch.backends import cudnn
import numpy as np
from model import GPT, GPTLMHead, GPTClsHead
from collections import defaultdict
import pickle
from statistics import mean
from pandas import DataFrame
from datetime import datetime
import os


class Instance:
    def __init__(self, seq, index, tag):
        self.seq = seq
        self.index = index
        self.tag = tag


def select_train_data_domain(train_set, sample, per=0.1):
    is_ready = True
    select_train_data = []
    store_data = []
    for seq in train_set:
        if 1 in seq.tags:
            select_train_data.append(seq)
    train_num = int(np.ceil(len(select_train_data) * sample))
    print("domain_seq_num:", len(select_train_data))
    train_data = random.sample(select_train_data, train_num)
    print(len(train_data))
    print("----------------------------------------")
    # for data in train_data:
    #     store_data.append(PretrainInputFeatures(data.input_ids))
    # torch.save(store_data, "sample_swing_{}_111".format(sample))
    # print(len(store_data))
    # print("data stored")

    return train_data


def select_dev_data_domain(dev_set, per=0.1):
    is_ready = True
    select_dev_data = []
    # store_data = []
    for seq in dev_set:
        if 1 in seq.tags:
            select_dev_data.append(seq)

    print("dev_domain_seq_num:", len(select_dev_data))

    return select_dev_data


def order_test_set(test_set, all_set, per=0.1):
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
    X_valid = None
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
                                 tokenizer,
                                 args,
                                 examples_1=None
                                 ):
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token

    # Create features
    features = defaultdict(list)
    features_1 = []
    project_list = []

    unk = 0
    all_word_num = 0
    pad_num = 0
    for i, example in enumerate(examples):
        # print(123)
        # print(len(example.text))
        word_index = []
        tags = []
        text = example.text.split(" ")[:-1]
        locate = example.text.split(" ")[-1]
        project_list.append(locate)
        # tokens = tokenizer.tokenize(" ".join(text))
        tokens = text

        actul_len = len(tokens[:args.max_seq_len - 2])
        tokens = [bos_token] + tokens[:args.max_seq_len - 2] + [eos_token]  # BOS, EOS
        tokens += [pad_token] * (args.max_seq_len - len(tokens))

        refine_tokens = []
        pred = 0
        for loc, token_1 in enumerate(tokens):

            refine_tokens.append(token_1.replace("#domain", "").replace("#nondomain", ""))

            if token_1 != "[PAD]":
                token = token_1.split("#")[0]
                if token.endswith("</t>") and loc > pred:
                    word_index.append(loc - pred)
                    if token_1.split("#")[1] == "domain":
                        tags.append(1)
                    elif token_1.split("#")[1] == "nondomain":
                        tags.append(0)
                    pred = loc

        tokens = tokenizer.tokenize(" ".join(refine_tokens[1:actul_len + 1]))
        tokens = [bos_token] + tokens[:args.max_seq_len - 2] + [eos_token]  # BOS, EOS
        tokens += [pad_token] * (args.max_seq_len - len(tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        unk += input_ids.count(1)
        pad_num += input_ids.count(0)
        all_word_num += len(input_ids)

        feature = PretrainInputFeatures_1(input_ids, word_index, locate, tags)
        features[locate].append(feature)

    project_list_1 = list(set(project_list))
    project_list_1.sort(key=project_list.index)
    return features, project_list_1


def create_examples(train_corpus, tokenizer, args):
    # Load data features from cache or dataset file
    corpus_path = train_corpus
    with open(corpus_path, 'r', encoding='utf-8') as reader:
        corpus = reader.readlines()
    # with open("/home/kangyuning/gpt-pytorch/TextCNN/data/encoded_apiseq_swing.txt", 'r', encoding='utf-8') as reader1:
    #     corpus1 = reader1.readlines()

    # Create examples
    # if args.pretrain:
    corpus = list(map(lambda x: x.strip(), corpus))
    corpus = list(filter(lambda x: len(x) > 0, corpus))
    # corpus1 = list(map(lambda x: x.strip(), corpus1))
    # corpus1 = list(filter(lambda x: len(x) > 0, corpus1))
    examples = [PretrainInputExample(text) for text in corpus]
    # examples_1 = [PretrainInputExample(text) for text in corpus1]
    # Convert examples to features
    features, project_list = convert_examples_to_features(examples, tokenizer, args)
    # print(len(features))

    # tain_data_all = []
    # for seq in features:
    #     for i in range(3,len(seq.input_ids)):
    #         tain_data_all.append(Instance(seq.input_ids[:i],seq.input_ids[i]))
    # print('Saving features into cached file', cached_features_file)
    # torch.save(features, cached_features_file)

    # if args.local_rank == 0: # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    #     dist.barrier()

    # # Create dataset with features
    # if args.pretrain:
    #     all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    #     dataset = TensorDataset(all_input_ids)
    # elif args.finetune:
    #     all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    #     all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)
    #     dataset = TensorDataset(all_input_ids, all_label_ids)

    return features, project_list


if __name__ == '__main__':
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = arg_config()
    # torch.cuda.maual_seed(666)
    opts = data_path_config('config/api_data_path.json')
    df_titles = ["project_name", "all_test_API_num", "domain_API_num", "nodomain_API_num", "all_test_API_Acc_1",
                 "all_test_API_Acc_3", "all_test_API_Acc_5", "all_test_API_Acc_10"]
    domain_col = ["domain_test_API_Acc_1", "domain_test_API_Acc_3", "domain_test_API_Acc_5", "domain_test_API_Acc_10"]
    df_titles.extend(domain_col)
    nondomain_col = ["nondomain_test_API_Acc_1", "nondomain_test_API_Acc_3", "nondomain_test_API_Acc_5",
                     "nondomain_test_API_Acc_10"]
    df_titles.extend(nondomain_col)
    df_titles.extend(["cross_api_type_num", "cross_api_per","cross_api_per_cor"])
    df = DataFrame(index=range(100), columns=df_titles, )

    pretrained_sp_model = "data/API/data/vocab_par_bpe_word.model"
    vocab_file = "data/API/data/vocab_par_bpe_word.vocab"

    # pretrained_sp_model = "data/API/data/vocab_com_par.model"
    # vocab_file = "data/API/data/vocab_com_par.vocab"
    # pretrained_sp_model = "data/API/data/bpe_net_word.model"
    # vocab_file = "data/API/data/bpe_net_word.vocab"

    tokenizer = PretrainedTokenizer(pretrained_model=pretrained_sp_model, vocab_file=vocab_file)

    # 载入查询表
    with open("data/API/data/classmethod_search_list", "rb") as f3:
        search_word_dict = pickle.load(f3)

    # print(len(project_list))
    # print(len(train_data_all))

    ACC = 0
    Perplexity = 0
    EP = 0

    dev_index = []
    train_data_sample = []

    word_vocab = []

    # with open("refine_word_clean.txt", "r", encoding="utf-8")as f1:
    #     for line in f1.readlines():
    #         word_vocab.append(line.strip())

    if torch.cuda.is_available():
        if args.device_index == -1:
            print("device:cpu")

        else:
            args.device = torch.device("cuda:{}".format(args.device_index))
    if args.mode == "train":
        # pretrain_gpt = torch.load("data/API/data/model_6l8h_dis.ep17")

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

        random_nodomain_api_word_top1 = 0.0
        random_nodomain_api_word_top3 = 0.0
        random_nodomain_api_word_top5 = 0.0
        random_nodomain_api_word_top10 = 0.0

        project_api_word_top1 = defaultdict(list)
        project_api_word_top3 = defaultdict(list)
        project_api_word_top5 = defaultdict(list)
        project_api_word_top10 = defaultdict(list)

        project_domain_api_word_top1 = defaultdict(list)
        project_domain_api_word_top3 = defaultdict(list)
        project_domain_api_word_top5 = defaultdict(list)
        project_domain_api_word_top10 = defaultdict(list)

        project_nodomain_api_word_top1 = defaultdict(list)
        project_nodomain_api_word_top3 = defaultdict(list)
        project_nodomain_api_word_top5 = defaultdict(list)
        project_nodomain_api_word_top10 = defaultdict(list)

        random_all_cross_api_type_num = 0
        random_all_cross_api_per = 0

        appendControlNodesString = [
            "IF", "CONDITION", "THEN", "ELSE",
            "WHILE", "BODY",
            "TRY", "TRYBLOCK", "CATCH", "FINALLY",
            "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
            "FOREACH", "VARIABLE", "ITERABLE",
        ]
        if args.domain == "jdbc":
            train_data_all, project_list = create_examples("./data/API/data/jdbc_token_bpe.txt", tokenizer, args)
            test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes",
                            "openGauss-connector-jdbc",
                            "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
        elif args.domain == "swing":
            train_data_all, project_list = create_examples("./data/API/data/swing_token_bpe.txt", tokenizer, args)
            test_project = ["java-swing-tips", "openhospital-gui", "School-Management-System",
                            "def-guide-to-java-swing", "pumpernickel",
                            "swingx",
                            "swingBasic", "beautyeye", "joxy-oxygen", "java2script"]
        elif args.domain == "io":
            pretrained_sp_model = "data/API/data/bpe_net_word.model"
            vocab_file = "data/API/data/bpe_net_word.vocab"
            tokenizer = PretrainedTokenizer(pretrained_model=pretrained_sp_model, vocab_file=vocab_file)
            train_data_all, project_list = create_examples("./data/API/data/io_token_bpe.txt", tokenizer, args)
            test_project = []
            for j in range(args.k):
                K_train_project, K_val_project = get_k_fold_data(args.k, j, project_list)
                K_val_project_order = order_test_set(K_val_project, train_data_all)
                for i, project in enumerate(K_val_project_order):
                    if i == 0:
                        print(project[0], project[1])
                        test_project.append(project[0])
                        # for test_seq in train_data_all[project[0]]:
                        #     if 1 in test_seq.input_ids:
                        #         continue
                        #     test_data[project[0]].append(test_seq)

        pretrain_vocab = set()
        # with open("data/API/all_remove_domain_clean_token_1.txt", "r", encoding="utf-8")as f:
        #     for line in f.readlines():
        #         for subword in line.split(" "):
        #             if subword.replace("</t>", "") not in appendControlNodesString and subword != ".":
        #                 # print(subword.strip())
        #                 pretrain_vocab.add(subword.strip())
        # if "Datagram" in pretrain_vocab:
        #     print("Datagram is in")
        # vocab = tokenizer.convert_id_to_token(seq.input_ids[seq_len]).replace("</t>", "").replace("▁", "")
        # # print(vocab)
        # # if vocab not in appendControlNodesString and seq.tags[loc] == 1 and vocab != ".":
        # test_jdbc_vocab.add(vocab)
        # seq_len += 1

        # print(len(test_data["swingBasic"]))
        # test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes", "openGauss-connector-jdbc",
        #               "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
        test_jdbc_vocab = set()
        test_jdbc_subword = set()
        train_jdbc_subword = set()
        # select test data
        for i, project in enumerate(test_project):
            print(project)
            # test_project.append(project)
            for test_seq in train_data_all[project]:
                if 1 in test_seq.input_ids:
                    continue
                test_data[project].append(test_seq)
        # select all train data
        for project_name in project_list:
            if project_name not in test_project:
                for train_seq in train_data_all[project_name]:
                    train_data.append(train_seq)

        sample_train_data = random.sample(train_data, int(len(train_data) * 0.8))
        # select all train data

        # for project_name in project_list_1:
        #     if project_name not in test_project:
        #         for train_seq in train_data_all_1[project_name]:
        #             train_data.append(train_seq)
        # random

        # select dev data
        dev_data = list(set(train_data).difference(set(sample_train_data)))
        select_dev_data = select_dev_data_domain(dev_data, per=args.per)

        idx = 0
        for epoch in range(3):
            # if epoch == 0:
            #     continue
            if args.domain == "io":
                pretrain_gpt = torch.load("data/API/data/model_io.ep14")
            else:
                pretrain_gpt = torch.load("data/API/data/model_6l8h.ep14")
            # pretrain_gpt = torch.load("data/API/data/model_6l8h_18.ep4")
            # pretrain_gpt = torch.load("data/API/data/model_6l8h.ep14")
            # pretrain_gpt = torch.load("data/API/data/model_with_jdbc_1.ep18")
            print("--------------------------------------")
            ft_model = GPTLMHead(pretrain_gpt).to(args.device)

            for para in ft_model.gpt.parameters():
                # b += 1
                para.requires_grad = True

            for para in ft_model.linear.parameters():
                # b += 1
                para.requires_grad = True
            s_time = datetime.now()
            print("SAMLE EPOCH:{}".format(epoch))
            api_word_top1 = 0.0
            api_word_top3 = 0.0
            api_word_top5 = 0.0
            api_word_top10 = 0.0

            domain_api_word_top1 = 0.0
            domain_api_word_top3 = 0.0
            domain_api_word_top5 = 0.0
            domain_api_word_top10 = 0.0

            nodomain_api_word_top1 = 0.0
            nodomain_api_word_top3 = 0.0
            nodomain_api_word_top5 = 0.0
            nodomain_api_word_top10 = 0.0

            all_cross_api_type_num = 0
            all_cross_api_per = 0

            all_rec_point_num = []
            all_domain_rec_point_num = []
            all_nodomain_rec_point_num = []
            select_train_data = select_train_data_domain(sample_train_data, args.sample, per=args.per)
            train_sample_domain_count = 0
            train_sample_count = 1
            test_sample_domain_count = 0
            test_sample_count = 1
            for seq in select_train_data:
                train_sample_domain_count += seq.tags.count(1)
                train_sample_count += len(seq.input_ids)

                seq_len_1 = 0
                for loc, word_len in enumerate(seq.word_index):
                    for j in range(word_len):
                        # train_jdbc_subword.add(tokenizer.convert_id_to_token(seq.input_ids[j]).replace("▁", ""))
                        vocab = tokenizer.convert_id_to_token(seq.input_ids[seq_len_1]).replace("</t>", "").replace(
                            "▁", "")
                        # pretrain_vocab.add(vocab)
                        seq_len_1 += 1
            # for i, project in enumerate(test_project):
            #
            #     print(project)
            #     # test_project.append(project[0])
            #     test_jdbc_subword = set()
            #     for test_seq in train_data_all[project]:
            #         # if 1 in test_seq.input_ids:
            #         #     # if project == "SwingBasic":
            #         #     #     print(1111)
            #         #     continue
            #         tes_seq_len = 0
            #         for loc, word_len in enumerate(test_seq.word_index):
            #             for _ in range(word_len):
            #                 # for ids in test_seq.input_ids:
            #                 # if test_seq.tags[loc] == 1:
            #                     # print(123)
            #                 test_jdbc_subword.add(tokenizer.convert_id_to_token(tes_seq_len).replace("▁", ""))
            #                 # else:
            #                 #     print(123)
            #                 tes_seq_len += 1
            #     f = train_jdbc_subword | pretrain_vocab
            #     print(len(test_jdbc_subword))
            #     # print(test_jdbc_subword & f)
            #     # print(test_jdbc_subword)
            #     print("sub_word coverage:",
            #           len(test_jdbc_subword & f) / len(test_jdbc_subword))
            # # a = pretrain_vocab | train_vocab
            # b = test_jdbc_vocab & pretrain_vocab
            # print("coverage:",len(b)/len(test_jdbc_vocab))
            # print(test_jdbc_subword)
            # print(train_jdbc_subword)
            # print(train_sample_domain_count)
            # print("domain_percent:",train_sample_domain_count/train_sample_count)
            for project, seq in test_data.items():
                for one_seq in seq:
                    test_sample_domain_count += one_seq.tags.count(1)
                    test_sample_count += len(one_seq.input_ids)

            print("test_domain_percent:", test_sample_domain_count / test_sample_count)

            classifier = Classifier(ft_model, None, args, tokenizer.vocab, None, None, tokenizer)

            acc, perplexitys, ep = classifier.train(select_train_data, select_dev_data, args.device, args)

            for project, project_test_data in test_data.items():
                if project in test_project:
                    dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity = classifier.validate(
                        project_test_data,
                        args.device,
                        args,
                        None,
                        False)
                    Perplexity += perplexity
                    test_word_acc_1, test_word_acc_3, test_word_acc_5, test_word_acc_10, test_num, domain_acc_list, nodomian_acc_list, domain_count, cross_info = classifier.evluate(
                        project_test_data, select_train_data, args.device, args, None, True, search_word_dict)
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

                    nodomain_api_word_top1 += nodomian_acc_list[0]
                    nodomain_api_word_top3 += nodomian_acc_list[1]
                    nodomain_api_word_top5 += nodomian_acc_list[2]
                    nodomain_api_word_top10 += nodomian_acc_list[3]

                    project_api_word_top1[project].append(test_word_acc_1)
                    project_api_word_top3[project].append(test_word_acc_3)
                    project_api_word_top5[project].append(test_word_acc_5)
                    project_api_word_top10[project].append(test_word_acc_10)

                    project_domain_api_word_top1[project].append(domain_acc_list[0])
                    project_domain_api_word_top3[project].append(domain_acc_list[1])
                    project_domain_api_word_top5[project].append(domain_acc_list[2])
                    project_domain_api_word_top10[project].append(domain_acc_list[3])

                    project_nodomain_api_word_top1[project].append(nodomian_acc_list[0])
                    project_nodomain_api_word_top3[project].append(nodomian_acc_list[1])
                    project_nodomain_api_word_top5[project].append(nodomian_acc_list[2])
                    project_nodomain_api_word_top10[project].append(nodomian_acc_list[3])

                    df.loc[idx]["project_name"] = project
                    df.loc[idx]["all_test_API_num"] = test_num
                    df.loc[idx]["domain_API_num"] = domain_count
                    df.loc[idx]["nodomain_API_num"] = test_num - domain_count

                    df.loc[idx]["all_test_API_Acc_1"] = test_word_acc_1
                    df.loc[idx]["all_test_API_Acc_3"] = test_word_acc_3
                    df.loc[idx]["all_test_API_Acc_5"] = test_word_acc_5
                    df.loc[idx]["all_test_API_Acc_10"] = test_word_acc_10

                    df.loc[idx]["domain_test_API_Acc_1"] = domain_acc_list[0]
                    df.loc[idx]["domain_test_API_Acc_3"] = domain_acc_list[1]
                    df.loc[idx]["domain_test_API_Acc_5"] = domain_acc_list[2]
                    df.loc[idx]["domain_test_API_Acc_10"] = domain_acc_list[3]
                    print(nodomian_acc_list)
                    df.loc[idx]["nondomain_test_API_Acc_1"] = nodomian_acc_list[0]
                    df.loc[idx]["nondomain_test_API_Acc_3"] = nodomian_acc_list[1]
                    df.loc[idx]["nondomain_test_API_Acc_5"] = nodomian_acc_list[2]
                    df.loc[idx]["nondomain_test_API_Acc_10"] = nodomian_acc_list[3]
                    df.loc[idx]["cross_api_type_num"] = cross_info[0]
                    df.loc[idx]["cross_api_per"] = cross_info[1]
                    df.loc[idx]["cross_api_per_cor"] = cross_info[2]

                    all_cross_api_type_num += cross_info[0]
                    all_cross_api_per += cross_info[1]
                    all_rec_point_num.append(test_num)
                    all_domain_rec_point_num.append(domain_count)
                    all_nodomain_rec_point_num.append(test_num - domain_count)
                    df.to_excel(os.path.join("results", "APIRECX_{}_{}_{}.xlsx".
                                             format(args.domain, args.sample, args.epoch)))
                    idx += 1

            # avg_for_random_sample
            df.loc[idx]["project_name"] = None
            df.loc[idx]["all_test_API_num"] = mean(all_rec_point_num)
            df.loc[idx]["domain_API_num"] = mean(all_domain_rec_point_num)
            df.loc[idx]["nodomain_API_num"] = mean(all_nodomain_rec_point_num)

            df.loc[idx]["all_test_API_Acc_1"] = api_word_top1 / args.k
            df.loc[idx]["all_test_API_Acc_3"] = api_word_top3 / args.k
            df.loc[idx]["all_test_API_Acc_5"] = api_word_top5 / args.k
            df.loc[idx]["all_test_API_Acc_10"] = api_word_top10 / args.k

            df.loc[idx]["domain_test_API_Acc_1"] = domain_api_word_top1 / args.k
            df.loc[idx]["domain_test_API_Acc_3"] = domain_api_word_top3 / args.k
            df.loc[idx]["domain_test_API_Acc_5"] = domain_api_word_top5 / args.k
            df.loc[idx]["domain_test_API_Acc_10"] = domain_api_word_top10 / args.k

            df.loc[idx]["nondomain_test_API_Acc_1"] = nodomain_api_word_top1 / args.k
            df.loc[idx]["nondomain_test_API_Acc_3"] = nodomain_api_word_top3 / args.k
            df.loc[idx]["nondomain_test_API_Acc_5"] = nodomain_api_word_top5 / args.k
            df.loc[idx]["nondomain_test_API_Acc_10"] = nodomain_api_word_top10 / args.k
            df.loc[idx]["cross_api_type_num"] = all_cross_api_type_num / args.k
            df.loc[idx]["cross_api_per"] = all_cross_api_per / args.k

            idx += 1
            df.loc[idx]["project_name"] = epoch
            idx += 1

            random_api_word_top1 += api_word_top1
            random_api_word_top3 += api_word_top3
            random_api_word_top5 += api_word_top5
            random_api_word_top10 += api_word_top10

            random_domain_api_word_top1 += domain_api_word_top1
            random_domain_api_word_top3 += domain_api_word_top3
            random_domain_api_word_top5 += domain_api_word_top5
            random_domain_api_word_top10 += domain_api_word_top10

            random_nodomain_api_word_top1 += nodomain_api_word_top1
            random_nodomain_api_word_top3 += nodomain_api_word_top3
            random_nodomain_api_word_top5 += nodomain_api_word_top5
            random_nodomain_api_word_top10 += nodomain_api_word_top10

            random_all_cross_api_type_num += all_cross_api_type_num
            random_all_cross_api_per += all_cross_api_per
            print("EPOCH:{}".format(epoch), datetime.now() - s_time)
            df.to_excel(os.path.join("results", "APIRECX_{}_{}_{}.xlsx".
                                     format(args.domain, args.sample, args.epoch)))
        idx += 1
        df.loc[idx]["project_name"] = "Random AVG"
        df.to_excel(os.path.join("results", "APIRECX_{}_{}_{}.xlsx".
                                 format(args.domain, args.sample, args.epoch)))
        for project in test_project:
            idx += 1
            df.loc[idx]["project_name"] = project

            df.loc[idx]["all_test_API_Acc_1"] = mean(project_api_word_top1[project])
            df.loc[idx]["all_test_API_Acc_3"] = mean(project_api_word_top3[project])
            df.loc[idx]["all_test_API_Acc_5"] = mean(project_api_word_top5[project])
            df.loc[idx]["all_test_API_Acc_10"] = mean(project_api_word_top10[project])
            # avg_project_api_word_top1 += sum(project_api_word_top1[project])
            # avg_project_api_word_top3 = sum(project_api_word_top3[project])
            # avg_project_api_word_top5 = sum(project_api_word_top5[project])
            # avg_project_api_word_top10 = sum(project_api_word_top10[project])

            df.loc[idx]["domain_test_API_Acc_1"] = mean(project_domain_api_word_top1[project])
            df.loc[idx]["domain_test_API_Acc_3"] = mean(project_domain_api_word_top3[project])
            df.loc[idx]["domain_test_API_Acc_5"] = mean(project_domain_api_word_top5[project])
            df.loc[idx]["domain_test_API_Acc_10"] = mean(project_domain_api_word_top5[project])
            # avg_project_domain_api_word_top1 = sum(project_domain_api_word_top1[project])
            # avg_project_domain_api_word_top3 = sum(project_domain_api_word_top3[project])
            # avg_project_domain_api_word_top5 = sum(project_domain_api_word_top5[project])
            # avg_project_domain_api_word_top10 = sum(project_domain_api_word_top5[project])

            df.loc[idx]["nondomain_test_API_Acc_1"] = mean(project_nodomain_api_word_top1[project])
            df.loc[idx]["nondomain_test_API_Acc_3"] = mean(project_nodomain_api_word_top1[project])
            df.loc[idx]["nondomain_test_API_Acc_5"] = mean(project_nodomain_api_word_top1[project])
            df.loc[idx]["nondomain_test_API_Acc_10"] = mean(project_nodomain_api_word_top1[project])
            df.to_excel(os.path.join("results", "APIRECX_{}_{}_{}.xlsx".
                                     format(args.domain, args.sample, args.epoch)))
            # avg_project_nodomain_api_word_top1 = sum(project_nodomain_api_word_top1[project])
            # avg_project_nodomain_api_word_top3 = sum(project_nodomain_api_word_top3[project])
            # avg_project_nodomain_api_word_top5 = sum(project_nonodomain_api_word_top5[project])
            # avg_project_nodomain_api_word_top10 = sum(project_nodomain_api_word_top5[project])

        idx += 1
        df.loc[idx]["all_test_API_Acc_1"] = random_api_word_top1 / 3 * args.k
        df.loc[idx]["all_test_API_Acc_3"] = random_api_word_top3 / 3 * args.k
        df.loc[idx]["all_test_API_Acc_5"] = random_api_word_top5 / 3 * args.k
        df.loc[idx]["all_test_API_Acc_10"] = random_api_word_top10 / 3 * args.k

        df.loc[idx]["domain_test_API_Acc_1"] = random_domain_api_word_top1 / 3 * args.k
        df.loc[idx]["domain_test_API_Acc_3"] = random_domain_api_word_top3 / 3 * args.k
        df.loc[idx]["domain_test_API_Acc_5"] = random_domain_api_word_top5 / 3 * args.k
        df.loc[idx]["domain_test_API_Acc_10"] = random_domain_api_word_top10 / 3 * args.k

        df.loc[idx]["nondomain_test_API_Acc_1"] = random_nodomain_api_word_top1 / 3 * args.k
        df.loc[idx]["nondomain_test_API_Acc_3"] = random_nodomain_api_word_top3 / 3 * args.k
        df.loc[idx]["nondomain_test_API_Acc_5"] = random_nodomain_api_word_top5 / 3 * args.k
        df.loc[idx]["nondomain_test_API_Acc_10"] = random_nodomain_api_word_top10 / 3 * args.k

        df.loc[idx]["cross_api_type_num"] = random_all_cross_api_type_num / 3 * args.k
        df.loc[idx]["cross_api_per"] = random_all_cross_api_per / 3 * args.k

        df.to_excel(os.path.join("results", "APIRECX_{}_{}_{}.xlsx".
                                 format(args.domain, args.sample, args.epoch)))


    elif args.mode == "test":
        for i in range(args.k):
            gpt = GPT(vocab_size=tokenizer.vocab_size,
                      seq_len=512,
                      d_model=256,
                      n_layers=6,
                      n_heads=8,
                      d_ff=512,
                      embd_pdrop=0.1,
                      attn_pdrop=0.1,
                      resid_pdrop=0.1,
                      pad_id=0)

            ft_model = GPTLMHead(gpt).to(args.device)
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
            # test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes",
            #                 "openGauss-connector-jdbc",
            #                 "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
            # test_project = ["StoreManager","SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
            appendControlNodesString = [
                "IF", "CONDITION", "THEN", "ELSE",
                "WHILE", "BODY",
                "TRY", "TRYBLOCK", "CATCH", "FINALLY",
                "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
                "FOREACH", "VARIABLE", "ITERABLE",
            ]
            # test_project = ["java-swing-tips", "openhospital-gui", "School-Management-System", "def-guide-to-java-swing", "pumpernickel",
            #                 "swingx",
            #                 "swingBasic", "beautyeye", "joxy-oxygen", "java2script"]
            # test_project_1 = ["beautyeye","joxy-oxygen"
            #                  ]
            pretrain_vocab = set()
            with open("data/API/all_remove_domain_clean_token.txt", "r", encoding="utf-8")as f:
                for line in f.readlines():
                    for subword in line.split(" "):
                        if subword.replace("</t>", "") not in appendControlNodesString and subword != ".":
                            pretrain_vocab.add(subword.strip().replace("</t>", ""))
            # vocab = tokenizer.convert_id_to_token(seq.input_ids[seq_len]).replace("</t>", "").replace("▁", "")
            # # print(vocab)
            # # if vocab not in appendControlNodesString and seq.tags[loc] == 1 and vocab != ".":
            # test_jdbc_vocab.add(vocab)
            # seq_len += 1
            # test_project = []
            # for j in range(args.k):
            #     K_train_project, K_val_project = get_k_fold_data(args.k, j, project_list)
            #     K_val_project_order = order_test_set(K_val_project, train_data_all)
            #     for i, project in enumerate(K_val_project_order):
            #         if i == 0:
            #             print(project[0], project[1])
            #             test_project.append(project[0])
            #             for test_seq in train_data_all[project[0]]:
            #                 if 1 in test_seq.input_ids:
            #                     continue
            #                 test_data[project[0]].append(test_seq)
            # print(len(test_data["swingBasic"]))
            test_project = ["pgjdbc-ng", "clickhouse4j", "StoreManager", "pgjdbc", "jdbc-recipes",
                            "openGauss-connector-jdbc",
                            "Game-Lease", "JDBC", "SoftwareSystemOfDlpu", "starschema-bigquery-jdbc"]
            test_jdbc_vocab = set()
            for i, project in enumerate(test_project):

                print(project)
                print("------------")
                for test_seq in train_data_all[project]:
                    if 1 in test_seq.input_ids:
                        continue
                    test_data[project].append(test_seq)
                    if project in test_project:
                        # for seq in test_seq.input_ids:
                        seq_len = 0
                        for loc, word_len in enumerate(test_seq.word_index):
                            for _ in range(word_len):
                                vocab = tokenizer.convert_id_to_token(test_seq.input_ids[seq_len]).replace("</t>",
                                                                                                           "").replace(
                                    "▁", "")

                                test_jdbc_vocab.add(vocab)
                                seq_len += 1

            for project_name in project_list:
                if project_name not in test_project:
                    for train_seq in train_data_all[project_name]:
                        train_data.append(train_seq)
            train_data_for_sum = []
            for project_name in project_list:
                for train_seq in train_data_all[project_name]:
                    train_data_for_sum.append(train_seq)
            # for project_name in project_list_1:
            #     if project_name not in test_project:
            #         for train_seq in train_data_all_1[project_name]:
            #             train_data.append(train_seq)
            train_data_1 = random.sample(train_data, int(len(train_data) * 0.8))
            dev_data = list(set(train_data).difference(set(train_data_1)))
            select_dev_data = select_dev_data_domain(dev_data, per=args.per)
            print("train data all num:", len(train_data_1))
            for i in range(1):
                api_word_top1 = 0.0
                api_word_top3 = 0.0
                api_word_top5 = 0.0
                api_word_top10 = 0.0
                domain_api_word_top1 = 0.0
                domain_api_word_top3 = 0.0
                domain_api_word_top5 = 0.0
                domain_api_word_top10 = 0.0

                select_train_data = select_train_data_domain(train_data_1, args.sample, per=args.per)
                # select_train_data = random.sample(train_data_1,int(len(train_data_1)*args.sample))
                train_sample_domain_count = 0
                train_sample_count = 1
                test_sample_domain_count = 0
                test_sample_count = 1
                for seq in select_train_data:
                    train_sample_domain_count += seq.tags.count(1)
                    train_sample_count += len(seq.input_ids)

                    seq_len_1 = 0
                    for loc, word_len in enumerate(seq.word_index):
                        for _ in range(word_len):
                            vocab = tokenizer.convert_id_to_token(seq.input_ids[seq_len_1]).replace("</t>", "").replace(
                                "▁", "")
                            # print(vocab)
                            # if vocab not in appendControlNodesString and seq.tags[loc] == 1 and vocab != ".":
                            pretrain_vocab.add(vocab)
                            seq_len_1 += 1
                # a = pretrain_vocab | train_vocab
                b = test_jdbc_vocab & pretrain_vocab
                print("coverage:", len(b) / len(test_jdbc_vocab))
                print(train_sample_domain_count)
                print("domain_percent:", train_sample_domain_count / train_sample_count)
                for project, seq in test_data.items():
                    for one_seq in seq:
                        test_sample_domain_count += one_seq.tags.count(1)
                        test_sample_count += len(one_seq.input_ids)
                print(test_sample_domain_count)
                print("test_domain_percent:", test_sample_domain_count / test_sample_count)
                model_lstm = None
                classifier = Classifier(ft_model, model_lstm, args, tokenizer.vocab, word_vocab, None, tokenizer)
                print("train data num:", len(select_train_data))
                acc, perplexitys, ep = classifier.train(select_train_data, select_dev_data, test_data, args.device,
                                                        args, i, search_word_dict)
                # ACC += acc
                # Perplexity += perplexity
                # EP += ep
                # sorted(test_data.items(), key=lambda d: len(d[1]), reverse=False)
                # escap_list = ["java-swing-tips","openhospital-gui","j2se_for_android","weblaf","pumpernickel","swingx","swingBasic","beautyeye","littleluck","java2script"]

                escap_list = ["java-htop-tips"]
                for project, project_test_data in test_data.items():
                    print(project, len(project_test_data))
                    # if project == "clickhouse4j":
                    if project in test_project:
                        dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity = classifier.validate(
                            project_test_data,
                            args.device,
                            args,
                            None,
                            False)
                        print(project, perplexity)
                        Perplexity += perplexity
                        test_word_acc_1, test_word_acc_3, test_word_acc_5, test_word_acc_10, test_num, domain_acc_list, domain_count = classifier.evluate(
                            project_test_data, train_data_for_sum, args.device, args, None, True, search_word_dict)
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





