from dataloader.DataloaderApi import *
import torch.nn as nn
import torch.optim
import time
import difflib
import torch.nn.functional as F
import datetime
from collections import Counter
import re
class Candidate:
    def __init__(self, pre_ids, pro, is_complete):
        self.pre_ids = pre_ids
        self.pro = pro
        self.is_complete = is_complete


class BestToken:
    def __init__(self, pre_ids, pro):
        self.pre_ids = pre_ids
        self.pro = pro
class Classifier:
    def __init__(self, model, args, vocab, word_vocab, Rvocab):
        self.model = model
        self.vocab = vocab
        self.args = args
        self.word_vocab = word_vocab
        self.Rvocab = Rvocab
        self.counter = Counter()
        self.next_api = 0
        self.ep = 0

    # 打印模型参数
    def summary(self):
        print(self.model)

    def train(self, train_data, dev_data, test_data, args_device, arg, epoch, train_batch_len=None, dev_batch_len=None):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)

        train_loss_list, train_acc_list = [], []
        best_acc = 0
        dev_loss_list, dev_acc_list = [], []
        patenice = 0
        pre_dev_acc = 0
        for ep in range(self.args.epoch):
            train_data_num = 0
            train_num = 1
            train_acc = 0
            train_acc_1 = 0
            train_loss = 0
            start_time = datetime.datetime.now()
            word_acc = 0
            self.ep = ep
            # 让模型进入训练模式
            self.model.train()
            batch_num = 0
            for onebatch in get_batch_train(train_data, self.args.batch_size, arg, train_batch_len):
                words, tags, mask, seq_lengths = batch_numberize(onebatch, self.vocab, args_device, arg)
                # print(words)
                targets = words[:, 1:].contiguous()

                pred = self.model(words)
                pred = pred[:, :-1].contiguous()
                # 反向传播，计算误差
                loss = self.compuate_loss(targets.view(-1), pred.view(-1, pred.shape[-1]))
                optimizer.zero_grad()
                loss.backward()
                #记得改回来
                optimizer.step()
                train_loss += loss.data.item()
                acc_1, num = self.compuate_acc(targets.view(-1), pred.view(-1, pred.shape[-1]))
                train_acc += acc_1
                train_data_num += num
            end_time = datetime.datetime.now()
            during_time = end_time - start_time
            dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity = self.validate(dev_data, args_device,
                                                                                               arg, dev_batch_len, True)
            pre_dev_acc = dev_acc
            train_acc /= train_data_num
            train_loss /= train_data_num

            train_loss_list.append(train_loss)

            if patenice > 2:
                self.ep = self.ep - patenice
                break
            if dev_acc > best_acc:
                print(dev_acc, best_acc)
                dev_acc_list.append(dev_acc)
                best_acc = dev_acc_list[-1]
                patenice = 0
            else:
                patenice += 1
                print(patenice)

            # epoch经过epoch轮，如果开发集acc没有上升或者loss没有下降，则停止训练
            # if (ep+1)% patience == 0 and dev_acc_list[ep]< dev_acc_list[ep-patience+1]:
            #     break
            # print(batch_num)
            print("[Epoch {}] train loss :{} train_acc:{} %  Time:{}  train_word_acc:{} 训练数据总数:{} 训练数据词总数:{}".format(
                ep + 1, train_loss, train_acc * 100, during_time, word_acc / train_num, train_data_num, train_num))
            print(
                "[Epoch {}] dev loss :{} dev_acc:{} dev_word_acc:{} % 测试数据总数:{} 测试数据词总数:{} 困惑度:{}".format(ep + 1,
                                                                                                          dev_loss,
                                                                                                          dev_acc * 100,
                                                                                                          dev_word_acc / dev_num,
                                                                                                          dev_data_num,
                                                                                                          dev_num,
                                                                                                          perplexity))

            # print(train_acc_1)
            # print(train_num)
            # print(self.counter)
            # print(self.next_api)

            # print(self.vocab)
            # test_acc,test_loss = self.evluate(test_data, args_device)
            # torch.save(self.model, f="data/API/trained_model_{}_jdbc".format(-1))
            # dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity = self.validate(dev_data, args_device, arg,
            #                                                                                  dev_batch_len,True)
        s = datetime.datetime.now()
        # test_word_acc_1,test_word_acc_3,test_word_acc_5,test_word_acc_10, test_num,domain_acc_list,domain_count = self.evluate(test_data, args_device, arg, dev_batch_len, True, search_word_dict)
        # print("test_word_acc_top1:{} test_word_acc_top3:{} test_word_acc_top5:{} test_word_acc_top10:{}".format(test_word_acc_1,test_word_acc_3,test_word_acc_5,test_word_acc_10))
        print(datetime.datetime.now() - s)
        # print(test_num)

        if arg.is_save:
            #     torch.save(self.model.state_dict(), "data/API/trained_GPT_{}_jdbc_bpe".format(-1))
            torch.save(self.model, "data/API/data/trained_GPT_{}_swing_bpe_1".format(-1))
        # print("{} round:training complete".format(epoch))
        return dev_acc_list[-1], perplexity, self.ep

    # 验证
    def validate(self, dev_data, args_device, arg, batch_len, is_refine):
        dev_loss = 0
        dev_acc = 0
        dev_acc_1 = 0
        perplexity = 0.0
        dev_data_num = 0
        dev_word_acc = 0
        batch_num = 0
        # 让模型进入评估模式
        self.model.eval()
        dev_num = 1
        # print(pre_acc)
        # with torch.no_grad:
        for onebatch in get_batch_train(dev_data, self.args.batch_size, arg, batch_len):
            batch_num += 1
            # 数据变量化
            words, tags, mask, seq_lengths = batch_numberize_pre(onebatch, self.vocab, args_device, arg)
            # 前向传播（数据喂给模型）

            targets = words[:, 1:].contiguous()
            pred = self.model(words)
            pred = pred[:, :-1].contiguous()
            loss = self.compuate_loss(targets.view(-1), pred.view(-1, pred.shape[-1]))
            perplexity += torch.exp(loss).data.item()
            dev_loss += loss.data.item()
            acc_1, num = self.compuate_acc(targets.view(-1), pred.view(-1, pred.shape[-1]))
            dev_acc += acc_1
            dev_data_num += num


        dev_acc /= dev_data_num
        dev_loss /= dev_data_num
        perplexity /= batch_num

        return dev_acc, dev_loss, dev_data_num, dev_word_acc, dev_num, perplexity

    # 评估模型
    def evluate(self, dev_data, args_device, arg, batch_len, is_refine, search_word_dict):
        reject_token = ["[EOS]", "[BOS]", "[PAD]", "[UNK]"]
        appendControlNodesStrings = [
            "IF", "CONDITION", "THEN", "ELSE",
            "WHILE", "BODY",
            "TRY", "TRYBLOCK", "CATCH", "FINALLY",
            "FOR", "INITIALIZATION", "COMPARE", "UPDATE",
            "FOREACH", "VARIABLE", "ITERABLE",
        ]
        length_pro = 0
        length_pro_1 = 0
        length_pro_2 = 0
        control_node = 0
        k = 10
        top1_len = Counter()
        top1_ground_true_len = Counter()
        top1_len_info = Counter()
        top3_len = Counter()
        top3_ground_true_len = Counter()
        beam_size = arg.boundary
        batch_num = 0
        dev_data_num = 0
        perplexity = 0.0
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
        self.model.eval()
        dev_num = 1
        refine_num = 0
        no_refine_num = 0
        domain_count = 1
        distance = []
        for line_num, onebatch in enumerate(get_batch_train(dev_data, 1, arg, None,False)):
            batch_num += 1
            print("*********new_seq***********")
            words, tags, mask, seq_lengths = batch_numberize_pre(onebatch, self.vocab,args_device, arg)
            targets = words[:, 1:].contiguous()

            pred = self.model(words)

            for word_loc,api in enumerate(onebatch[0].input_ids):
                if self.Rvocab.get(targets[0, word_loc:word_loc + 1].item()) == "[EOS]":
                    break
                true_token = []
                if word_loc == 0:
                    true_token1 = self.Rvocab.get(targets[0, :1].item())

                    print("number one word", true_token1)
                    continue

                true_token.append(self.Rvocab.get(targets[0, word_loc:word_loc + 1].item()))
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
                    # if self.Rvocab.get(targets[0, word_loc:word_loc + 1].item()) == "[EOS]":
                # else:
                #     continue
                a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>", "", true_api)
                true_api_nop = re.sub(u"\\(\\)|\\{|\\[\\]|\\>|\\<", "", a)
                class_name = true_api.split(".")[0]
                # print(true_api)
                # print(class_name)
                # if class_name in appendControlNodesStrings:
                #    pass
                # else:
                #    pass

                init_candidate_list, init_bestTokens = self.compuate_acc_2(
                    pred[0, word_loc:word_loc + 1, :], None, k, reject_token,
                    search_word_dict, class_name, appendControlNodesStrings, beam_size=beam_size)

                bestToken_list = init_bestTokens
                final_result = []
                final_result_check = []
                final_result_nop = []
                final_class_result = []

                bestToken_list = sorted(bestToken_list, key=lambda x: x.pro, reverse=True)
                for best_token in bestToken_list[:10]:
                    # print("".join([self.Rvocab.get(index) for index in best_token.pre_ids]).replace("▁", ""))
                    final_result.append("".join(
                        [self.Rvocab.get(index) for index in best_token.pre_ids]))
                    # final_result_check.append("".join(
                    # #     [self.Rvocab.get(index) for index in best_token.pre_ids]))
                    # final_result_check.append(best_token.pro)
                    # final_class_result.append("".join(
                    #     [self.Rvocab.get(index) for index in best_token.pre_ids]).replace("▁",
                    #                                                                                          "").replace(
                    #     "</t>", "").replace("[EOS]", "").replace("[UNK]", "").split(".")[0])
                    # raw_api = class_name_var + "".join(
                    #     [self.Rvocab.get(index) for index in best_token.pre_ids]).replace("▁",
                    #                                                                                          "").replace(
                    #     "</t>", "").replace("[EOS]", "").replace("[UNK]", "").replace("[PAD]", "")
                    # a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>", "", raw_api)
                    # true_api_nop_can = re.sub(u"\\(\\)|\\{|\\[\\]|\\>|\\<", "", a)
                    # final_result_nop.append(true_api_nop_can)
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
                        # print (self.Rvocab.get(words[:,word_loc:word_loc+1].item()))
                        # print (onebatch[0].tags)
                        # print (onebatch[0].input_ids)
                        # # print (self.Rvocab.get(pred[0, word_loc:word_loc + 1, :].item()))
                        # print ("--------------------------")
                        dev_word_acc_class_10 += 1
                    distance.append(word_loc)
                else:
                    if true_api.replace("</t>", "") not in appendControlNodesStrings:
                        num_1 += 1
                        if true_api_nop.replace("</t>", "") in final_result_nop:
                            print("参数错误")
                        else:
                            if onebatch[0].tags[word_loc] == 1:
                                num_5 += 1
                                # print("not par error")
                                # print(final_result_nop)

                    else:
                        num_2 += 1


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
        print("------------------------------")
        print(dev_num)
        print(domain_count)
        print(sum(distance) / len(distance))
        return word_acc_1, word_acc_3, word_acc_5, word_acc_10, dev_num, [dev_1, dev_3, dev_5, dev_10], domain_count

    def pretrain(self,train_data, args_device, arg ,train_batch_len=None):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)

        train_loss_list, train_acc_list = [], []
        best_acc = 0
        dev_loss_list, dev_acc_list = [], []
        patenice = 0
        pre_dev_acc = 0
        init_loss = 100
        for ep in range(self.args.epoch):
            train_data_num = 0
            train_num = 1
            train_acc = 0
            train_acc_1 = 0
            train_loss = 0
            start_time = datetime.datetime.now()
            word_acc = 0
            self.ep = ep
            # 让模型进入训练模式
            self.model.train()
            batch_num = 0
            for onebatch in get_batch_train(train_data, self.args.batch_size, arg, train_batch_len):
                train_data_num += 1
                words, tags, mask, seq_lengths = batch_numberize_pre(onebatch, self.vocab, args_device, arg)
                # print(words.shape)
                targets = words[:, 1:].contiguous()

                pred = self.model(words)
                pred = pred[:, :-1].contiguous()
                # 反向传播，计算误差
                loss = self.compuate_loss(targets.view(-1), pred.view(-1, pred.shape[-1]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()
            end_time = datetime.datetime.now()
            during_time = end_time - start_time
            train_loss /= train_data_num

            train_loss_list.append(train_loss)

            print("[Epoch {}] train loss :{} Time:{}".format(
                ep + 1, train_loss, during_time))

            if train_loss < init_loss:
                init_loss = train_loss
            else:
                print ("train finined")
                break

            if arg.is_save:
                # torch.save(self.model.state_dict(), "data/API/pretrain/trained_model_{}_jdk".format(-1))
                torch.save(self.model, f="data/API/pretrain/jdbc/trained_model_{}_jdk_{}_word_sample_{}_1e-3_2layer".format(ep,self.args.hidden_size,self.args.sample))
        print("{} round:training complete".format(self.args.epoch))




    # 计算准确率
    def compuate_acc(self, true_tags, logit):
        select_index = []
        correct_num = 0
        for i in range(logit.shape[0]):
            if true_tags[i].item() != 0:
                # prediction[i] = logit[i]
                select_index.append(i)
        # if (len(select_index)) == 0:
        #     continue
        logit = torch.index_select(logit, 0, torch.tensor(select_index).long().to(self.args.device))
        true_tags = torch.index_select(true_tags, 0, torch.tensor(select_index).long().to(self.args.device))
        logit = F.softmax(logit, dim=1)
        for i in range(logit.shape[0]):
            if true_tags[i] in torch.argsort(logit[i], descending=True)[: 2]:
                correct_num += 1
        # 返回正确的item的数目,eq是返回一个矩阵，sum之后返回总数

        # return torch.eq(torch.argmax(logit, dim=1), true_tags).sum().item(), true_tags.shape[0]
        return correct_num, true_tags.shape[0]

    # 计算损失
    #
    def compuate_loss(self, true_tags, logit):

        # CrossEntropyLoss = LogSoftmax + NLLoss
        # print(true_tags)
        # print(logit.shape)
        # print(true_tags.shape)
        loss = nn.CrossEntropyLoss(ignore_index=self.word_vocab)
        # true_tags = true_tags[:, :1]
        # true_tags = true_tags.squeeze(dim=1)
        loss = loss(logit, true_tags)
        # loss = loss(logit.view(-1, logit.shape[-1]), true_tags.view(-1))
        # print(loss.data)
        return loss

    def compuate_acc_2(self, logit, pre_info, k, reject_token, search_dict, class_name, control_lable, beam_size,
                       target=None):
        bestTokens = []
        pre_candidate = []
        lowest_pro = 0.0
        logit = F.softmax(logit, dim=1)
        sort = torch.argsort(logit, dim=1, descending=True)
        flag1 = False
        flag2 = False

        for j in range(logit.shape[1]):
            if len(bestTokens) < k:
                method_name = self.Rvocab.get(sort[0][j].item())
                # print(sort[0][j].item())
                # print(method_name)
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
    # def compuate_acc_2(self, logit, pre_info, k, reject_token, beam_size, target=None):
    #     bestTokens = []
    #     pre_candidate = []
    #     lowest_pro = 0.0
    #     logit = F.softmax(logit, dim=1)
    #     sort = torch.argsort(logit, dim=1, descending=True)
    #     flag1 = False
    #     flag2 = False
    #     # acc_num = 0
    #     # print(target)
    #     # print(sort[0][:5].tolist())
    #     # if target in sort[0][:5].tolist():
    #     #     acc_num += 1
    #
    #     if len(pre_info) != 1:
    #         for i in range(logit.shape[0]):
    #             for j in range(self.args.boundary):
    #                 append_info.append((sort[-1][j].item() % self.tokenizer.vocab_size,
    #                                     logit[-1][sort[-1][j].item()].item() *
    #                                     pre_info[int(sort[-1][j].item() / self.tokenizer.vocab_size)][1]))
    #                 pre_candidate.append(pre_info[int(sort[-1][j].item() / self.tokenizer.vocab_size)][0])
    #     else:
    #         # for i in range(logit.shape[0]):
    #         for j in range(logit.shape[1]):
    #             if flag1 and flag2:
    #                 break
    #             # print(pre_info[0][1])
    #             # print(logit[-1][sort[-1][j].item()])
    #             # print(sort[-1][j].item())
    #             # print(target)
    #             if len(pre_candidate) < beam_size:
    #                 if self.Rvocab.get(sort[0][j].item()).find(
    #                         "</t>") == -1 and self.Rvocab.get(
    #                     sort[0][j].item()) not in reject_token:
    #                     pre_candidate.append(
    #                         Candidate([sort[0][j].item()], logit[0][sort[0][j].item()].item(), False))
    #             else:
    #                 flag1 = True
    #             if len(bestTokens) < k:
    #                 if self.Rvocab.get(sort[0][j].item()).find(
    #                         "</t>") != -1 or self.Rvocab.get(sort[0][j].item()) in reject_token:
    #                     # print(self.Rvocab.get(sort[0][j].item()))
    #                     bestTokens.append(BestToken([sort[0][j].item()], logit[0][sort[0][j].item()].item()))
    #             else:
    #                 flag2 = True
    #
    #                 # pre_candidate.append(sort[-1][j].item() % 4000)
    #     bestTokens = sorted(bestTokens, key=lambda x: x.pro, reverse=True)
    #     # print(lowest_pro)
    #     return pre_candidate, bestTokens, bestTokens[0].pro

    def refine(self, pred, true_tags):
        tokens = []
        true_token = []
        flag = False
        refine_words = []
        # 概率值top5
        for j in range(pred.shape[0]):
            true_token.append(self.Rvocab.get(true_tags[j].item()).replace("▁", ""))
        for i in range(5):
            token = []
            for j in range(pred.shape[0]):
                top5 = torch.argsort(pred[j], descending=True)[: 5]
                token.append(self.Rvocab.get(top5[i].item()).replace("▁", ""))

            word = "".join(token)
            tokens.append(word)

            # print(true_word)
            refine_word = difflib.get_close_matches(word, self.word_vocab, 1, 0.6)
            if len(refine_word) == 0:
                refine_words.append("null")
            else:
                refine_words.append(refine_word[0])
        true_word = "".join(true_token)
        # print(word, true_word, refine_word)
        # if len(refine_word) != 0:
        if true_word in refine_words:
            # print(true_word,word,refine_word[0])
            flag = True
        else:
            pass
            # print(true_word,refine_word[0])
        return flag
