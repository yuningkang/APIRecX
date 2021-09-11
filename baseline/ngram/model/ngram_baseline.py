import torch.nn as nn
import torch.nn.functional as F
import torch
class NGramLanguageModeler(nn.Module):

    # 初始化时需要指定：单词表大小、想要嵌入的维度大小、上下文的长度
    def __init__(self, vocab_size, context_size,args):
        # 继承自nn.Module，例行执行父类super 初始化方法
        embedding_dim = 256
        super(NGramLanguageModeler, self).__init__()
        # 建立词嵌入模块
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(context_size * embedding_dim, args.hidden_size)
        # 线性层2，隐藏层 hidden_size 为128
        self.linear2 = nn.Linear(args.hidden_size, vocab_size)

    # 重写的网络正向传播方法
    # 只要正确定义了正向传播
    # PyTorch 可以自动进行反向传播
    def forward(self, inputs):

        embeds = self.embeddings(inputs).view((inputs.shape[0],1, -1))

        out = F.relu(self.linear1(embeds))
        # 通过线性层2后
        out = self.linear2(out)
        log_probs = out.squeeze(dim=1)
        return log_probs