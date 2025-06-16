import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import copy

# embedding = nn.Embedding(20, 3)
# input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
# # print(embedding(input1))
# # print(input1.shape)
#
# embedding = nn.Embedding(10, 3, padding_idx=0)
# input1 = torch.LongTensor([[0,2,0,5]])
# print(embedding(input1))

#构建Embedding类来实现文本嵌入
class Embeddings (nn.Module):
    def __init__(self,d_model, vocab):
        #d_model：词嵌入的纬度
        #vocab:词表的大小
        super(Embeddings, self).__init__()
        #定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        #x:代表输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        #d_model：嵌入词纬度
        #dropout：置零比率
        #句子最大长度
        super(PositionalEncoding, self).__init__()

        #实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)

        #初始化一个位置矩阵，
        pe = torch.zeros(max_len, d_model)

        #初始化一个绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        #定义一个变换矩阵， 跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        #将前面定义的变化矩阵进行奇偶分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #二维变成三维
        pe = pe.unsqueeze(0)

        #将位置编码注册成buffer，不随优化器而更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:文本的词嵌入表示
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508],[491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print(embr)

d_model = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)




