import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    #传入模型维度和最大长度，生成位置编码
    def __init__(self,d_model,max_len=512):
        super().__init__()
        pe = torch.zeros(max_len,d_model)#初始化位置编码矩阵，形状为(max_len,d_model)，每行表示一个位置的编码
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)#生成位置索引，形状为(max_len,1)，每行表示一个位置
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))#根据维度计算位置编码的频率，使用指数函数来生成不同频率的正弦和余弦函数
        pe[:,0::2]=torch.sin(position*div_term)#根据位置和维度计算正弦和余弦函数，生成位置编码矩阵
        pe[:,1::2]=torch.cos(position*div_term)#根据位置和维度计算正弦和余弦函数，生成位置编码矩阵
        self.register_buffer('pe',pe.unsqueeze(0))#将位置编码注册为模型的缓冲区，这样它就不会被更新，但会随模型一起保存和加载

    #前向传播时将位置编码加到输入上
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]#将输入和位置编码相加，位置编码根据输入的长度进行切片，以适应不同长度的输入序列
    
class TextTransformer(nn.Module):
    def __init__(self,vocab_size,embed_dim,nhead,num_layers,num_classes):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)#词嵌入层，将输入的词索引转换为嵌入向量
        self.pos_encoder=PositionalEncoding(embed_dim)#位置编码层，生成位置编码并
        
        #Transformer编码器层，使用nn.TransformerEncoderLayer来构建多层Transformer编码器
        encoder_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=nhead,batch_first=True)#定义Transformer编码器层，指定模型维度、头数和批处理维度
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)#将编码器层堆叠成多层Transformer编码器，指定层数
        
        self.fc=nn.Linear(embed_dim,num_classes)#全连接层，将Transformer编码器的输出映射到类别数，进行分类任务
        
    def forward(self,x):
        x=self.embedding(x)*math.sqrt(x.size(-1))#将输入的词索引转换为嵌入向量，并进行缩放，乘以嵌入维度的平方根来保持数值稳定性
        x=self.pos_encoder(x)#将位置编码加到嵌入向量上，提供位置信息给模型
        x=self.transformer(x)#通过Transformer编码器处理输入，提取特征
        return self.fc(x.mean(dim=1)) # 取平均作为句向量 (Pooling)