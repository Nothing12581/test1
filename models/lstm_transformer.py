import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        #d_model是Transformer的特征维度，本例为32
        #max_len是支持的最大序列长度，默认为5000

        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        #预生成一个全零矩阵

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        #不同维度使用不同频率的正弦波，用来分辨是序列中的第几个点

        pe[:, 1::2] = torch.cos(position * div_term)
        #偶数用sin，奇数用cos

        pe = pe.unsqueeze(0).transpose(0, 1)
        #调整矩阵形状为[max_len,1,d_model]用来适配后面的Transformer的输入格式

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LSTMTransformerModel(nn.Module):

    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, transformer_dim, 
                 transformer_layers, num_heads, sequence_length, output_dim=1, dropout=0.1):
        super(LSTMTransformerModel, self).__init__()

        self.input_dim = input_dim#输入特征维度
        self.lstm_hidden_dim = lstm_hidden_dim#LSTM隐藏层维度，默认128
        self.lstm_layers = lstm_layers#LSTM层数，默认2
        self.transformer_dim = transformer_dim#Transformer的特征维度d_model，默认32
        self.transformer_layers = transformer_layers#Transformer的编码器层数，默认4
        self.num_heads = num_heads#多头注意力头数，默认8，可以理解为不同领域的专家分析数据
        self.sequence_length = sequence_length#输入序列长度，96个时间点
        self.output_dim = output_dim#预测输出维度


        #LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        #投影层 将LSTM输出的高维特征128映射到低维32，便于Transformer计算
        self.projection = nn.Linear(lstm_hidden_dim, transformer_dim)

        #位置编码
        self.pos_encoding = PositionalEncoding(transformer_dim, max_len=sequence_length)

        #Transformer编码器
        # 由多个 TransformerEncoderLayer 堆叠而成
        # 每层包括：多头自注意力，前馈网络，残差连接 + 层归一化
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,#前馈子网络隐藏层维度
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        #Transformer编码器的输出再次进行自注意力计算
        self.attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )


        #预测头，全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(transformer_dim, transformer_dim // 2)
        self.fc2 = nn.Linear(transformer_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):#向前传播

        batch_size, seq_len, _ = x.shape

        #LSTM特征提取
        lstm_out, (h_n, c_n) = self.lstm(x)

        #将LSTM输出从128投影到32
        transformer_input = self.projection(lstm_out)

        transformer_input = transformer_input.transpose(0, 1)
        transformer_input = self.pos_encoding(transformer_input)
        transformer_input = transformer_input.transpose(0, 1)

        #自注意力让序列中任意两个时间步直接交互，可以捕捉长距离依赖
        transformer_out = self.transformer_encoder(transformer_input)


        #额外自注意力精炼，用同一序列q,k,v，再次提取关键特征
        query = transformer_out
        key = transformer_out
        value = transformer_out

        attn_output, attn_weights = self.attention(query, key, value)


        #取最后一个时间步的特征，因为最新的时间步包含了最丰富的近期趋势信息
        h_out = attn_output[:, -1, :]
        output = self.dropout(h_out)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)


        #返回预测值和以及注意力权重，注意力权重可以用于后续分析模型重点关注了那些时刻
        return output, attn_weights

    def get_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'LSTM-Transformer Hybrid',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_layers': self.lstm_layers,
            'transformer_dim': self.transformer_dim,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim
        }