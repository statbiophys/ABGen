
import ablang
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import from_numpy
from torch.cuda import is_available
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from lib.utils.env import get_tokenizer
import math


def _same_pad(k=1, dil=1):
    # assumes stride length of 1
    # p = math.ceil((l - 1) * s - l + dil*(k - 1) + 1)
    p = math.ceil(dil*(k - 1))
    #print("padding:", p)
    return p

class AbEmbeddings(torch.nn.Module):
    """
    Residue embedding and Positional embedding
    """
    
    def __init__(self, args):
        super().__init__()
        self.pad_token_id = args.pad_token_id
        
        self.AAEmbeddings = torch.nn.Embedding(args.num_tokens+1, args.small_embedding, padding_idx=self.pad_token_id)
        self.PositionEmbeddings = torch.nn.Embedding(args.gen_max_len, args.small_embedding, padding_idx=0) # here padding_idx is always 0
        self.PositionEmbeddings2 = torch.nn.Embedding(args.gen_max_len, args.small_embedding, padding_idx=0) # here padding_idx is always 0
        
        self.LayerNorm = torch.nn.LayerNorm(args.small_embedding, eps=args.layer_norm_eps)
        self.Dropout = torch.nn.Dropout(args.hidden_dropout_prob)
        self.UpEmbedding = torch.nn.Linear(args.small_embedding,args.hidden_size*2)
        self.args = args

    def forward(self, src,length):
        
        inputs_embeds = self.AAEmbeddings(src)
        
        position_ids,position_ids_2 = self.create_position_ids_from_input_ids(src,length, self.pad_token_id)
        position_embeddings = self.PositionEmbeddings(position_ids)
        position_embeddings_2 = self.PositionEmbeddings2(position_ids_2)

        embeddings = inputs_embeds + position_embeddings + position_embeddings_2

        embedding = self.Dropout(self.LayerNorm(embeddings))
        return self.Dropout(self.UpEmbedding(embedding))
        
    def create_position_ids_from_input_ids(self, input_ids,length, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.
        """
        mask = input_ids.ne(padding_idx).int()
 
        pid = torch.cumsum(mask, dim=1).long() * mask
        pid2 = torch.cumsum(mask, dim=1).long() * mask
        for i in range(pid.shape[0]):
            pid2[i] = length[i] - pid[i] + 2
        pid2 *= mask
        return pid,pid2

class ByteNetBlock(nn.Module):
    def __init__(self,args,r):
        super(ByteNetBlock,self).__init__()
        #self.layer_norm_1 = torch.nn.LayerNorm(args.hidden_size*2,eps = args.layer_norm_eps)
        self.batch_norm_1 = torch.nn.BatchNorm1d(args.hidden_size*2,eps = args.layer_norm_eps)
        self.RELU_1 = torch.nn.ReLU()
        self.conv_1 = torch.nn.Conv1d(in_channels = args.hidden_size*2, out_channels = args.hidden_size, kernel_size = 1)
        #self.layer_norm_2 = torch.nn.LayerNorm(args.hidden_size,eps = args.layer_norm_eps)
        self.batch_norm_2 = torch.nn.BatchNorm1d(args.hidden_size,eps = args.layer_norm_eps)
        self.RELU_2 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv1d(args.hidden_size,args.hidden_size,kernel_size = 3, dilation = r)
        p = _same_pad(3,r)
        if p % 2 == 1:
            padding = [p // 2 + 1, p // 2]
        else:
            padding = (p // 2, p // 2)
        self.pad = nn.ConstantPad1d(padding, 0.)
        self.RELU_3 = torch.nn.ReLU()
        #self.layer_norm_3 = torch.nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.batch_norm_3 = torch.nn.BatchNorm1d(args.hidden_size,eps = args.layer_norm_eps)
        self.conv_3 = torch.nn.Conv1d(in_channels = args.hidden_size, out_channels = args.hidden_size*2, kernel_size = 1)

    def forward(self,x):
        y = x
        x = self.RELU_1(self.batch_norm_1(x))
        x = self.conv_1(x)
        x = self.RELU_2(self.batch_norm_2(x))
        x = self.pad(x)
        x = self.conv_2(x)
        x = self.RELU_3(self.batch_norm_3(x))
        x = self.conv_3(x)
        x = x + y
        return x

class ByteNetDecoder(torch.nn.Module):
    """
    Head for masked sequence prediction.
    """

    def __init__(self, args):
        super(ByteNetDecoder,self).__init__()
        self.size = args.hidden_size*args.gen_max_len*2
        self.dense = torch.nn.Linear(args.hidden_size*2, args.hidden_size*2)
        self.layer_norm = torch.nn.LayerNorm(args.hidden_size*2, eps=args.layer_norm_eps)
        self.RELU = torch.nn.ReLU()
        self.decoder = torch.nn.Linear(args.hidden_size*2, 20)

    def forward(self, features, **kwargs):
        x = self.dense(features)

        x = self.RELU(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class ByteNetDecoder2(torch.nn.Module):
    """
    Head for masked sequence prediction.
    """

    def __init__(self, args):
        super(ByteNetDecoder2,self).__init__()
        self.size = args.hidden_size*args.gen_max_len*2
        self.dense = torch.nn.Linear(self.size, 256)
        self.layer_norm = torch.nn.LayerNorm(256, eps=args.layer_norm_eps)
        self.RELU = torch.nn.ReLU()
        self.decoder = torch.nn.Linear(256, 1)

    def forward(self, features, **kwargs):
        x = self.dense(features)

        x = self.RELU(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x



class ByteNet(nn.Module):
    def __init__(self,args):
        super(ByteNet,self).__init__()
        self.EmbeddingLayer = AbEmbeddings(args)
        self.r_list = [2**i if 2**i <= args.max_r else args.max_r for i in range(args.n_layers)]
        self.blocks = nn.Sequential(*[ByteNetBlock(args,i) for i in self.r_list])
        #self.decoder = ByteNetDecoder(args)
        self.decoder_2 = ByteNetDecoder2(args)
        self.task = args.task_train


    def forward(self,x,l):
        if self.task == 'masked':
            x = self.EmbeddingLayer(x,l)
            x = self.blocks(x)
            x = self.decoder(x)
            return x
        else:
            x = self.EmbeddingLayer(x,l)
            x = torch.transpose(x,1,2)
            x = self.blocks(x)
            x = torch.transpose(x,1,2)
            x = x.reshape(x.shape[0],-1)
            x = self.decoder_2(x)
            return x

    def save_upper_params(self):
        torch.save(self.EmbeddingLayer.state_dict(), './model_params/final_emb_10.pt')
        torch.save(self.blocks.state_dict(),'./model_params/final_blocks_10.pt')

    def load_upper_params(self):
        self.EmbeddingLayer.load_state_dict(torch.load('./model_params/final_emb_10.pt'))
        self.blocks.load_state_dict(torch.load('./model_params/final_blocks_10.pt'))

    def freeze_blocks(self):
        for params in self.blocks.parameters():
            params.requires_grad = False

    def unfreeze_blocks(self):
        for params in self.blocks.parameters():
            params.requires_grad = True