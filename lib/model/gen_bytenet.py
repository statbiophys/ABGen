import numpy as np

from torch import nn
from torch import from_numpy
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import math

class args:

    def __init__(self):
        self.vocab_size = 22
        self.gen_small_embedding = 16
        self.layer_norm_eps = 1e-12
        self.cnn_hidden_size = 16
        self.pad_token_id = 21
        self.hidden_dropout_prob = 0.1
        self.gen_max_len = 35
        self.cnn_n_layers = 4
        self.cnn_max_r = 1
        self.device = 'cuda'

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
    
        self.AAEmbeddings = torch.nn.Embedding(args.vocab_size, args.gen_small_embedding, padding_idx=self.pad_token_id)
        self.PositionEmbeddings = torch.nn.Embedding(args.gen_max_len, args.gen_small_embedding, padding_idx=0) # here padding_idx is always 0
        self.PositionEmbeddings2 = torch.nn.Embedding(args.gen_max_len, args.gen_small_embedding, padding_idx=0) # here padding_idx is always 0
        
        self.LayerNorm = torch.nn.LayerNorm(args.gen_small_embedding, eps=args.layer_norm_eps)
        self.Dropout = torch.nn.Dropout(args.hidden_dropout_prob)
        self.UpEmbedding = torch.nn.Linear(args.gen_small_embedding,args.cnn_hidden_size*2)
        self.args = args

    def forward(self, src,length):
        inputs_embeds = self.AAEmbeddings(src)
        
        position_ids = self.create_position_ids_from_input_ids(src,length, self.pad_token_id)
        position_embeddings = self.PositionEmbeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        embedding = self.Dropout(self.LayerNorm(embeddings))
        return self.Dropout(self.UpEmbedding(embedding))
        
    def create_position_ids_from_input_ids(self, input_ids,length, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.
        """
        mask = input_ids.ne(padding_idx).int()
 
        pid = torch.cumsum(mask, dim=1).long() * mask
        '''
        pid2 = torch.cumsum(mask, dim=1).long() * mask
        for i in range(pid.shape[0]):
            pid2[i] = length[i] - pid[i] + 2
        pid2 *= mask
        '''
        return pid #,pid2

class MaskedConv1d(nn.Conv1d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kL = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kL // 2:] = 0
        #self.mask[:, :, kH // 2 + 1:] = 0

    def simplify(self):
        self.weight.data[:,:,:] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

class TransposeNorm(nn.Module):
    def __init__(self,args,feature_size):
        super(TransposeNorm,self).__init__()
        self.layer_norm = torch.nn.LayerNorm(feature_size,eps = args.layer_norm_eps)

    def forward(self,x):
        x = torch.transpose(x,1,2)
        x = self.layer_norm(x)
        x = torch.transpose(x,1,2)
        return x

class ByteNetBlock(nn.Module):
    def __init__(self,args,r):
        super(ByteNetBlock,self).__init__()
        self.kernel_size = 17
        #self.layer_norm_1 = torch.nn.LayerNorm(args.cnn_hidden_size*2,eps = args.layer_norm_eps)
        self.transpose_norm_1 = TransposeNorm(args, args.cnn_hidden_size*2)
        #self.batch_norm_1 = torch.nn.BatchNorm1d(args.cnn_hidden_size*2,eps = args.layer_norm_eps)
        self.RELU_1 = torch.nn.ReLU()
        self.conv_1 = torch.nn.Conv1d(in_channels = args.cnn_hidden_size*2, out_channels = args.cnn_hidden_size, kernel_size = 1)
        #self.layer_norm_2 = torch.nn.LayerNorm(args.cnn_hidden_size,eps = args.layer_norm_eps)
        self.transpose_norm_2 = TransposeNorm(args, args.cnn_hidden_size)
        #self.batch_norm_2 = torch.nn.BatchNorm1d(args.cnn_hidden_size,eps = args.layer_norm_eps)
        self.RELU_2 = torch.nn.ReLU()
        #self.conv_2 = torch.nn.Conv1d(args.cnn_hidden_size,args.cnn_hidden_size,kernel_size = 3, dilation = r)
        self.conv_2 = MaskedConv1d('A',args.cnn_hidden_size,args.cnn_hidden_size,kernel_size = self.kernel_size, dilation = r)
        p = _same_pad(self.kernel_size,r)
        if p % 2 == 1:
            padding = [p // 2 + 1, p // 2]
        else:
            padding = (p // 2, p // 2)
        self.pad = nn.ConstantPad1d(padding, 0.)
        self.RELU_3 = torch.nn.ReLU()
        #self.layer_norm_3 = torch.nn.LayerNorm(args.cnn_hidden_size, eps=args.layer_norm_eps)
        self.transpose_norm_3 = TransposeNorm(args, args.cnn_hidden_size)
        #self.batch_norm_3 = torch.nn.BatchNorm1d(args.cnn_hidden_size,eps = args.layer_norm_eps)
        self.conv_3 = torch.nn.Conv1d(in_channels = args.cnn_hidden_size, out_channels = args.cnn_hidden_size*2, kernel_size = 1)

    def forward(self,x):
        y = x
        x = self.RELU_1(self.transpose_norm_1(x))
        x = self.conv_1(x)
        x = self.RELU_2(self.transpose_norm_2(x))
        x = self.pad(x)
        x = self.conv_2(x)
        x = self.RELU_3(self.transpose_norm_3(x))
        x = self.conv_3(x)
        x = x + y
        return x

class ByteNetDecoder(torch.nn.Module):
    
    def __init__(self, args):
        super(ByteNetDecoder,self).__init__()
        self.dense = torch.nn.Linear(args.cnn_hidden_size*2 + 1, args.cnn_hidden_size*2 + 1)
        self.layer_norm = torch.nn.LayerNorm(args.cnn_hidden_size*2 + 1, eps=args.layer_norm_eps)
        self.RELU = torch.nn.ReLU()
        self.decoder = torch.nn.Linear(args.cnn_hidden_size*2 + 1, args.vocab_size-1)

    def forward(self, features, **kwargs):
        x = self.dense(features)

        x = self.RELU(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x

class GenByteNet(nn.Module):
    def __init__(self,args):
        super(GenByteNet,self).__init__()
        self.EmbeddingLayer = AbEmbeddings(args)
        self.r_list = [2**i if 2**i <= args.cnn_max_r else args.cnn_max_r for i in range(args.cnn_n_layers)]
        self.blocks = nn.Sequential(*[ByteNetBlock(args,i) for i in self.r_list])
        self.decoder = ByteNetDecoder(args)
        partition_init = -300
        self.args = args
        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)
        self.seed = torch.tensor([ 6,  5, 17, 10, 12, 16, 20,  6,  8, 16,  8, 20, 16,  3,  6, 15, 15, 17,
          5, 20,  6,  3, 16, 18,  6, 15,  1,  1,  6, 17,  5,  3, 16, 21, 21]).to(args.device)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return list(self.EmbeddingLayer.parameters()) + list(self.blocks.parameters()) + list(self.decoder.parameters())

    def Z_param(self):
        return [self._Z]

    def forward(self,x,lens,weights,index=0,return_all = False):
        [n,m] = x.size()
        dist = self.get_mut_count(x)
        if return_all:
            y = self.EmbeddingLayer(x,lens)
            y = torch.transpose(y,1,2)
            y = self.blocks(y)
            y = torch.transpose(y,1,2)
            #y = torch.cat([y,weights.repeat(n,self.args.gen_max_len,1)],dim=2)
            #y = y.reshape(y.shape[0],self.args.gen_max_len*2*self.args.cnn_hidden_size)
            dist = dist.reshape((dist.shape[0],dist.shape[1],1))
            y = torch.cat((y,dist),dim = 2)
            y = self.decoder(y)
            return y

        x = self.EmbeddingLayer(x,lens)
        x = torch.transpose(x,1,2)
        #x = x.reshape(x.shape[0],self.args.gen_max_len*2*self.args.cnn_hidden_size)
        x = self.blocks(x)
        #w = weights.repeat(n,1)
        #x = torch.cat([x[:,:,index],weights.repeat(n,1)],dim=1)
        dist = dist.reshape((dist.shape[0],1,dist.shape[1]))
        x = torch.cat((x,dist),dim = 1)
        x = self.decoder(x[:,:,index])
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

    def get_mut_count(self,x):
        n,m = x.shape
        seeds = self.seed.repeat(n,1)
        diff = x != seeds
        diff = torch.cumsum(diff, dim = 1)
        return diff

def test_gen():
    torch.manual_seed(0)
    a = args()
    m = GenByteNet(a)
    m.eval()
    seqs = torch.tensor([[1,2,3,4,5,6,7,7,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21],[1,2,3,4,5,6,7,7,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21]]).int()
    lens = torch.tensor([[8,8]]).int()
    w = torch.tensor([1,0,0]).float()
    print(lens.shape)
    print(seqs.shape)
    r = m(seqs,return_all = True, lens = lens,weights = w)
    r2 = m.forward2(seqs,return_all = True, lens = lens,weights = w)
    print(r[0,0])
    print(r2.transpose(0,1)[0,0])

if __name__ == '__main__':
    test_gen()