import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers,args, max_len=60, dropout=0.1,
                 partition_init=150.0,
                 **kwargs):
        super().__init__()
        self.input = nn.Linear((num_tokens-1) * max_len + args.gen_max_len, num_hid)
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(num_hid, num_hid))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(num_hid, num_outputs)
        #self.output = nn.Linear(num_tokens * max_len, num_outputs)
        self.max_len = max_len
        self.num_tokens = num_tokens
        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)
        self.num_tok = num_tokens
        self.max_len = max_len
        self.args = args


    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output.parameters())

    def output_params(self):
        return list(self.output.bias)

    def Z_param(self):
        return [self._Z]

    def preprocess(self,x,lens,index=0):
        pos = (torch.ones(x.shape[0])*index).to(torch.int64)
        pos = F.one_hot(pos,num_classes = self.args.gen_max_len).to(torch.float32).to(self.args.device)
        inp = F.one_hot(x, num_classes=self.num_tokens)[:, :, :-1].to(torch.float32)
        inp = inp.reshape(x.shape[0], -1).to(self.args.device)
        inp2 = torch.cat((inp,pos),dim=1)
        return inp2

    def preprocess_mask(self,x,lens,index=0):
        pos = (torch.ones(x.shape[0])*index).to(torch.int64)
        pos = F.one_hot(pos,num_classes = self.args.gen_max_len).to(torch.float32).to(self.args.device)
        inp = F.one_hot(x, num_classes=self.num_tokens)[:, :, :-1].to(torch.float32)
        inp = inp.reshape(x.shape[0], -1).to(self.args.device)
        mask = torch.cat((torch.ones(x.shape[0], (self.num_tokens-1) * index), torch.zeros(x.shape[0], (self.num_tokens-1) * (lens[0] - index))), axis=1)
        mask = mask.to(self.args.device)
        masked_input = mask * inp
        inp2 = torch.cat((masked_input,pos),dim=1)
        return inp2


    def forward(self, x, mask, return_all=False, lens=None,index=0):
        if return_all:
            outputs = []
            for i in range(lens[0]):
                '''
                mask_2 = torch.tensor([[21]*lens[0]]*x.shape[0])
                mask_2 = F.one_hot(mask_2, num_classes=self.num_tokens).to(x.device)
                mask_2 = mask_2.reshape(x.shape[0],-1)
                masked_input = torch.cat((x[0:x.shape[0], 0:(self.num_tokens) * i],mask_2[0:x.shape[0], (self.num_tokens) * i:lens[0]*(self.num_tokens)]),axis = 1)
                mask = torch.cat((torch.ones(x.shape[0], (self.num_tokens-1) * i), torch.zeros(x.shape[0], (self.num_tokens-1) * (lens[0] - i))), axis=1)
                mask = mask.to(x.device)
                print(x[0].reshape(lens[0],-1))
                masked_input = mask * x
                '''
                masked_input = self.preprocess_mask(x,lens,i)
                out = self.input(masked_input)
                out = self.hidden(out)
                outputs.append(self.output(out).unsqueeze(0))
                #outputs.append(self.output(masked_input).unsqueeze(0))
            out = torch.cat(outputs, axis=0)
            return out
        x = self.preprocess(x,lens,index)
        out = self.input(x)
        out = self.hidden(out)
        return self.output(out)

