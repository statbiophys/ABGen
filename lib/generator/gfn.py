import torch
import torch.nn.functional as F

from lib.generator.base import GeneratorBase
from lib.model.mlp import MLP
from lib.model.gen_bytenet import GenByteNet

LOGINF = 1000

class TBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        #self.reward_exp_min = args.reward_exp_min
        self.pad_tok = args.pad_token_id
        self.num_tokens = args.vocab_size
        self.max_len = args.gen_max_len
        self.tokenizer = tokenizer
        self.weights = torch.tensor([0.33,0.33,0.33,1.0]).to(args.device)

        if self.args.gen_model_type ==  'mlp':
            self.model = MLP(num_tokens=self.num_tokens, 
                                    num_outputs=self.num_tokens-1, 
                                    num_hid=256,
                                    num_layers=2,
                                    max_len=self.max_len,
                                    dropout=0.1,
                                    args = args,
                                    partition_init=args.gen_partition_init,
                                    causal=args.gen_do_explicit_Z)
            self.model.to(args.device)

        else:
            self.model = GenByteNet(args)
            self.model.to(args.device)
    

        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2,
                           betas=(0.9, 0.999))
    
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), args.gen_Z_learning_rate, weight_decay=args.gen_L2,
                            betas=(0.9, 0.999))

        self.device = args.device
        self.logsoftmax = torch.nn.LogSoftmax(1)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.args = args

    def get_learning_rate(self):
        for g in self.opt.param_groups:
            print(g['lr'])

    def change_learning_rate(self,new_lr):
        for g in self.opt.param_groups:
            g['lr'] = new_lr

    def set_weights(self,weights):
        self.weights = torch.from_numpy(weights).float().to(self.args.device)

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)

        self.opt.step()
        self.opt_Z.step()

        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        return loss, info

    @property
    def Z(self):
        return self.model.Z
    
    def get_loss(self, batch):
        strs, r = zip(*batch["bulk_trajs"])
        s = self.tokenizer.process(strs).to(self.device)
        r = torch.tensor(r).to(self.device)
        
        if self.args.gen_model_type == 'mlp':
            lens = [self.max_len for i in s]
            pol_logits = self.logsoftmax2(self.model(s, None, return_all=True, lens=lens))
            if (self.args.task == "amp" or self.args.task == "random") and s.shape[1] != self.max_len:
                s = F.pad(s, (0, self.max_len - s.shape[1]), "constant", 21)
                mask = s.eq(21)
            else:
                mask = s.eq(self.num_tokens-1)
            s = s.transpose(0,1)
            n = s.shape[0] * s.shape[1]
        else:
            lens = [len(i) for i in strs]

            pol_logits = self.logsoftmax2(self.model(s,return_all = True, lens = lens,weights = self.weights))
            pol_logits = pol_logits.transpose(0,1)
            if (self.args.task == "amp" or self.args.task == "random") and s.shape[1] != self.max_len:
                s = F.pad(s, (0, self.max_len - s.shape[1]), "constant", 21)
                mask = s.eq(21)
            else:
                mask = s.eq(self.num_tokens-1)
            s = s.transpose(0,1)
            n = s.shape[0] * s.shape[1]
        #seq_logits are the logits for every position, shape is num_seqs*max_len by num_tokens-1
        seq_logits = pol_logits.reshape((n,self.num_tokens-1))
        #s1 is the numerical sequence with the padding token changed to 20
        s1 = (s.reshape((-1,))).clamp(0, self.num_tokens-2)
        #here we select the logits that were actually used in the sequence
        seq_logits = seq_logits[torch.arange(n, device=self.device),(s1)]

        seq_logits = seq_logits.reshape(s.shape)
        seq_logits = seq_logits*mask.transpose(0,1).logical_not().float()
        seq_logits = seq_logits.sum(0)

        # p(x) = R/Z <=> log p(x) = log(R) - log(Z) <=> log p(x) - log(Z)
        loss = (self.model.Z + seq_logits - r).pow(2).mean()
      
        return loss, {}

    def forward(self, x, lens, return_all=False, coef=1, pad=2 ,index = 0):
        x = self.tokenizer.process(x).to(self.device)
        if self.args.gen_model_type == "mlp":
            out = self.model(x, None, lens=lens, return_all=return_all,index = index)
            return out
        else:
            out = self.model(x,lens=lens, return_all=return_all,index = index,weights = self.weights)
            return out