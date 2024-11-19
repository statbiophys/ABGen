import math
from iglm import IgLM
import numpy as np
import torch

class ppost_oracle:
    def __init__(self):
        self.iglm = IgLM()
        self.V = 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYY'
        self.J = 'GQGTMVTVSS'
        self.V1 = 'EVQLVETGGGLVQPGGSLRLSCAAS'
        self.V2 = 'WVRQAPGKGPEWVSV'
        self.V3 = 'KGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAK'
        self.V4 = 'WGQGTLVTVSS'
        self.CDR1 = 'GFTLNSYGIS'
        self.CDR2 = 'IYSDGRRTFYGDSV'
        self.CDR3 = 'GRAAGTFDS'
        self.startCDR1 = len(self.V1)
        self.startCDR2 = len(self.V1+self.CDR1+self.V2)
        self.startCDR3 = len(self.V1+self.CDR1+self.V2+self.CDR2+self.V3)
        self.device = 'cuda'

    def log_likelihood(
        self,
        sequence,
        chain_token,
        species_token,
    ):
        sequence = list(sequence)

        token_seq = [chain_token, species_token] + sequence

        token_seq += [self.iglm.tokenizer.sep_token]

        token_seq = torch.Tensor([
            self.iglm.tokenizer.convert_tokens_to_ids(token_seq)
        ]).int().to(self.iglm.device)

        assert (token_seq != self.iglm.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        eval_start = 1

        logits = self.iglm.model(token_seq).logits
        shift_logits = logits[..., eval_start:-1, :].contiguous()
        shift_labels = token_seq[..., eval_start + 1:].contiguous().long()
        nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )

        return -nll.item()


    def score(self,seqs):
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"
        in_range = (98,107)
        s = 10*np.array([self.iglm.log_likelihood(i,chain_token,species_token,infill_range=in_range) for i in seqs])
        return s

    def score_all(self,seqs):
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"
        #s1 = 119 * np.array([self.iglm.log_likelihood(i,chain_token,species_token) for i in seqs])
        s = np.array([self.log_likelihood(i,chain_token,species_token) for i in seqs])
        return s

    def full_score(self,seqs):
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"
        lengths = [len(i) for i in seqs]
        s = np.array([self.iglm.log_likelihood(i,chain_token,species_token) for i in seqs])
        perplexity = np.exp(-s)
        return perplexity

    def pseudo_likelihood(self,seqs):
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"
        in_range = (self.startCDR1,self.startCDR1+len(self.CDR1))
        log_prob_CDR1 = (len(self.CDR1)+1)*np.array([self.iglm.log_likelihood(i,chain_token,species_token,infill_range=in_range) for i in seqs])
        in_range = (self.startCDR2,self.startCDR2+len(self.CDR2))
        log_prob_CDR2 = (len(self.CDR2)+1)*np.array([self.iglm.log_likelihood(i,chain_token,species_token,infill_range=in_range) for i in seqs])
        in_range = (self.startCDR3,self.startCDR3+len(self.CDR3))
        log_prob_CDR3 = (len(self.CDR3)+1)*np.array([self.iglm.log_likelihood(i,chain_token,species_token,infill_range=in_range) for i in seqs])
        return log_prob_CDR1 + log_prob_CDR2 + log_prob_CDR3
                    
    def make_seq(self,cdr1 = None,cdr2 = None,cdr3 = None):
        seq = self.V1
        if cdr1 is not None:
            seq += cdr1 + self.V2
        else:
            seq += self.CDR1 + self.V2

        if cdr2 is not None:
            seq += cdr2 + self.V3
        else:
            seq += self.CDR2 + self.V3

        if cdr3 is not None:
            seq += cdr3 + self.V4
        else:
            seq += self.CDR3 + self.V4
        return seq

def main():
    ora = ppost_oracle()
    full = ora.V + 'CAK' + ora.CDR3 + 'W' + ora.J
    full_2 = ora.V + 'CAK' + 'GRACGTFDSWWWWWWWWWWW' + 'W' + ora.J
    #print(ora.score([full]*8))
    print(ora.score_all([full]*8))
    print(ora.score_all([full_2]*8))
    

if __name__ == '__main__':
    main()
