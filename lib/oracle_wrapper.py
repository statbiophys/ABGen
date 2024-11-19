import numpy as np
import os
import pandas as pd
import olga.load_model as olga_load_model
import olga.sequence_generation as seq_gen
from Bio.Seq import translate
import olga.generation_probability as pgen
import multiprocessing as mp
from SASA.SASA_oracle import SASA_oracle_2
import warnings
from lib.utils.env import get_tokenizer
from torch import nn
import torch
import esm
from lib.model.bytenet import ByteNet
from lib.Covid.exactgp import get_covid_gp_model
from lib.Covid.exact12 import get_covid_gp_model_12
from lib.post.ppost_oracle import ppost_oracle
from lib.true_aff.true_aff_gp import get_trueaff_gp_model
from lib.true_aff.true_aff_gp import update_gp_model
from lib.true_aff.true_aff_oracle import true_aff_oracle
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.special import comb
from scipy.stats import norm

from Levenshtein import distance

def get_oracle(args):
    return OracleWrapper(args)

class small_nn(nn.Module):
    def __init__(self,args):
        super(small_nn, self).__init__()
        self.hidden_size = 64
        self.linear_1 = nn.Linear(args.gen_max_len*args.num_tokens,self.hidden_size)
        self.RELU = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_4 = nn.Linear(self.hidden_size,1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.RELU(x)
        x = self.linear_2(x)
        x = self.RELU(x)
        x = self.linear_4(x)
        return x

class args_nn:
    def __init__(self):
        self.task = 'random'
        self.gen_max_len = 25
        self.num_tokens = 22
        self.device = torch.device('cuda')

class args_cnn:
    def __init__(self):
        self.task = 'random'
        self.gen_max_len = 27
        self.num_tokens = 22
        self.small_embedding = 16
        self.device = torch.device('cuda')
        self.hidden_size = 64
        self.pad_token_id = 21
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.n_layers = 4
        self.max_r = 4
        self.task_train = 'nmasked'
        self.epochs = 20
        self.model = 'cnn'

class OracleWrapper:
    def __init__(self, args):
        self.V = 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAK'
        self.J = 'WGQGTLVTVSS'

        self.V1 = 'EVQLVETGGGLVQPGGSLRLSCAAS'
        self.V2 = 'WVRQAPGKGPEWVSV'
        self.V3 = 'KGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAK'
        self.V4 = 'WGQGTLVTVSS'

        self.true_aff_seed = 'KAPPLEDLF'

        self.args = args
        self.args_cnn = args_cnn()

        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model.eval()
        self.esm_model.cuda()
        ### tokenizer
        self.tok = get_tokenizer(args)
        ### neural network to predict SASA
        self.SASA_oracle = SASA_oracle_2()
        self.post_ora = ppost_oracle()
        self.enc_type = 'esm_t6'
        self.aff_gp_model,self.aff_likelihood = get_covid_gp_model(self.enc_type)
        self.true_aff_gp_var = 1.0
        self.true_aff_perc = 0.8
        self.true_aff_method = 'simple'
        self.true_aff_model,self.true_aff_likelihood = get_trueaff_gp_model(self.true_aff_gp_var,self.true_aff_perc,self.true_aff_method)
        self.true_aff_ora = true_aff_oracle()
        self.amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.weights = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        self.temp_sasa = 1.0
        self.temp_pgen = 1.0
        self.temp_ppost = 1.0
        self.temp_aff = 1.0
        self.temp_immuno = 1.0
        self.global_temp = 1.0
        self.aff_var_temp = 1.0
        self.beta = 1.0
        self.gen_all_cdr = True
        self.use_hum = True
        self.covidseed = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'

        self.antBO_min_sol = 0
        self.antBO_min_hum = -130
        self.true_aff_best_score = 1.0

    def set_weights(self,w):
        self.weights = w

    def set_antBO_thresholds(self,t):
        self.antBO_min_sol = t[0]
        self.antBO_min_hum = t[1]
        self.weights[4] = t[2]

    def update_true_aff_gp(self,var,sigma,method):
        self.true_aff_gp_var = var
        self.true_aff_perc = sigma
        if method == 'hard':
            self.true_aff_best_score = -1.6628364105557754
            self.true_aff_ora.load_hard_pwm()
        else:
            self.true_aff_best_score = -1.6628364105557754
            self.true_aff_ora.load_simple_pwm()
        self.true_aff_method = method
        self.true_aff_model,self.true_aff_likelihood = get_trueaff_gp_model(self.true_aff_gp_var,self.true_aff_perc,self.true_aff_method)

    def get_aff_score(self,seqs):
        good_ind = []
        good_seqs = []
        restrict = self.args.restrict_aff
        cdr = 'CAKGRAAGTFDSW'
        for i in range(len(seqs)):
            if len(seqs[i]) == 13:
                good_ind.append(i)
                good_seqs.append(seqs[i])
        
        for i in range(len(good_seqs)):
            good_seqs[i] = 'GFTLNSYGISVIYSDGRRTFYGDSVK'+good_seqs[i][3:-1]
        if len(good_seqs) > 0:
        ### tokenize sequences
            length = torch.tensor([len(i) for i in good_seqs]).to(self.args.device)

            x = self.tok2.process(good_seqs).to(self.cov_args.device)
            x = nn.functional.one_hot(x,num_classes=-1).to(torch.float32)
            x = x.reshape(x.shape[0],-1)

            ### preprocess for aff prediction
            
            with torch.no_grad():
                aff_r = self.aff_model(x,length).squeeze(dim= (0)).cpu()

        count = 0
        reward = []
        for i in range(len(seqs)):
            if i in good_ind:
                if restrict:
                    d = distance(seqs[i],cdr)
                    if d > 6:
                        reward.append(0)
                    else:
                        reward.append(7 - aff_r[count].item())
                    count += 1
                else:
                    reward.append(7 - aff_r[count].item())
                    count += 1
            else:
                reward.append(-30/self.weights[2])
        reward = torch.tensor(reward).cpu()
        return reward

    def process_esm(self,seqlist):
        batch_converter = self.esm_alphabet.get_batch_converter()
        batch_size = 1
        num_of_batches = int(len(seqlist)/batch_size)
        res = torch.empty(0,len(seqlist[0])+2,320)
        for j in range(num_of_batches):
            data = []
            for i in range(batch_size):
                data.append(('protein{}'.format(i),seqlist[batch_size*j + i]))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to('cuda')
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
            token_representations = results["representations"][6]
            res = torch.cat((res,token_representations.cpu()))
        return res

    def process_esm_no_batch(self,seqlist):
        batch_converter = self.esm_alphabet.get_batch_converter()
        data = []
        for i in range(len(seqlist)):
            data.append(('protein{}'.format(i),seqlist[i]))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to('cuda')
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]
        return token_representations.cpu()

    
    def get_sasa_score(self,seqs):
        if self.gen_all_cdr:
            full_seqs = [self.get_full_seq_2(i) for i in seqs]
        else:
            full_seqs = [self.get_fullSeq(i) for i in seqs]
        sasa_r = -self.SASA_oracle(full_seqs)
        return sasa_r

    def get_full_sasa_score(self,full_seqs):
        sasa_r = -self.SASA_oracle(full_seqs)
        return sasa_r

    def get_aff_score_gp_esm(self,seqs):
        if self.gen_all_cdr:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 33]
            full_seqs = [self.get_full_seq_2(i) for i in seqs if len(i) == 33]
        else:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 13]
            full_seqs = [self.get_fullSeq(i) for i in seqs if len(i) == 13]
        if len(full_seqs) > 0:
            res = self.process_esm(full_seqs)
            seqs = torch.cat((res[:,26:36,:],res[:,50:66,:],res[:,99:108,:]),1).cuda()
            seqs = seqs.reshape(seqs.shape[0],-1)
            with torch.no_grad():
                pred = self.aff_gp_model(seqs)
                aff = pred.mean.cpu().numpy()
                var = np.sqrt(pred.variance.cpu().numpy())
        return aff,var

    def get_aff_score_gp_ohe(self,seqs):
        if self.gen_all_cdr:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 33]
            full_seqs = [self.get_full_seq_2(i) for i in seqs if len(i) == 33]
        else:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 13]
            full_seqs = [self.get_fullSeq(i) for i in seqs if len(i) == 13]
        if len(full_seqs) > 0:
            seqs_i = []
            for seq in full_seqs:
                seqs_i.append([self.amino_acid_dic[k] for k in seq])
            one_hot_vec = nn.functional.one_hot(torch.LongTensor(seqs_i),num_classes = 20)
            one_hot_vec = one_hot_vec.reshape((one_hot_vec.shape[0],-1)).cuda()
            with torch.no_grad():
                pred = self.aff_gp_model(one_hot_vec)
                aff = pred.mean.cpu().numpy()
                var = np.sqrt(pred.variance.cpu().numpy())
        return aff,var

    def get_ppost_score(self,seqs):
        if self.gen_all_cdr:
            full_seqs = [self.get_full_seq_2(i) for i in seqs]
            post_r = self.post_ora.score_all(full_seqs)
        else:
            full_seqs = [self.get_fullSeq(i) for i in seqs]
            post_r = self.post_ora.score_all(full_seqs)
        return torch.tensor(post_r)

    def get_ppost_full_score(self,full_seqs):
        post_r = self.post_ora.score_all(full_seqs)
        return torch.tensor(post_r)

    def get_len_score(self,seqs):
        if self.gen_all_cdr:
            len_r = [0 if len(i) == 33 else -30 for i in seqs]
        else:
            len_r = [0 if len(i) == 13 else -30 for i in seqs]
        return np.array(len_r)

    def get_true_aff_len_score(self,seqs):
        len_r = [0 if len(i) == 9 else -100 for i in seqs]
        return np.array(len_r)

    def compute_charge(self,seq):
        charge = 0.0
        for s in seq:
            if s in ['R','K']:
                charge += 1
            elif s in ['H']:
                charge += 0.1
            elif s in ['D','E']:
                charge += -1
        return charge

    def get_dev_score(self,seqs):
        if self.gen_all_cdr:
            full_seqs = [self.get_full_seq_2(i) for i in seqs]
        else:
            full_seqs = [self.get_fullSeq(i) for i in seqs]
        analysis = [ProteinAnalysis(i) for i in full_seqs]
        instability = np.array([i.instability_index() for i in analysis])
        hydrophobicity = np.array([i.gravy() for i in analysis])
        charge = np.array([self.compute_charge(i) for i in full_seqs])
        return instability,hydrophobicity,charge

    def get_trueaff_score(self,seqs):
        seqs_i = []
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        encodings = nn.functional.one_hot(torch.LongTensor(seqs_i),num_classes = 20)
        encodings = encodings.reshape((encodings.shape[0],-1)).cuda()
        with torch.no_grad():
            pred = self.true_aff_model(encodings)
            aff = pred.mean.cpu().numpy()
            var = np.sqrt(pred.variance.cpu().numpy())
        return aff,var

    def score_true_aff_seqs(self,seqs):
        ### compute cov similarity scores
        if self.gen_all_cdr:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 33]
            good_seqs = [i for i in seqs if len(i) == 33]
            score = np.ones(len(seqs)) * (-500)
        if len(good_seqs) > 1:
            sasa_r = (self.get_sasa_score(good_seqs) + 6)*self.temp_sasa
            len_r = self.get_len_score(good_seqs)
            dist_r = self.get_dist_score(good_seqs)
            aff_r, var_r = self.get_trueaff_score(good_seqs)
            ppost_r = (self.get_ppost_score(good_seqs))*self.temp_ppost
            final_score = ppost_r + self.weights[3] * (self.weights[1] * sasa_r + self.weights[2] * (-aff_r + self.weights[4] * var_r)) + len_r + dist_r
            score[good_idx] = final_score
        return score

    def score_true_aff_seqs_indiv(self,seqs):
        self.gen_all_cdr = True
        sasa_r = self.get_sasa_score(seqs)
        aff_r,var_r = self.get_trueaff_score(seqs)
        aff_r = 12 - aff_r
        return sasa_r,aff_r,var_r

    def update_true_aff_gp_model(self,seqs,score):
        seqs_i = []
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        encodings = nn.functional.one_hot(torch.LongTensor(seqs_i),num_classes = 20)
        encodings = encodings.reshape((encodings.shape[0],-1))
        self.true_aff_model,self.true_aff_likelihood = update_gp_model(self.true_aff_model,encodings,score)

    def get_dist_score(self,seqs):
        dist = [0 if distance(i,self.covidseed) < 7 else (-np.log(20) * (distance(i,self.covidseed) - 6) - np.log(comb(33,min(int(33/2),distance(i,self.covidseed))) - comb(33,6)) - 20) for i in seqs]
        return np.array(dist)

    def __call__(self, seqs, batch_size=256, return_aff = False):       
        ### compute cov similarity scores
        if self.gen_all_cdr:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 33]
            good_seqs = [i for i in seqs if len(i) == 33]
            score = np.ones(len(seqs)) * (-500)
        if len(good_seqs) > 1:
            sasa_r = (self.get_sasa_score(good_seqs))*self.temp_sasa
            dist_r = self.get_dist_score(good_seqs)
            if self.enc_type == 'esm_t6':
                aff_r,var_r = self.get_aff_score_gp_esm(good_seqs)
            else:
                aff_r,var_r = self.get_aff_score_gp_ohe(good_seqs)
            ##normalize scores
            ppost_r = (self.get_ppost_score(good_seqs))*self.temp_ppost
            final_score = ppost_r + self.weights[3] * (self.weights[1] * sasa_r + self.weights[2] * (-aff_r + self.weights[4] * var_r)) + dist_r
            
            score[good_idx] = final_score
        if return_aff:
            return score, aff_r
        else:
            return score

    def expected_improvement(self,mean,std):
        best = -self.true_aff_best_score
        a = (mean - best)/std
        rv = norm()
        return (mean - best) * rv.cdf(a) + std * rv.pdf(a)

    def trust_region_true_aff(self,seqs):
        if self.gen_all_cdr:
            good_idx = [i for i in range(len(seqs)) if len(seqs[i]) == 33]
            good_seqs = [i for i in seqs if len(i) == 33]
            score = np.ones(len(seqs)) * (-500)
        if len(good_seqs) > 1:
            sasa_r = (self.get_sasa_score(good_seqs))
            ppost_r = (self.get_ppost_score(good_seqs).numpy())
            aff_r, var_r = self.get_trueaff_score(good_seqs)
            aff_r = self.expected_improvement(-aff_r,var_r)
            valid = (sasa_r > self.antBO_min_sol) * (ppost_r > self.antBO_min_hum)
        return aff_r,valid,sasa_r,ppost_r

    def return_indiv_scores(self,seqs):
        sasa_r = self.get_sasa_score(seqs)
        if self.enc_type == 'esm_t6':
            aff_r,var_r = self.get_aff_score_gp_esm(seqs)
        else:
            aff_r,var_r = self.get_aff_score_gp_ohe(seqs)
        aff_r = 7.0 - aff_r
        return sasa_r,aff_r,var_r

    def get_fullSeq(self,cdrAa):
        if len(cdrAa) == 0:
            cdrAa = '%'

        if cdrAa[-1] == '%':
            cdrAa = cdrAa[:-1]
        return self.V + cdrAa + self.J

    def get_full_seq_2(self,cdrs):
        if len(cdrs) == 0:
            print(cdrs)
            cdrs = '%'

        if cdrs[-1] == '%':
            cdrs = cdrs[:-1]
        cdr1 = cdrs[:10]
        cdr2 = cdrs[10:24]
        cdr3 = cdrs[24:]
        return self.V1 + cdr1 + self.V2 + cdr2 + self.V3 + cdr3 + self.V4
    

class Vocabb:
    def __init__(self, alphabet) -> None:
        self.stoi = {}
        self.itos = {}
        for i, alphabet in enumerate(alphabet):
            self.stoi[alphabet] = i
            self.itos[i] = alphabet

if __name__ == '__main__':
    a = OracleWrapper(alphabet)
