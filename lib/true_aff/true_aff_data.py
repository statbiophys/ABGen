import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import math
from Levenshtein import distance
from numpy import random
#from true_aff_oracle import true_aff_oracle
from lib.true_aff.true_aff_oracle import true_aff_oracle
import matplotlib.pyplot as plt
import os

from scipy.stats import pearsonr

class TrueAffDataset(Dataset):

    def __init__(self):
        self.script_dir = os.path.dirname(__file__)
        self.oracle = true_aff_oracle()
        self.wt = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
        self.seqs = self.get_seqs()
        self.mut_count = [distance(i,self.wt) for i in self.seqs]
        self.var_noise = 0.0
        #self.sigma = 1.0
        #self.score = np.load('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts_score_noise:{}_sigma:{}.npy'.format(self.var_noise,self.sigma))
        #self.true_score = np.load('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts_score_noise:{}_sigma:{}.npy'.format(0.0,self.sigma))
        self.method = 'simple'
        file_path = os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(self.var_noise,self.method,0))
        self.score = np.load(file_path)
        file_path = os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(0.0,self.method,0))
        self.true_score = np.load(file_path)
        self.amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.seqs_i = []
        for seq in self.seqs:
            self.seqs_i.append([self.amino_acid_dic[k] for k in seq])
        self.encodings = nn.functional.one_hot(torch.LongTensor(self.seqs_i),num_classes = 20)
        self.encodings = self.encodings.reshape((self.encodings.shape[0],-1))
        self.distances = [distance(i,self.wt) for i in self.seqs]
        self.process_higher_order_muts(0.0)
    
    def __len__(self):
        return(len(self.seqs))

    def __getitem__(self, idx):
        return self.encodings[idx], self.score[idx]

    def process_encoding(self,seqs):
        seqs_i = []
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        encodings = nn.functional.one_hot(torch.LongTensor(seqs_i),num_classes = 20)
        encodings = encodings.reshape((encodings.shape[0],-1))
        return encodings

    def rescore(self,var):
        self.var_noise = var
        self.process_higher_order_muts(var)
        file_path = os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(self.var_noise,self.method,0))
        self.score = np.load(file_path)
        file_path = os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(0.0,self.method,0))
        self.true_score = np.load(file_path)


    def get_seqs(self):
        wt = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
        seqs = []
        file_path = os.path.join(self.script_dir, 'muts.txt')
        with open(file_path,'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs

    def get_higher_order_mutants_seqs(self,n_mutations):
        seqs = []
        file_path = os.path.join(self.script_dir, 'muts_{}.txt'.format(n_mutations))
        with open(file_path,'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs

    def process_higher_order_muts(self,var):
        self.seqs_muts_2 = self.get_higher_order_mutants_seqs(2)
        self.seqs_muts_3 = self.get_higher_order_mutants_seqs(3)
        self.seqs_muts_4 = self.get_higher_order_mutants_seqs(4)
        self.seqs_muts_5 = self.get_higher_order_mutants_seqs(5)
        self.seqs_muts_6 = self.get_higher_order_mutants_seqs(6)
        self.seqs_enc_muts_2 = self.process_encoding(self.seqs_muts_2)
        self.seqs_enc_muts_3 = self.process_encoding(self.seqs_muts_3)
        self.seqs_enc_muts_4 = self.process_encoding(self.seqs_muts_4)
        self.seqs_enc_muts_5 = self.process_encoding(self.seqs_muts_5)
        self.seqs_enc_muts_6 = self.process_encoding(self.seqs_muts_6)
        self.score_muts_2 = np.load(os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(var,self.method,2)))
        self.score_muts_3 = np.load(os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(var,self.method,3)))
        self.score_muts_4 = np.load(os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(var,self.method,4)))
        self.score_muts_5 = np.load(os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(var,self.method,5)))
        self.score_muts_6 = np.load(os.path.join(self.script_dir, 'muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(var,self.method,6)))

    def get_best_seqs(self):
        idx = np.argsort(self.score).tolist()
        idx.reverse()
        best_seqs = [self.seqs[i] for i in idx]
        with open('./sorted_seqs_noise:{}_sigma:{}.txt'.format(self.var_noise,self.sigma),'w') as f:
            for seq in best_seqs:
                f.write(seq+'\n')

    def random_split(self,perc):
        random.seed(0)
        train_size = int(len(self.seqs)*perc)
        train_idx = random.choice(range(len(self.seqs)),size = train_size, replace = False).tolist()
        valid_idx = [i for i in range(len(self.seqs)) if i not in train_idx]
        train_seqs = self.encodings[train_idx]
        train_score = self.score[train_idx]
        valid_seqs = self.encodings[valid_idx]
        valid_score = self.score[valid_idx]
        self.valid_mut_count = [self.mut_count[i] for i in valid_idx]
        self.true_valid_score = self.true_score[valid_idx]
        return torch.Tensor(train_seqs),torch.Tensor(train_score),torch.Tensor(valid_seqs),torch.Tensor(valid_score)

class TrueAffDataset2(Dataset):

    def __init__(self):
        self.oracle = true_aff_oracle2()
        self.seqs = self.get_seqs()
        self.score = np.array(self.oracle.score_with_noise(self.seqs))
        self.true_score = np.array(self.oracle.score_without_noise(self.seqs))
        self.amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.seqs_i = []
        for seq in self.seqs:
            self.seqs_i.append([self.amino_acid_dic[k] for k in seq])
        self.encodings = nn.functional.one_hot(torch.LongTensor(self.seqs_i),num_classes = 20)
        self.encodings = self.encodings.reshape((self.encodings.shape[0],-1))
        self.script_dir = os.path.dirname(__file__)
    
    def __len__(self):
        return(len(self.seqs))

    def __getitem__(self, idx):
        return self.encodings[idx], self.score[idx]

    def get_seqs(self):
        seqs = []
        file_path = os.path.join(self.script_dir, 'muts_2.txt')
        with open(file_path,'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs

    def get_best_seqs(self):
        idx = np.argsort(self.score).tolist()
        idx.reverse()
        best_seqs = [self.seqs[i] for i in idx]
        with open('./sorted_seqs_2.txt','w') as f:
            for seq in best_seqs:
                f.write(seq+'\n')

    def random_split(self,perc):
        random.seed(0)
        train_size = int(len(self.seqs)*perc)
        train_idx = random.choice(range(len(self.seqs)),size = train_size, replace = False).tolist()
        valid_idx = [i for i in range(len(self.seqs)) if i not in train_idx]
        train_seqs = self.encodings[train_idx]
        train_score = self.score[train_idx]
        valid_seqs = self.encodings[valid_idx]
        valid_score = self.score[valid_idx]
        self.true_valid_score = self.true_score[valid_idx]
        return torch.Tensor(train_seqs),torch.Tensor(train_score),torch.Tensor(valid_seqs),torch.Tensor(valid_score)

if __name__ == '__main__':
    np.random.seed(0)
    dataset = TrueAffDataset()
    print(dataset.seqs[0:5])
    print(dataset.score[0:5])
    print(dataset.true_score[0:5])
    idx = np.argsort(dataset.true_score)[:700]
    print(dataset.score[idx])
    best_seqs = [dataset.seqs[i] for i in idx]
    d = [distance(i,dataset.wt) for i in best_seqs]
    print(np.mean(d))
    print(d[:20])