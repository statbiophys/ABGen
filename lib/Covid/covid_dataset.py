import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import from_numpy
from torch.cuda import is_available
import sys 
import os
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from lib.utils.env import get_tokenizer
#from utils.env import get_tokenizer
import torch
import math
from tqdm import tqdm
from Levenshtein import distance
import esm
from numpy import random
from antiberty import AntiBERTyRunner
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

class args:
    def __init__(self):
        self.task = 'random'
        self.gen_max_len = 37
        self.num_tokens = 22
        self.task_train = 'nmasked'
        self.model = 'mlp'


script_dir = os.path.dirname(__file__)

AAY49_CDR1 = 'GFTLNSYGIS'
AAY49_CDR2 = 'VIYSDGRRTFYGDSVK'
AAY49_CDR3 = 'GRAAGTFDS'
AAY49_SEED = 'GFTLNSYGISVIYSDGRRTFYGDSVKGRAAGTFDS'
AAY49_SEED2 = 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSS'

def get_covid_seqs_gflow():
    seed_cdr = AAY49_SEED
    seed_cdr12 = 'GFTLNSYGISVIYSDGRRTFYGDSVK'
    database = './lib/Covid/data/best_covid.csv'
    df = pd.read_csv(database,sep=',')
    data = df[['POI','Target','Assay','CDRH1','CDRH2','CDRH3','Pred_affinity']].values
    POI = []
    CDRList = []
    score = []
    length = []
    mut_count = []
    print('processing')
    for i in data:
        seq = i[3]+i[4]+i[5]
        k = distance(seed_cdr,seq)
        seq2 = i[3]+i[4]
        k2 = distance(seq2,seed_cdr12)
        if (k == 1 or k == 2 or k == 3) and k2 == 0:
            if i[0].startswith('AAYL49') and i[0] not in POI and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                POI.append(i[0])
                CDRList.append('CAK'+i[5]+'W')
                score.append(i[6])
                length.append(len(seq))
                mut_count.append(k)
    return CDRList,score

def fix_dataset():
    database = './data/best_covid.csv'
    df = pd.read_csv(database,sep=',')
    data = df[['POI','Target','Assay','CDRH1','CDRH2','CDRH3','Pred_affinity','Replicate']].values
    for i in data:
        if i[0] == 'AAYL49_234' and i[1] == 'MIT_Target':
            print(i)

def process_esm(seqlist,esm_type):
    if esm_type == 't6':
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        depth = 6
        res = torch.zeros(len(seqlist),len(seqlist[0])+2,320)
        batch_size = 32
    elif esm_type == 't30':
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        depth = 30
        res = torch.zeros(len(seqlist),len(seqlist[0])+2,640)
        batch_size = 16
    elif esm_type == 't33':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        depth = 33
        res = torch.zeros(len(seqlist),len(seqlist[0])+2,1280)
        batch_size = 8
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    model = model.to('cuda')
    num_of_batches = int(len(seqlist)/batch_size)
    for j in tqdm(range(num_of_batches)):
        data = []
        for i in range(batch_size):
            data.append(('protein{}'.format(i),seqlist[batch_size*j + i]))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        # Extract per-residue representations (on CPU)
        batch_tokens = batch_tokens.to('cuda')
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[depth], return_contacts=True)
        token_representations = results["representations"][depth]
        res[batch_size * j:batch_size * (j+1)] = token_representations.cpu()
    return res

def process_antiberty(seqs):
    antiberty = AntiBERTyRunner()
    embeddings = []
    embs = torch.zeros((len(seqs),len(seqs[0])+2,512))
    with torch.no_grad():
        for i in range(len(seqs)):
            embs[i] = antiberty.embed([seqs[i]])[0].cpu()
    return embs

class ESMTrainBestCovidDataset(Dataset):

    def __init__(self,args):
        self.tok = get_tokenizer(args)
        self.args = args
        self.seed_cdr = 'GFTLNSYGISVIYSDGRRTFYGDSVKGRAAGTFDS'
        self.seed = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
        self.seed_cdr1 = 'GFTLNSYGIS'
        self.seed_cdr2 = 'VIYSDGRRTFYGDSVK'
        self.seed_cdr3 = 'GRAAGTFDS'
        self.only_cdr1_idx = []
        self.only_cdr2_idx = []
        self.only_cdr3_idx = []
        self.other_idx = []
        self.amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.extract_data()
        #self.make_embeddings()
        self.load_data()
        self.dataset_size = len(self.esm_enc_t33)
        self.get_valid_mutations()

    def __len__(self):
        return(self.dataset_size)

    def __getitem__(self, idx):

        return self.seqs[idx], self.score[idx]

    def extract_data(self):
        script_dir = os.path.dirname(__file__)
        database = os.path.join(script_dir, 'data','best_covid.csv')
        df = pd.read_csv(database,sep=',')
        self.data = df[['POI','Target','Assay','CDRH1','CDRH2','CDRH3','Pred_affinity','Sequence']].values
        self.POI = {}
        self.seqlist = []
        self.score = []
        self.adjusted_scores = []
        self.length = []
        self.mut_count = []
        self.mut_count_cdr1 = []
        self.mut_count_cdr2 = []
        self.mut_count_cdr3 = []
        self.cdrlist = []
        self.cdrlist2 = []
        count = 0
        for i in self.data:
            seq = i[3]+i[4]+i[5]
            seq2 = i[3]+i[4][1:-1]+i[5]
            k = distance(self.seed_cdr,seq)
            if k == 1 or k == 2 or k == 3:
                if distance(seq2,self.seed) == 0:
                    count += 1
                if i[0].startswith('AAYL49') and i[0] not in self.POI and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                    self.POI[i[0]] = [i[7],[i[6]],k,distance(self.seed_cdr1,i[3]),distance(self.seed_cdr2,i[4]),distance(self.seed_cdr3,i[5]),seq2,seq,[i[6]]]
                elif i[0].startswith('AAYL49') and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                    self.POI[i[0]][1].append(i[6])
        for key in self.POI:
            self.seqlist.append(self.POI[key][0][:118])
            self.score.append(np.mean(self.POI[key][1]))
            self.length.append(len(self.POI[key][0]))
            self.mut_count.append(self.POI[key][2])
            self.mut_count_cdr1.append(self.POI[key][3])
            self.mut_count_cdr2.append(self.POI[key][4])
            self.mut_count_cdr3.append(self.POI[key][5])
            self.cdrlist.append(self.POI[key][6])
            self.cdrlist2.append(self.POI[key][7])
            self.adjusted_scores.append(np.mean(self.POI[key][8]))
        self.top_aff = np.max(self.score)
        self.top_seq = self.cdrlist[np.argmax(self.score)]
        script_dir = os.path.dirname(__file__)
        self.mut_idx_1 = np.load(os.path.join(script_dir, 'data','AAYL49_mut1_idx.npy'))
        self.mut_idx_2 = np.load(os.path.join(script_dir, 'data','AAYL49_mut2_idx.npy'))
        self.mut_idx_3 = np.load(os.path.join(script_dir, 'data','AAYL49_mut3_idx.npy'))
        self.sol_score = np.load(os.path.join(script_dir,'..','dataset','full_covid_sol_score.npy'))
        self.hum_score = np.load(os.path.join(script_dir,'..','dataset','full_covid_ppost_score.npy'))

    def extract_seed_score(self):
        script_dir = os.path.dirname(__file__)
        database = os.path.join(script_dir,'data','best_covid.csv')
        df = pd.read_csv(database,sep=',')
        self.data = df[['POI','Target','Assay','CDRH1','CDRH2','CDRH3','Pred_affinity','Sequence']].values
        self.POI = {}
        self.seqlist = []
        line_count = 1
        s = []
        for i in self.data:
            seq = i[3]+i[4]+i[5]
            seq2 = i[3]+i[4][1:-1]+i[5]
            k = distance(self.seed_cdr,seq)
            line_count += 1
            if k == 0 and i[0].startswith('AAYL49') and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                s.append(i[6])
        return np.mean(s)

    def mount_fuji(self):
        fuji_dist = np.zeros((len(self.score),len(self.score)))
        fuji_score = np.zeros((len(self.score),len(self.score)))
        for i in range(len(self.score)):
            for j in range(len(self.score)):
                fuji_dist[i] = distance(self.cdrlist[i],self.cdrlist[j])
                fuji_score[i] = np.abs(self.score[i] - self.score[j])
        script_dir = os.path.dirname(__file__)
        np.save(os.path.join(script_dir,'data','fuji_dist_AAYL49.npy'),fuji_dist)
        np.save(os.path.join(script_dir,'data','fuji_score_AAYL49.npy'),fuji_score)

    def get_sol_score(self):
        train_sol = self.sol_score[self.train_idx]
        valid_sol = self.sol_score[self.valid_idx]
        return train_sol,valid_sol

    def make_diff(self):
        self.pwm_matrix = np.ones((33,20)) * 8
        self.good_muts = {}
        mut_1_seqs_idx = [i for i in range(len(self.mut_count)) if self.mut_count[i] == 1]
        mut_1_seqs = [self.cdrlist[i] for i in mut_1_seqs_idx]
        mut_1_score = self.score[mut_1_seqs_idx]
        mean_mut_1_score = np.mean(mut_1_score) - 1.5
        count = 0
        for i in range(len(mut_1_seqs)):
            seq = mut_1_seqs[i]
            for pos in range(len(seq)):
                if seq[pos] != self.seed[pos]:
                    count += 1
                    aa = self.amino_acid_dic[seq[pos]]
                    self.pwm_matrix[pos][aa] = mut_1_score[i] - mean_mut_1_score
                    if pos in self.good_muts:
                        self.good_muts[pos].append(aa)
                    else:
                        self.good_muts[pos] = [aa]

        deltas = []
        good_seqs_scores = []
        first_order_terms = []
        delta_mut_counts = []
        for i in range(len(self.cdrlist) - 1):
            invalid = False
            if self.mut_count[i] == 2 or self.mut_count[i] == 3:
                seq = self.cdrlist[i]
                score = self.score[i] - mean_mut_1_score
                first_order_term = 0
                for pos in range(len(seq)):
                    if seq[pos] != self.seed[pos]:
                        aa = self.amino_acid_dic[seq[pos]]
                        if aa in self.good_muts[pos]:
                            score -= self.pwm_matrix[pos][aa]
                            first_order_term += self.pwm_matrix[pos][aa]
                        else:
                            invalid = True
                if not invalid:
                    good_seqs_scores.append(self.score[i])
                    deltas.append(score)
                    first_order_terms.append(first_order_term)
                    delta_mut_counts.append(self.mut_count[i])

        print(np.mean(good_seqs_scores))
        print(len(deltas))
        script_dir = os.path.dirname(__file__)
        os.path.join(script_dir,'data','AAYL49_deltas.npy')
        np.save(os.path.join(script_dir,'data','AAYL49_deltas.npy'),np.array(deltas))
        np.save(os.path.join(script_dir,'data','AAYL49_first_order.npy'),self.pwm_matrix)
        np.save(os.path.join(script_dir,'data','AAYL49_first_order_terms.npy'),np.array(first_order_terms))
        np.save(os.path.join(script_dir,'data','AAYL49_delta_mut_counts.npy'),np.array(delta_mut_counts))
        
    def make_embeddings(self):
        print('processing antiberty')
        self.ant_enc = process_antiberty(self.seqlist)
        self.ant_enc = torch.cat((self.ant_enc[:,26:36,:],self.ant_enc[:,50:66,:],self.ant_enc[:,99:108,:]),1)
        self.ant_enc = self.ant_enc.reshape(self.ant_enc.shape[0],-1)

        print('processing esm t33')
        self.esm_enc_t33 = process_esm(self.seqlist,'t33')
        self.esm_enc_t33 = torch.cat((self.esm_enc_t33[:,26:36,:],self.esm_enc_t33[:,50:66,:],self.esm_enc_t33[:,99:108,:]),1)
        self.esm_enc_t33 = self.esm_enc_t33.reshape(self.esm_enc_t33.shape[0],-1).cpu()

        print('processing esm t30')
        self.esm_enc_t30 = process_esm(self.seqlist,'t30')
        self.esm_enc_t30 = torch.cat((self.esm_enc_t30[:,26:36,:],self.esm_enc_t30[:,50:66,:],self.esm_enc_t30[:,99:108,:]),1)
        self.esm_enc_t30 = self.esm_enc_t30.reshape(self.esm_enc_t30.shape[0],-1).cpu()
        
        print('processing esm t6')
        self.esm_enc_t6 = process_esm(self.seqlist,'t6')
        self.esm_enc_t6 = torch.cat((self.esm_enc_t6[:,26:36,:],self.esm_enc_t6[:,50:66,:],self.esm_enc_t6[:,99:108,:]),1)
        self.esm_enc_t6 = self.esm_enc.reshape(self.esm_enc.shape[0],-1).cpu()
       
        self.seqs_i = []
        for seq in self.seqlist:
            self.seqs_i.append([self.amino_acid_dic[k] for k in seq])
        self.one_hot_vec = nn.functional.one_hot(torch.LongTensor(self.seqs_i),num_classes = 20)
        self.one_hot_vec = self.one_hot_vec.reshape((self.one_hot_vec.shape[0],-1))[:13920]

        self.score = np.array(self.score)
        self.length = np.array(self.length)
        self.save_data(self.esm_enc_t6,self.esm_enc_t30,self.esm_enc_t33,self.ant_enc,self.one_hot_vec,self.score,self.length,self.mut_count,self.mut_count_cdr1,self.mut_count_cdr2,self.mut_count_cdr3)

    def find_mutations(self,seq):
        list_of_mutations = []
        for i in range(len(seq)):
            if seq[i] != self.seed_cdr[i]:
                list_of_mutations.append((i,self.amino_acid_dic[seq[i]]))
        return list_of_mutations

    def get_valid_mutations(self):
        self.valid_mutations = np.zeros((35,20))
        self.position_weight_matrix = np.zeros((35,20))
        for i in range(len(self.mut_count)):
            if self.mut_count[i] == 1:
                mut = self.find_mutations(self.cdrlist2[i])[0]
                self.valid_mutations[mut[0]][mut[1]] = 1
                self.position_weight_matrix[mut[0]][mut[1]] = self.score[i]

        print(self.valid_mutations)
        print(np.sum(self.valid_mutations))
        print(self.position_weight_matrix)

    def double_epistasis(self):
        true_kd = []
        sum_single_order = []
        for i in range(len(self.mut_count)):
            if self.mut_count[i] == 2:
                muts = self.find_mutations(self.cdrlist2[i])
                if self.valid_mutations[muts[0][0]][muts[0][1]] == 1 and self.valid_mutations[muts[1][0]][muts[1][1]] == 1:
                    true_kd.append(self.score[i])
                    sum_single_order.append(self.position_weight_matrix[muts[0][0]][muts[0][1]] + self.position_weight_matrix[muts[1][0]][muts[1][1]])
        print(len(true_kd))
        print(len(sum_single_order))
        sum_single_order = np.array(sum_single_order)
        plt.scatter(sum_single_order,true_kd)
        plt.xlabel('single_order_terms')
        plt.ylabel('true_kd')
        plt.savefig('./figures/double_epistasis.png')
        print(spearmanr(sum_single_order,true_kd))
        print(r2_score(true_kd,sum_single_order))

    def triple_epistasis(self):
        true_kd = []
        sum_single_order = []
        for i in range(len(self.mut_count)):
            if self.mut_count[i] == 3:
                muts = self.find_mutations(self.cdrlist2[i])
                if self.valid_mutations[muts[0][0]][muts[0][1]] == 1 and self.valid_mutations[muts[1][0]][muts[1][1]] == 1 and self.valid_mutations[muts[2][0]][muts[2][1]] == 1:
                    true_kd.append(self.score[i])
                    sum_single_order.append(self.position_weight_matrix[muts[0][0]][muts[0][1]] + self.position_weight_matrix[muts[1][0]][muts[1][1]] + self.position_weight_matrix[muts[2][0]][muts[2][1]])
        print(len(true_kd))
        print(len(sum_single_order))
        sum_single_order = np.array(sum_single_order)
        plt.scatter(sum_single_order,true_kd)
        plt.xlabel('single_order_terms')
        plt.ylabel('true_kd')
        plt.savefig('./figures/triple_epistasis.png')
        print(spearmanr(sum_single_order,true_kd))
        print(r2_score(true_kd,sum_single_order))

    def make_energy_emb(self):
        s_muts = []
        unique_s_muts = []
        for s in self.cdrlist:
            list_of_muts = []
            for i in range(len(s)):
                if s[i] != self.seed[i]:
                    list_of_muts.append((str(i),s[i]))
                    unique_s_muts.append((str(i),s[i]))
            s_muts.append(list_of_muts)

        d_muts = []
        unique_d_muts = []
        for single in s_muts:
            if len(single) < 2:
                d_muts.append([])
            else:
                list_d_muts = []
                for j in range(len(single) - 1):
                    for k in range(j+1,len(single)):
                        list_d_muts.append((single[j][0] + '_',single[k][0],single[j][1],single[k][1]))
                        unique_d_muts.append((single[j][0] + '_',single[k][0],single[j][1],single[k][1]))
                d_muts.append(list_d_muts)

        unique_s_muts = list(set(unique_s_muts))
        unique_d_muts = list(set(unique_d_muts))
        unique_d_dict = {}
        unique_s_dict = {}
        for i in range(len(unique_d_muts)):
            unique_d_dict[''.join(unique_d_muts[i])] = i + 626
        for i in range(len(unique_s_muts)):
            unique_s_dict[''.join(unique_s_muts[i])] = i

        emb = torch.zeros((len(self.cdrlist),len(unique_s_muts)+len(unique_d_muts)))
        print(len(unique_s_muts))
        print(len(unique_d_muts))
        for i in range(len(s_muts)):
            for m in s_muts[i]:
                key = ''.join(m)
                emb[i][unique_s_dict[key]] = 1.0
        for i in range(len(d_muts)):
            for m in d_muts[i]:
                key = ''.join(m)
                emb[i][unique_d_dict[key]] = 1.0


    def mean_per_mut_count(self):
        mut_1_idx = [i for i in range(len(self.score)) if self.mut_count[i] == 1]
        mut_2_idx = [i for i in range(len(self.score)) if self.mut_count[i] == 2]
        mut_3_idx = [i for i in range(len(self.score)) if self.mut_count[i] == 3]
        print('mean score mut_1')
        print(np.mean(self.score[mut_1_idx]))
        print(np.var(self.score[mut_1_idx]))
        print('mean score mut_2')
        print(np.mean(self.score[mut_2_idx]))
        print(np.var(self.score[mut_2_idx]))
        print('mean score mut_3')
        print(np.mean(self.score[mut_3_idx]))
        print(np.var(self.score[mut_3_idx]))

    def get_mut_count(self,mut_c):
        idx = [i for i in range(len(self.mut_count)) if self.mut_count[i] == mut_c]
        seqs = self.esm_enc_t6[idx]
        score = self.score[idx]
        return seqs,score

    def k_fold_idx(self,k):
        idx = np.arange(self.dataset_size)
        idx = np.shuffle(idx)
        batch_size = int(self.dataset_size/k)

    def save_mut_count_idx(self):
        mut_1 = [i for i in range(len(self.mut_count)) if self.mut_count[i] == 1]
        mut_2 = [i for i in range(len(self.mut_count)) if self.mut_count[i] == 2]
        mut_3 = [i for i in range(len(self.mut_count)) if self.mut_count[i] == 3]
        script_dir = os.path.dirname(__file__)
        np.save(os.path.join(script_dir, 'data','AAYL49_mut1_idx.npy'),np.array(mut_1))
        np.save(os.path.join(script_dir, 'data','AAYL49_mut2_idx.npy'),np.array(mut_2))
        np.save(os.path.join(script_dir, 'data','AAYL49_mut3_idx.npy'),np.array(mut_3))

    def make_train_valid_idx(self,perc):
        random.seed(0)
        train_size = int(self.dataset_size*perc)
        train_idx = random.choice(range(self.dataset_size),size = train_size, replace = False).tolist()
        valid_idx = [i for i in range(self.dataset_size) if i not in train_idx]
        self.train_idx = np.array(train_idx)
        self.valid_idx = np.array(valid_idx)
        script_dir = os.path.dirname(__file__)
        np.save(os.path.join(script_dir, 'data','train_idx_perc:{}.npy'.format(perc)),self.train_idx)
        np.save(os.path.join(script_dir, 'data','valid_idx_perc:{}.npy'.format(perc)),self.valid_idx)
        
    def random_split(self,enc_type,perc):
        script_dir = os.path.dirname(__file__)
        os.path.join(script_dir, 'data','valid_idx_perc:{}.npy'.format(perc))
        os.path.join(script_dir, 'data','valid_idx_perc:{}.npy'.format(perc))
        self.train_idx = np.load(os.path.join(script_dir, 'data','train_idx_perc:{}.npy'.format(perc)))
        self.valid_idx = np.load(os.path.join(script_dir, 'data','valid_idx_perc:{}.npy'.format(perc)))
        if enc_type == 'esm_t6':
            train_seqs = self.esm_enc_t6[self.train_idx]
            valid_seqs = self.esm_enc_t6[self.valid_idx]
        elif enc_type == 'esm_t30':
            train_seqs = self.esm_enc_t30[self.train_idx]
            valid_seqs = self.esm_enc_t30[self.valid_idx]
        elif enc_type == 'esm_t33':
            train_seqs = self.esm_enc_t33[self.train_idx]
            valid_seqs = self.esm_enc_t33[self.valid_idx]
        elif enc_type == 'ant':
            train_seqs = self.ant_enc[self.train_idx]
            valid_seqs = self.ant_enc[self.valid_idx]
        elif enc_type == 'energy':
            train_seqs = self.energy_emb[self.train_idx]
            valid_seqs = self.energy_emb[self.valid_idx]
        else:
            train_seqs = self.one_hot_vec[self.train_idx]
            valid_seqs = self.one_hot_vec[self.valid_idx]
        train_score = self.score[self.train_idx]
        valid_score = self.score[self.valid_idx]
        self.validation_sequences = [self.cdrlist[i] for i in self.train_idx]
        self.valid_mut_count_cdr1 = self.mut_count_cdr1[self.valid_idx]
        self.valid_mut_count_cdr2 = self.mut_count_cdr2[self.valid_idx]
        self.valid_mut_count_cdr3 = self.mut_count_cdr3[self.valid_idx]
        self.valid_mut_count = self.mut_count[self.valid_idx]
        self.valid_not_mut_cdr3_idx = [i for i in range(len(self.valid_idx)) if self.valid_mut_count_cdr3[i] < 2]
        self.valid_mut_cdr3_idx = [i for i in range(len(self.valid_idx)) if self.valid_mut_count_cdr3[i] >= 2]
        self.split_by_muts()
        self.split_by_aff(train_seqs,train_score,valid_seqs,valid_score)

        return torch.Tensor(train_seqs),torch.Tensor(train_score),torch.Tensor(valid_seqs),torch.Tensor(valid_score)

    def split_mut(self,enc_type):
        self.train_idx = np.concatenate((self.mut_idx_1,self.mut_idx_2))
        self.valid_idx = self.mut_idx_3
        self.train_seqs = [self.seqlist[i] for i in self.train_idx]
        self.valid_seqs = [self.seqlist[i] for i in self.valid_idx]
        if enc_type == 'esm_t6':
            train_seqs = self.esm_enc_t6[self.train_idx]
            valid_seqs = self.esm_enc_t6[self.valid_idx]
        elif enc_type == 'esm_t30':
            train_seqs = self.esm_enc_t30[self.train_idx]
            valid_seqs = self.esm_enc_t30[self.valid_idx]
        elif enc_type == 'esm_t33':
            train_seqs = self.esm_enc_t33[self.train_idx]
            valid_seqs = self.esm_enc_t33[self.valid_idx]
        elif enc_type == 'ant':
            train_seqs = self.ant_enc[self.train_idx]
            valid_seqs = self.ant_enc[self.valid_idx]
        elif enc_type == 'energy':
            train_seqs = self.energy_emb[self.train_idx]
            valid_seqs = self.energy_emb[self.valid_idx]
        else:
            train_seqs = self.one_hot_vec[self.train_idx]
            valid_seqs = self.one_hot_vec[self.valid_idx]
        train_score = self.score[self.train_idx]
        valid_score = self.score[self.valid_idx]
        self.validation_sequences = [self.cdrlist[i] for i in self.train_idx]
        self.valid_mut_count_cdr1 = self.mut_count_cdr1[self.valid_idx]
        self.valid_mut_count_cdr2 = self.mut_count_cdr2[self.valid_idx]
        self.valid_mut_count_cdr3 = self.mut_count_cdr3[self.valid_idx]
        self.valid_not_mut_cdr3_idx = [i for i in range(len(self.valid_idx)) if self.valid_mut_count_cdr3[i] < 2]
        self.valid_mut_cdr3_idx = [i for i in range(len(self.valid_idx)) if self.valid_mut_count_cdr3[i] >= 2]
        self.split_by_muts()
        self.split_by_aff(train_seqs,train_score,valid_seqs,valid_score)

        return torch.Tensor(train_seqs),torch.Tensor(train_score),torch.Tensor(valid_seqs),torch.Tensor(valid_score)

    def split_by_muts(self):
        for i in range(len(self.valid_mut_count_cdr1)):
            if self.valid_mut_count_cdr1[i] > 0 and self.valid_mut_count_cdr2[i] == 0 and self.valid_mut_count_cdr3[i] == 0:
                self.only_cdr1_idx.append(i)
            elif self.valid_mut_count_cdr1[i] == 0 and self.valid_mut_count_cdr2[i] > 0 and self.valid_mut_count_cdr3[i] == 0:
                self.only_cdr2_idx.append(i)
            elif self.valid_mut_count_cdr1[i] == 0 and self.valid_mut_count_cdr2[i] == 0 and self.valid_mut_count_cdr3[i] > 0:
                self.only_cdr3_idx.append(i)
            else:
                self.other_idx.append(i)

    def split_by_aff(self,train_emb,train_score,valid_emb,valid_score):
        self.low_kd_train_idx = [i for i in range(len(train_score)) if train_score[i] < 2]
        self.low_kd_train_emb = train_emb[self.low_kd_train_idx]
        self.low_kd_train_score = train_score[self.low_kd_train_idx]
        self.low_kd_valid_idx = [i for i in range(len(valid_score)) if valid_score[i] < 2] 
        self.low_kd_valid_emb = valid_emb[self.low_kd_valid_idx]
        self.low_kd_valid_score = valid_score[self.low_kd_valid_idx]
 
    def save_data(self,esm_enc_t6,esm_enc_t30,esm_enc_t33,ant_enc,one_hot_vec,score,length,mut_count,mut_count_cdr1,mut_count_cdr2,mut_count_cdr3):
        script_dir = os.path.dirname(__file__)
        torch.save(esm_enc_t6, os.path.join(script_dir, 'embeddings','covid_seqs_esm_t6.pt'))
        torch.save(esm_enc_t30,os.path.join(script_dir, 'embeddings','covid_seqs_esm_t30.pt'))
        torch.save(esm_enc_t33,os.path.join(script_dir, 'embeddings','covid_seqs_esm_t33.pt'))
        torch.save(one_hot_vec,os.path.join(script_dir, 'embeddings','covid_one_hot_vec.pt'))
        torch.save(ant_enc,os.path.join(script_dir, 'embeddings','covid_seqs_ant_enc.pt'))
        np.save(os.path.join(script_dir, 'data','covid_scores.npy'),score)
        np.save(os.path.join(script_dir, 'data','covid_length.npy'),length)
        np.save(os.path.join(script_dir, 'data','covid_mut.npy'),mut_count)
        np.save(os.path.join(script_dir, 'data','covid_mut_cdr1.npy'),mut_count_cdr1)
        np.save(os.path.join(script_dir, 'data','covid_mut_cdr2.npy'),mut_count_cdr2)
        np.save(os.path.join(script_dir, 'data','covid_mut_cdr3.npy'),mut_count_cdr3)
    
    def load_data(self):
        script_dir = os.path.dirname(__file__)
        self.esm_enc_t6 = torch.load(os.path.join(script_dir, 'embeddings','covid_seqs_esm_t6.pt')).to(torch.float32)
        self.esm_enc_t30 = torch.load(os.path.join(script_dir, 'embeddings','covid_seqs_esm_t30.pt')).to(torch.float32)
        self.esm_enc_t33 = torch.load(os.path.join(script_dir, 'embeddings','covid_seqs_esm_t33.pt')).to(torch.float32)
        self.ant_enc = torch.load(os.path.join(script_dir, 'embeddings','covid_seqs_ant_enc.pt')).to(torch.float32)[:-1]
        self.one_hot_vec = torch.load(os.path.join(script_dir, 'embeddings','covid_one_hot_vec.pt')).to(torch.float32)
        self.energy_emb = torch.load(os.path.join(script_dir, 'embeddings','energy_embs.pt')).to(torch.float32)[:-1]
        self.score = np.load(os.path.join(script_dir, 'data','covid_scores.npy'))
        self.score = self.score[:-1]
        self.length = np.load(os.path.join(script_dir, 'data','covid_length.npy'))
        self.length = self.length[:-1]
        self.mut_count = np.load(os.path.join(script_dir, 'data','covid_mut.npy'))
        self.mut_count = self.mut_count[:-1]
        self.mut_count_cdr1 = np.load(os.path.join(script_dir, 'data','covid_mut_cdr1.npy'))
        self.mut_count_cdr1 = self.mut_count_cdr1[:-1]
        self.mut_count_cdr2 = np.load(os.path.join(script_dir, 'data','covid_mut_cdr2.npy'))
        self.mut_count_cdr2 = self.mut_count_cdr2[:-1]
        self.mut_count_cdr3 = np.load(os.path.join(script_dir, 'data','covid_mut_cdr3.npy'))
        self.mut_count_cdr3 = self.mut_count_cdr3[:-1]

    def test_neigh_hypothesis(self):
        mut_1 = []
        mut_2 = []
        mut_3 = []
        mut_4 = []
        for i in tqdm(range(len(self.score))):
            if self.score[i] < 1:
                for j in range(len(self.score)):
                    k = distance(self.seqlist[i],self.seqlist[j])
                    if k == 1:
                        mut_1.append(np.abs(self.score[i] - self.score[j]))
                    if k == 2:
                        mut_2.append(np.abs(self.score[i] - self.score[j]))
                    if k == 3:
                        mut_3.append(np.abs(self.score[i] - self.score[j]))
                    if k == 4:
                        mut_4.append(np.abs(self.score[i] - self.score[j]))

        counts, bins = np.histogram(mut_1)
        plt.stairs(counts,bins)
        plt.xlabel('diff in kd')
        plt.ylabel('counts')
        plt.title('1 mutation')
        plt.savefig('./figures/diffinkd1mutAAYL49.png')
        plt.clf()

        counts, bins = np.histogram(mut_2)
        plt.stairs(counts,bins)
        plt.xlabel('diff in kd')
        plt.ylabel('counts')
        plt.title('2 mutations')
        plt.savefig('./figures/diffinkd2mutAAYL49.png')
        plt.clf()

        counts, bins = np.histogram(mut_3)
        plt.stairs(counts,bins)
        plt.xlabel('diff in kd')
        plt.ylabel('counts')
        plt.title('3 mutations')
        plt.savefig('./figures/diffinkd3mutAAYL49.png')
        plt.clf()

        counts, bins = np.histogram(mut_4)
        plt.stairs(counts,bins)
        plt.xlabel('diff in kd')
        plt.ylabel('counts')
        plt.title('4 mutations')
        plt.savefig('./figures/diffinkd4mutAAYL49.png')
        plt.clf()

class ESMTrainBestCovidDataset2(Dataset):

    def __init__(self,args):
        self.tok = get_tokenizer(args)
        self.args = args
        self.seed_cdr = 'GFTFDDYAMHGISWNSGSIGYADSVKVGRGGGYFDY'
        self.seed = 'GFTFDDYAMHGISWNSGSIGYADSVKVGRGGGYFDY'
        self.seed_cdr1 = 'GFTFDDYAMH'
        self.seed_cdr2 = 'GISWNSGSIGYADSVK'
        self.seed_cdr3 = 'VGRGGGYFDY'
        self.only_cdr1_idx = []
        self.only_cdr2_idx = []
        self.only_cdr3_idx = []
        self.other_idx = []
        self.amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.extract_data()
        self.dataset_size = len(self.esm_enc_t33)

    def __len__(self):
        return(self.dataset_size)

    def __getitem__(self, idx):

        return self.seqs[idx], self.score[idx]

    def extract_data(self):
        script_dir = os.path.dirname(__file__)
        database = os.path.join(script_dir, 'data','best_covid.csv')
        df = pd.read_csv(database,sep=',')
        self.data = df[['POI','Target','Assay','CDRH1','CDRH2','CDRH3','Pred_affinity','Sequence']].values
        self.POI = {}
        self.seqlist = []
        self.score = []
        self.length = []
        self.mut_count = []
        self.mut_count_cdr1 = []
        self.mut_count_cdr2 = []
        self.mut_count_cdr3 = []
        self.cdrlist = []
        for i in self.data:
            seq = i[3]+i[4]+i[5]
            seq2 = i[3]+i[4][1:-1]+i[5]
            k = distance(self.seed_cdr,seq)
            if k == 1 or k == 2 or k == 3:
                if i[0].startswith('AAYL51') and i[0] not in self.POI and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                    self.POI[i[0]] = [i[7],[i[6]],k,distance(self.seed_cdr1,i[3]),distance(self.seed_cdr2,i[4]),distance(self.seed_cdr3,i[5]),seq]
                elif i[0].startswith('AAYL51') and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                    self.POI[i[0]][1].append(i[6])
        for key in self.POI:
            self.seqlist.append(self.POI[key][0][130:])
            self.score.append(np.mean(self.POI[key][1]))
            self.length.append(len(self.POI[key][0]))
            self.mut_count.append(self.POI[key][2])
            self.mut_count_cdr1.append(self.POI[key][3])
            self.mut_count_cdr2.append(self.POI[key][4])
            self.mut_count_cdr3.append(self.POI[key][5])
            self.cdrlist.append(self.POI[key][6])
        self.top_aff = np.max(self.score)
        self.top_seq = self.cdrlist[np.argmax(self.score)]

    def mount_fuji(self):
        fuji_dist = np.zeros(len(self.cdrlist),len(self.cdrlist))
        fuji_score = np.zeros(len(self.cdrlist),len(self.cdrlist))
        for i in range(len(self.score)):
            fuji_dist[i] = distance(self.cdrlist[i],self.cdrlist[j])
            fuji_score[i] = np.abs(np.self.score[i] - np.self.score[j])
        script_dir = os.path.dirname(__file__)
        np.save(os.path.join(script_dir, 'data','fuji_dist_AAYL51.npy'),fuji_dist)
        np.save(os.path.join(script_dir, 'data','fuji_score_AAYL51.npy'),fuji_score)

    def make_embeddings(self):
        print('processing antiberty')
        self.ant_enc = process_antiberty(self.seqlist)
        self.ant_enc = torch.cat((self.ant_enc[:,26:36,:],self.ant_enc[:,50:66,:],self.ant_enc[:,99:109,:]),1)
        self.ant_enc = self.ant_enc.reshape(self.ant_enc.shape[0],-1)

        print('processing esm t33')
        self.esm_enc_t33 = process_esm(self.seqlist,'t33')
        self.esm_enc_t33 = torch.cat((self.esm_enc_t33[:,26:36,:],self.esm_enc_t33[:,50:66,:],self.esm_enc_t33[:,99:109,:]),1)
        self.esm_enc_t33 = self.esm_enc_t33.reshape(self.esm_enc_t33.shape[0],-1).cpu()

        print('processing esm t30')
        self.esm_enc_t30 = process_esm(self.seqlist,'t30')
        self.esm_enc_t30 = torch.cat((self.esm_enc_t30[:,26:36,:],self.esm_enc_t30[:,50:66,:],self.esm_enc_t30[:,99:109,:]),1)
        self.esm_enc_t30 = self.esm_enc_t30.reshape(self.esm_enc_t30.shape[0],-1).cpu()
        
        print('processing esm t6')
        self.esm_enc_t6 = process_esm(self.seqlist,'t6')
        self.esm_enc_t6 = torch.cat((self.esm_enc_t6[:,26:36,:],self.esm_enc_t6[:,50:66,:],self.esm_enc_t6[:,99:109,:]),1)
        self.esm_enc_t6 = self.esm_enc_t6.reshape(self.esm_enc_t6.shape[0],-1).cpu()
       
        self.seqs_i = []
        for seq in self.seqlist:
            self.seqs_i.append([self.amino_acid_dic[k] for k in seq])
        self.one_hot_vec = nn.functional.one_hot(torch.LongTensor(self.seqs_i),num_classes = 20)
        self.one_hot_vec = self.one_hot_vec.reshape((self.one_hot_vec.shape[0],-1))[:13920]

        self.score = np.array(self.score)
        self.length = np.array(self.length)
        self.save_data(self.esm_enc_t6,self.esm_enc_t30,self.esm_enc_t33,self.ant_enc,self.one_hot_vec,self.score,self.length,self.mut_count,self.mut_count_cdr1,self.mut_count_cdr2,self.mut_count_cdr3)
        
        '''
        with open('./data/cdrlist.txt','w') as f:
            for s in self.cdrlist:
                f.write(s+'\n')
        '''

    
    def extract_seed_score(self):
        script_dir = os.path.dirname(__file__)
        database = os.path.join(script_dir, 'data','best_covid.csv')
        df = pd.read_csv(database,sep=',')
        self.data = df[['POI','Target','Assay','CDRH1','CDRH2','CDRH3','Pred_affinity','Sequence']].values
        self.POI = {}
        self.seqlist = []
        line_count = 1
        s = []
        for i in self.data:
            seq = i[3]+i[4]+i[5]
            seq2 = i[3]+i[4][1:-1]+i[5]
            k = distance(self.seed_cdr,seq)
            line_count += 1
            if k == 0 and i[0].startswith('AAYL51') and i[1] == 'MIT_Target' and not math.isnan(i[6]):
                print(i[6])
                print(line_count)
                s.append(i[6])
        return np.mean(s)

    def make_diff(self):
        self.pwm_matrix = np.ones((36,20)) * 8
        self.good_muts = {}
        mut_1_seqs_idx = [i for i in range(len(self.mut_count)) if self.mut_count[i] == 1]
        print(len(mut_1_seqs_idx))
        print(len(self.cdrlist[0]))
        mut_1_seqs = [self.cdrlist[i] for i in mut_1_seqs_idx]
        mut_1_score = self.score[mut_1_seqs_idx]
        mean_mut_1_score = np.mean(mut_1_score)
        count = 0
        for i in range(len(mut_1_seqs)):
            seq = mut_1_seqs[i]
            for pos in range(len(seq)):
                if seq[pos] != self.seed[pos]:
                    count += 1
                    aa = self.amino_acid_dic[seq[pos]]
                    self.pwm_matrix[pos][aa] = mut_1_score[i] - mean_mut_1_score
                    if pos in self.good_muts:
                        self.good_muts[pos].append(aa)
                    else:
                        self.good_muts[pos] = [aa]
        print(count)

        deltas = []
        for i in range(len(self.cdrlist) - 7):
            invalid = False
            if i not in mut_1_seqs_idx:
                seq = self.cdrlist[i]
                for pos in range(len(seq)):
                    score = self.score[i] - mean_mut_1_score
                    if seq[pos] != self.seed[pos]:
                        aa = self.amino_acid_dic[seq[pos]]
                        if aa in self.good_muts[pos]:
                            score -= self.pwm_matrix[pos][aa]
                        else:
                            invalid = True
                if not invalid:
                    deltas.append(score)
        script_dir = os.path.dirname(__file__)
        np.save(os.path.join(script_dir, 'data','AAYL51_deltas.npy'),np.array(deltas))
        np.save(os.path.join(script_dir, 'data','AAYL51_first_order.npy'),self.pwm_matrix)

    def get_mut_count(self,mut_c):
        idx = [i for i in range(len(self.mut_count)) if self.mut_count[i] == mut_c]
        seqs = self.esm_enc_t6[idx]
        score = self.score[idx]
        return seqs,score
        
    def random_split(self,enc_type,perc):
        random.seed(0)
        train_size = int(self.dataset_size*perc)
        train_idx = random.choice(range(self.dataset_size),size = train_size, replace = False).tolist()
        valid_idx = [i for i in range(self.dataset_size) if i not in train_idx]
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        if enc_type == 'esm_t6':
            train_seqs = self.esm_enc_t6[train_idx]
            valid_seqs = self.esm_enc_t6[valid_idx]
        elif enc_type == 'esm_t30':
            train_seqs = self.esm_enc_t30[train_idx]
            valid_seqs = self.esm_enc_t30[valid_idx]
        elif enc_type == 'esm_t33':
            train_seqs = self.esm_enc_t33[train_idx]
            valid_seqs = self.esm_enc_t33[valid_idx]
        elif enc_type == 'ant':
            train_seqs = self.ant_enc[train_idx]
            valid_seqs = self.ant_enc[valid_idx]
        else:
            train_seqs = self.one_hot_vec[train_idx]
            valid_seqs = self.one_hot_vec[valid_idx]
        train_score = self.score[train_idx]
        valid_score = self.score[valid_idx]
        self.validation_sequences = [self.cdrlist[i] for i in self.train_idx]
        self.valid_mut_count_cdr1 = self.mut_count_cdr1[valid_idx]
        self.valid_mut_count_cdr2 = self.mut_count_cdr2[valid_idx]
        self.valid_mut_count_cdr3 = self.mut_count_cdr3[valid_idx]
        self.split_by_muts()
        self.split_by_aff(train_seqs,train_score,valid_seqs,valid_score)

        return torch.Tensor(train_seqs),torch.Tensor(train_score),torch.Tensor(valid_seqs),torch.Tensor(valid_score)

    def split_by_muts(self):
        for i in range(len(self.valid_mut_count_cdr1)):
            if self.valid_mut_count_cdr1[i] > 0 and self.valid_mut_count_cdr2[i] == 0 and self.valid_mut_count_cdr3[i] == 0:
                self.only_cdr1_idx.append(i)
            elif self.valid_mut_count_cdr1[i] == 0 and self.valid_mut_count_cdr2[i] > 0 and self.valid_mut_count_cdr3[i] == 0:
                self.only_cdr2_idx.append(i)
            elif self.valid_mut_count_cdr1[i] == 0 and self.valid_mut_count_cdr2[i] == 0 and self.valid_mut_count_cdr3[i] > 0:
                self.only_cdr3_idx.append(i)
            else:
                self.other_idx.append(i)

    def split_by_aff(self,train_emb,train_score,valid_emb,valid_score):
        self.low_kd_train_idx = [i for i in range(len(train_score)) if train_score[i] < 2]
        self.low_kd_train_emb = train_emb[self.low_kd_train_idx]
        self.low_kd_train_score = train_score[self.low_kd_train_idx]
        self.low_kd_valid_idx = [i for i in range(len(valid_score)) if valid_score[i] < 2] 
        self.low_kd_valid_emb = valid_emb[self.low_kd_valid_idx]
        self.low_kd_valid_score = valid_score[self.low_kd_valid_idx]
 
    def save_data(self,esm_enc_t6,esm_enc_t30,esm_enc_t33,ant_enc,one_hot_vec,score,length,mut_count,mut_count_cdr1,mut_count_cdr2,mut_count_cdr3):
        script_dir = os.path.dirname(__file__)
        torch.save(esm_enc_t6, os.path.join(script_dir, 'embeddings','AAYL51_seqs_esm_t6.pt'))
        torch.save(esm_enc_t30,os.path.join(script_dir, 'embeddings','AAYL51_seqs_esm_t30.pt'))
        torch.save(esm_enc_t33,os.path.join(script_dir, 'embeddings','AAYL51_seqs_esm_t33.pt'))
        torch.save(one_hot_vec,os.path.join(script_dir, 'embeddings','AAYL51_one_hot_vec.pt'))
        torch.save(ant_enc,os.path.join(script_dir, 'embeddings','AAYL51_seqs_ant_enc.pt'))
        np.save(os.path.join(script_dir, 'data','AAYL51_scores.npy'),score)
        np.save(os.path.join(script_dir, 'data','AAYL51_length.npy'),length)
        np.save(os.path.join(script_dir, 'data','AAYL51_mut.npy'),mut_count)
        np.save(os.path.join(script_dir, 'data','AAYL51_mut_cdr1.npy'),mut_count_cdr1)
        np.save(os.path.join(script_dir, 'data','AAYL51_mut_cdr2.npy'),mut_count_cdr2)
        np.save(os.path.join(script_dir, 'data','AAYL51_mut_cdr3.npy'),mut_count_cdr3)
    
    def load_data(self):
        script_dir = os.path.dirname(__file__)
        self.esm_enc_t6 = torch.load(os.path.join(script_dir, 'embeddings','AAYL51_seqs_esm_t6.pt')).to(torch.float32)
        self.esm_enc_t30 = torch.load(os.path.join(script_dir, 'embeddings','AAYL51_seqs_esm_t30.pt')).to(torch.float32)
        self.esm_enc_t33 = torch.load(os.path.join(script_dir, 'embeddings','AAYL51_seqs_esm_t33.pt')).to(torch.float32)
        self.ant_enc = torch.load(os.path.join(script_dir, 'embeddings','AAYL51_seqs_ant_enc.pt')).to(torch.float32)[:-7]
        self.one_hot_vec = torch.load(os.path.join(script_dir, 'embeddings','AAYL51_one_hot_vec.pt')).to(torch.float32)        
        self.score = np.load(os.path.join(script_dir, 'data','AAYL51_scores.npy'))
        self.score = self.score[:-7]
        self.length = np.load(os.path.join(script_dir, 'data','AAYL51_length.npy'))
        self.length = self.length[:-7]
        self.mut_count = np.load(os.path.join(script_dir, 'data','AAYL51_mut.npy'))
        self.mut_count = self.mut_count[:-7]
        self.mut_count_cdr1 = np.load(os.path.join(script_dir, 'data','AAYL51_mut_cdr1.npy'))
        self.mut_count_cdr1 = self.mut_count_cdr1[:-7]
        self.mut_count_cdr2 = np.load(os.path.join(script_dir, 'data','AAYL51_mut_cdr2.npy'))
        self.mut_count_cdr2 = self.mut_count_cdr2[:-7]
        self.mut_count_cdr3 = np.load(os.path.join(script_dir, 'data','AAYL51_mut_cdr3.npy'))
        self.mut_count_cdr3 = self.mut_count_cdr3[:-7]


def top_k_perc(scores,p):
    cutoffs = [0.5 * i for i in range(-2,14)]
    for c in cutoffs:
        count = np.sum(scores < c)
        if count/len(scores) > p:
            return c,(count/len(scores))

def plot_histo_scores():
    task = 'AAYL49'
    b = args()
    if task == 'AAYL49':
        c = ESMTrainBestCovidDataset(b)
        init = np.round(c.extract_seed_score(),decimals = 2)
        c1 = 1.5
        c2 = 2.0
    else:
        c = ESMTrainBestCovidDataset2(b)
        init = np.round(c.extract_seed_score(),decimals = 2)
        c1 = 1.5
        c2 = 2.5
    counts, bins = np.histogram(c.score,bins = [0.5 * i for i in range(-2,14)])
    plt.stairs(counts,bins)
    y = np.linspace(0,max(counts))
    #plt.plot([init for i in range(len(y))],y,'b--',label = 'WT KD')
    #plt.plot([c1 for i in range(len(y))],y,'g--',label = 'c1 cutoff')
    #plt.plot([c2 for i in range(len(y))],y,'r--',label = 'c2 cutoff')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="upper left")
    plt.ylabel('counts')
    plt.xlabel('kd (log 10 nM)')
    plt.title('distribution of kd for {} sequences'.format(task))
    plt.tight_layout()
    plt.savefig('./figures/histo_score_{}.png'.format(task))

def plot_histo_scores_mut():
    task = 'AAYL49'
    b = args()
    fig,axs = plt.subplots(2,2,figsize = (16,12))

    if task == 'AAYL49':
        dataset = ESMTrainBestCovidDataset(b)
        init = np.round(dataset.extract_seed_score(),decimals = 2)
    else:
        dataset = ESMTrainBestCovidDataset2(b)
        init = np.round(dataset.extract_seed_score(),decimals = 2)


    counts, bins = np.histogram(dataset.score,bins = [0.5 * i for i in range(-2,14)],density = True)
    print('all muts')
    mean = np.mean(dataset.score)
    variance = np.var(dataset.score)
    textstr = 'Mean: {} \nVariance:{}'.format(np.round(mean,2),np.round(variance,2))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0][0].stairs(counts,bins)
    axs[0][0].text(-1, 0.35, textstr, fontsize=14,
        verticalalignment='top',bbox = props)
    axs[0][0].set_xlabel('KD')
    axs[0][0].set_label('Counts')
    axs[0][0].set_title('All mutants')

    counts, bins = np.histogram(dataset.score[dataset.mut_idx_1],bins = [0.5 * i for i in range(-2,14)],density = True)
    print('1 mut')
    mean = np.mean(dataset.score[dataset.mut_idx_1])
    variance = np.var(dataset.score[dataset.mut_idx_1])
    textstr = 'Mean: {} \nVariance:{}'.format(np.round(mean,2),np.round(variance,2))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0][1].stairs(counts,bins)
    axs[0][1].text(-1, 0.35, textstr, fontsize=14,
        verticalalignment='top',bbox = props)
    axs[0][1].set_xlabel('KD')
    axs[0][1].set_label('Counts')
    axs[0][1].set_title('1 mutation')
    counts, bins = np.histogram(dataset.score[dataset.mut_idx_2],bins = [0.5 * i for i in range(-2,14)],density = True)
    print('2 mut')
    mean = np.mean(dataset.score[dataset.mut_idx_2])
    variance = np.var(dataset.score[dataset.mut_idx_2])
    textstr = 'Mean: {} \nVariance:{}'.format(np.round(mean,2),np.round(variance,2))
    axs[1][0].stairs(counts,bins)
    axs[1][0].text(-1, 0.28, textstr, fontsize=14,
        verticalalignment='top',bbox = props)
    axs[1][0].set_xlabel('KD')
    axs[1][0].set_label('Counts')
    axs[1][0].set_title('2 mutations')
    counts, bins = np.histogram(dataset.score[dataset.mut_idx_3],bins = [0.5 * i for i in range(-2,14)],density = True)
    print('3 mut')
    mean = np.mean(dataset.score[dataset.mut_idx_3])
    variance = np.var(dataset.score[dataset.mut_idx_3])
    textstr = 'Mean: {} \nVariance:{}'.format(np.round(mean,2),np.round(variance,2))
    axs[1][1].stairs(counts,bins)
    axs[1][1].text(-1, 0.35, textstr, fontsize=14,
        verticalalignment='top',bbox = props)
    axs[1][1].set_xlabel('KD')
    axs[1][1].set_label('Counts')
    axs[1][1].set_title('3 mutations')
    plt.tight_layout()
    plt.savefig('./figures/histo_score_mut_split_{}.png'.format(task))
    
def plot_histo_adjusted_scores_mut():
    task = 'AAYL49'
    b = args()
    if task == 'AAYL49':
        dataset = ESMTrainBestCovidDataset(b)

    counts, bins = np.histogram(dataset.adjusted_scores[dataset.mut_idx_1],bins = [0.5 * i for i in range(-2,14)],density = True)
    plt.stairs(counts,bins,label = 'mut_1')
    counts, bins = np.histogram(dataset.adjusted_scores[dataset.mut_idx_2],bins = [0.5 * i for i in range(-2,14)],density = True)
    plt.stairs(counts,bins,label = 'mut_2')
    counts, bins = np.histogram(dataset.adjusted_scores[dataset.mut_idx_3],bins = [0.5 * i for i in range(-2,14)],density = True)
    plt.stairs(counts,bins,label = 'mut_3')
    y = np.linspace(0,max(counts))
    plt.plot([init for i in range(len(y))],y,'b--',label = 'WT KD')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="upper left")
    plt.ylabel('counts')
    plt.xlabel('adjusted kd')
    plt.title('distribution of kd for {} sequences'.format(task))
    plt.tight_layout()
    plt.savefig('./figures/histo_adjusted_score_mut_split_{}.png'.format(task))

def plot_histo_scores_mut_2():
    task = 'AAYL49'
    b = args()
    if task == 'AAYL49':
        dataset = ESMTrainBestCovidDataset(b)
        init = np.round(dataset.extract_seed_score(),decimals = 2)
    else:
        dataset = ESMTrainBestCovidDataset2(b)
        init = np.round(dataset.extract_seed_score(),decimals = 2)

    counts, bins = np.histogram(dataset.score[np.concatenate((dataset.mut_idx_1,dataset.mut_idx_2))],bins = [0.5 * i for i in range(-2,14)],density = True)
    plt.stairs(counts,bins,label = 'mut_1 + mut_2')
    counts, bins = np.histogram(dataset.score[dataset.mut_idx_3],bins = [0.5 * i for i in range(-2,14)],density = True)
    plt.stairs(counts,bins,label = 'mut_3')
    y = np.linspace(0,max(counts))
    plt.plot([init for i in range(len(y))],y,'b--',label = 'WT KD')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="upper left")
    plt.ylabel('counts')
    plt.xlabel('kd')
    plt.title('distribution of kd for {} sequences'.format(task))
    plt.tight_layout()
    plt.savefig('./figures/histo_score_mut_split_{}_2.png'.format(task))

def diff():
    task = 'AAYL49'
    b = args()
    if task == 'AAYL49':
        dataset = ESMTrainBestCovidDataset(b)
    else:
        dataset = ESMTrainBestCovidDataset2(b)
    dataset.make_diff()

if __name__ == '__main__':
    b = args()
    c = ESMTrainBestCovidDataset(b)
   
    
