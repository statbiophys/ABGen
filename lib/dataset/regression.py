import numpy as np
import os.path as osp
from lib.dataset.base import Dataset
#from base import Dataset
import math
import queue
import random
import os


class RandomDataset(Dataset):
    def __init__(self,split,nfold,args,oracle):
        super().__init__(args,oracle)
        x,ppost_score,sasa_score,aff_score = self.readseqs()
        #x,ppost_score,sasa_score,aff_score = self.remove_bad_aff(x,ppost_score,sasa_score,aff_score)
        print(len(x))
        #y = [math.exp((i+75)) for i in y]
        self.current = len(ppost_score)
        self.max_size = 11000
        split = int(0.8*self.current)
        self.train = x[:split]
        self.valid = x[split:]
        self.weights = np.array([0.33,0.33,0.33,1.0])
        self.current = len(self.train)
        #self.pgen_train_scores = pgen_score[:split]
        #self.pgen_valid_scores = pgen_score[split:]
        self.ppost_train_scores = ppost_score[:split]
        self.ppost_valid_scores = ppost_score[split:]
        self.sasa_train_scores = sasa_score[:split]
        self.sasa_valid_scores = sasa_score[split:]
        self.aff_train_scores = aff_score[:split]
        self.aff_valid_scores = aff_score[split:]
        #self.immuno_train_scores = immuno_score[:split]
        #self.immuno_valid_scores = immuno_score[split:]
        self.train_scores = self.weights[0]*self.ppost_train_scores+self.weights[1]*self.sasa_train_scores+self.weights[2]*self.aff_train_scores
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.weights[0]*self.ppost_valid_scores+self.weights[1]*self.sasa_valid_scores+self.weights[2]*self.aff_valid_scores
        self.nfold = 5
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def add_seq(self,seq,r):
        if seq[-1] == '%':
            seq = seq[:-1]
        if len(seq) == 13:
            if self.current < self.max_size:
                self.train.append(seq)
            else:
                self.train[self.current % self.max_size] = seq
            self.train_scores[self.current % self.max_size] = r
            self.current += 1

    def readseqs(self):
        x = []
        #pgen_score = []
        ppost_score = []
        sasa_score = []
        aff_score = []
        #immuno_score = []
        count = 0
        sasa_temp = 2.0
        pgen_temp = 1.0
        aff_temp = 5.0
        ppost_temp = 1.0
        immuno_temp = 20.0

        '''
        with open('./lib/dataset/pgen_len13.csv','r') as f:
            for l in f:
                a = l.split(',')
                if len(a[0]) == 13:
                    x.append(a[0])
                    pgen_score.append((float(a[1])+45)*pgen_temp)
                    count += 1
                    if count > 800000:
                        break
        '''
        with open('./lib/dataset/ppost_iglm.csv','r') as f:
            for l in f:
                a = l.split(',')
                if len(a[0]) == 13:
                    x.append(a[0])
                    ppost_score.append((float(a[1]))*ppost_temp)
                    count += 1
                    if count > 800000:
                        break
        '''
        with open('./lib/dataset/immuno_len13.csv','r') as f:
            for l in f:
                a = l.split(',')
                if len(a[0]) == 13:
                    immuno_score.append((float(a[1]))*immuno_temp)
                    count += 1
                    if count > 800000:
                        break
        '''
        with open('./lib/dataset/sasa_iglm.csv','r') as f:
            for l in f:
                a = l.split(',')
                if len(a[0]) == 13:
                    sasa_score.append((float(a[1])+6.0)*sasa_temp)
                    count += 1
                    if count > 800000:
                        break

        with open('./lib/dataset/aff_iglm_3.csv','r') as f:
            for l in f:
                a = l.split(',')
                if len(a[0]) == 13:
                    aff_score.append(float(a[1])*aff_temp)
                    count += 1
                    if count > 800000:
                        break
    
        #pgen_score = np.array(pgen_score)
        ppost_score = np.array(ppost_score)
        sasa_score = np.array(sasa_score)
        aff_score = np.array(aff_score)
        #immuno_score = np.array(immuno_score)
        print('size of dataset')
        print(len(x))
        return x,ppost_score,sasa_score,aff_score

    def remove_bad_aff(self,x,ppost,sol,aff):
        goodx = []
        goodp = []
        gooda = []
        goods = []
        for i in range(len(aff)):
            if aff[i] > 0:
                goodx.append(x[i])
                goodp.append(ppost[i])
                gooda.append(aff[i])
                goods.append(sol[i])
        return goodx,np.array(goodp),np.array(goods),np.array(gooda)


    def set_weights(self,w):
        self.weights = w
        self.train_scores = self.weights[3]*(self.weights[0]/self.weights[3]*self.ppost_train_scores+self.weights[1]*self.sasa_train_scores+self.weights[2]*self.aff_train_scores)
        self.current = len(self.train_scores)
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.weights[3]*(self.weights[0]/self.weights[3]*self.ppost_valid_scores+self.weights[1]*self.sasa_valid_scores+self.weights[2]*self.aff_valid_scores)

    def sample(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def sample_valid(self,n):
        indices = np.random.randint(0, len(self.valid), n)
        return ([self.valid[i] for i in indices],
                [self.valid_scores[i] for i in indices])

    def sample_first_n(self,n):
        return(self.train[0:n],self.train_scores[0:n])

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/self.nfold):
                self.valid.append(x)
                val.append(score)
            else:
                self.train.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)

    def validation_set(self):
        return self.valid, self.valid_scores

    def _top_k(self, data, k):
        topk_scores, topk_prots = [], []
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.concatenate((topk_scores, data[1][indices]))
        topk_prots = np.concatenate((topk_prots, np.array(data[0])[indices]))
        return topk_prots.tolist(), topk_scores

    def top_k(self, k):
        data = (self.train + self.valid, np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]))
        data = (seqs, scores)
        return self._top_k(data, k)


class CovidAllDataset(Dataset):
    def __init__(self,split,nfold,args,oracle):
        #super().__init__(args,oracle)
        self.script_dir = os.path.dirname(__file__)
        self.readseqs()
        self.oracle = oracle
        self.replicate = args.seed 
        #x,ppost_score,sasa_score,aff_score = self.remove_bad_aff(x,ppost_score,sasa_score,aff_score)
        print(len(self.seqs[0]))
        #y = [math.exp((i+75)) for i in y]
        self.current = len(self.seqs)
        self.max_size = 10000
        split = int(0.2*self.current)
        self.train = self.seqs[:split]
        self.valid = self.seqs[split:]
        self.train_score = self.score[:split]
        self.valid_score = self.score[split:]
        self.weights = np.array([1.0,0.0,1.0,20.0,1.0,1.0])
        self.current = len(self.train)
        self.ppost_train_scores = self.ppost_score[:split]
        self.ppost_valid_scores = self.ppost_score[split:]
        self.sasa_train_scores = self.sol_score[:split]
        self.sasa_valid_scores = self.sol_score[split:]
        self.aff_train_scores = self.aff_score[:split]
        self.aff_valid_scores = self.aff_score[split:]
        self.var_train_scores = self.var_score[:split]
        self.var_valid_scores = self.var_score[split:]
        self.train_scores = self.ppost_train_scores+self.weights[3] * (self.weights[1]*self.sasa_train_scores + self.weights[2]*(-self.aff_train_scores + self.weights[4]*self.var_train_scores))
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.ppost_valid_scores+self.weights[3] * (self.weights[1]*self.sasa_valid_scores + self.weights[2]*(-self.aff_valid_scores + self.weights[4]*self.var_valid_scores))
        self.nfold = 5
        
    def add_seq(self,seq,r):
        if seq[-1] == '%':
            seq = seq[:-1]
        if len(seq) == 33:
            if self.current < self.max_size:
                self.train.append(seq)
            else:
                self.train[self.current % self.max_size] = seq
            self.train_scores[self.current % self.max_size] = r
            self.current += 1

    def save_dataset(self):
        file_path = os.path.join(self.script_dir, 'saved_datasets','train_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'w') as f:
            for seq in self.train:
                f.write(seq+'\n')
        file_path = os.path.join(self.script_dir, 'saved_datasets','valid_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'w') as f:
            for seq in self.valid:
                f.write(seq+'\n')
        file_path = os.path.join(self.script_dir, 'saved_datasets','train_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        np.save(file_path,self.train_scores)
        file_path = os.path.join(self.script_dir, 'saved_datasets','valid_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        np.save(file_path,self.valid_scores)

    def load_dataset(self):
        train_seqs = []
        valid_seqs = []
        file_path = os.path.join(self.script_dir, 'saved_datasets','train_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                train_seqs.append(line.split('\n')[0])
        file_path = os.path.join(self.script_dir, 'saved_datasets','valid_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                valid_seqs.append(line.split('\n')[0])
        self.train = train_seqs
        self.valid = valid_seqs
        self.current = len(self.train)
        file_path = os.path.join(self.script_dir, 'saved_datasets','train_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        self.train_scores = np.load(file_path)
        file_path = os.path.join(self.script_dir, 'saved_datasets','valid_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        self.valid_scores = np.load(file_path)


    def readseqs(self):
        self.seqs = []
        file_path = os.path.join(self.script_dir,'..','Covid','data/cdrlist.txt')
        with open(file_path,'r') as f:
            for line in f:
                self.seqs.append(line.split('\n')[0])
        self.score = np.load(os.path.join(self.script_dir,'..','Covid','data','covid_scores.npy'))
        self.ppost_score = np.load(os.path.join(self.script_dir,'full_covid_ppost_score.npy'))
        self.sol_score = np.load(os.path.join(self.script_dir,'full_covid_sol_score.npy'))
        self.aff_score = np.load(os.path.join(self.script_dir,'full_covid_aff_score.npy'))
        self.var_score = np.load(os.path.join(self.script_dir,'full_covid_var_score.npy'))
    
    def set_weights(self,w):
        self.weights = w
        self.train_scores = self.weights[0] * self.ppost_train_scores + self.weights[3] * (self.weights[1]*self.sasa_train_scores + self.weights[2]*(-self.aff_train_scores + self.weights[4]*self.var_train_scores))
        self.current = len(self.train_scores)
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.weights[0] * self.ppost_valid_scores + self.weights[3] * (self.weights[1]*self.sasa_valid_scores + self.weights[2]*(-self.aff_valid_scores + self.weights[4]*self.var_valid_scores))
        self.get_mcmc_weights(self.weights)

    def get_mcmc_weights(self,weights):
        self.mcmc_seqs = []
        file_path = os.path.join(self.script_dir,'gen_seqs','mcmc','covid','mcmc_all_exp_lim_burn_dupl_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:1.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                self.mcmc_seqs.append(line.split('\n')[0])
        self.mcmc_length = len(self.mcmc_seqs)
        file_path = os.path.join(self.script_dir,'gen_seqs','mcmc','covid','mcmc_all_exp_lim_burn_dupl_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:1.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],self.replicate))
        self.mcmc_score = np.load(file_path)
        sol_r,aff_r,var_r = self.oracle.return_indiv_scores(random.choices(self.mcmc_seqs,k = min(self.mcmc_length,500)))
        self.expected_aff = np.mean(aff_r + weights[4] * var_r)
        self.expected_sol = np.mean(sol_r)
        split = int(0.8*self.mcmc_length)
        train_indices = sorted(np.random.randint(0, len(self.mcmc_seqs), split).tolist())
        idx = 0
        valid_indices = []
        for i in range(len(self.mcmc_seqs)):
            if not i == train_indices[idx]:
                valid_indices.append(i)
            else:
                idx = idx + 1
            
        #valid_indices = [i for i in range(len(self.mcmc_seqs)) if i not in train_indices]
        self.mcmc_seqs_train = [self.mcmc_seqs[i] for i in train_indices]
        self.mcmc_seqs_valid = [self.mcmc_seqs[i] for i in valid_indices]
        self.mcmc_score_train = self.mcmc_score[train_indices]
        self.mcmc_score_valid = self.mcmc_score[valid_indices]

    def sample(self, n):
        n = min(n,len(self.train))
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def sample_valid(self,n):
        n = min(n,len(self.valid))
        indices = np.random.randint(0, len(self.valid), n)
        return ([self.valid[i] for i in indices],
                [self.valid_scores[i] for i in indices])

    def sample_mcmc(self,n):
        n = min(n,len(self.mcmc_seqs))
        indices = np.random.randint(0, len(self.mcmc_seqs), n)
        return ([self.mcmc_seqs[i] for i in indices],
                [self.mcmc_score[i] for i in indices])

    def sample_mcmc_valid(self,n):
        n = min(n,len(self.mcmc_seqs_valid))
        indices = np.random.randint(0, len(self.mcmc_seqs_valid), n)
        return ([self.mcmc_seqs_valid[i] for i in indices],
                [self.mcmc_score_valid[i] for i in indices])

    def sample_first_n(self,n):
        return(self.train[0:n],self.train_scores[0:n])

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/self.nfold):
                self.valid.append(x)
                val.append(score)
            else:
                self.train.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)

    def validation_set(self):
        return self.valid, self.valid_scores

    def _top_k(self, data, k):
        topk_scores, topk_prots = [], []
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.concatenate((topk_scores, data[1][indices]))
        topk_prots = np.concatenate((topk_prots, np.array(data[0])[indices]))
        return topk_prots.tolist(), topk_scores

    def top_k(self, k):
        data = (self.train + self.valid, np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]))
        data = (seqs, scores)
        return self._top_k(data, k)


class TrueAffDataset(Dataset):
    def __init__(self,split,nfold,args,oracle):
        #super().__init__(args,oracle)
        print('using True Aff Dataset')
        self.script_dir = os.path.dirname(__file__)
        self.readseqs()
        self.oracle = oracle
        self.replicate = args.seed 
        #x,ppost_score,sasa_score,aff_score = self.remove_bad_aff(x,ppost_score,sasa_score,aff_score)
        print(len(self.seqs[0]))
        #y = [math.exp((i+75)) for i in y]
        self.current = len(self.seqs)
        self.max_size = 10000
        split = int(0.2*self.current)
        self.train = self.seqs[:split]
        self.valid = self.seqs[split:]
        self.weights = np.array([1.0,0.0,1.0,20.0,1.0,1.0])
        self.current = len(self.train)
        self.ppost_train_scores = self.ppost_score[:split]
        self.ppost_valid_scores = self.ppost_score[split:]
        self.sasa_train_scores = self.sol_score[:split]
        self.sasa_valid_scores = self.sol_score[split:]
        self.aff_train_scores = self.aff_score[:split]
        self.aff_valid_scores = self.aff_score[split:]
        self.var_train_scores = self.var_score[:split]
        self.var_valid_scores = self.var_score[split:]
        self.train_scores = self.ppost_train_scores+self.weights[3] * (self.weights[1]*self.sasa_train_scores + self.weights[2]*(-self.aff_train_scores + self.weights[4]*self.var_train_scores))
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.ppost_valid_scores+self.weights[3] * (self.weights[1]*self.sasa_valid_scores + self.weights[2]*(-self.aff_valid_scores + self.weights[4]*self.var_valid_scores))
        self.nfold = 5
        
    def add_seq(self,seq,r):
        if seq[-1] == '%':
            seq = seq[:-1]
        if len(seq) == 33:
            if self.current < self.max_size:
                self.train.append(seq)
            else:
                self.train[self.current % self.max_size] = seq
            self.train_scores[self.current % self.max_size] = r
            self.current += 1

    def save_dataset(self):
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_train_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'w') as f:
            for seq in self.train:
                f.write(seq+'\n')
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_valid_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'w') as f:
            for seq in self.valid:
                f.write(seq+'\n')
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_train_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        np.save(file_path,self.train_scores)
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_valid_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        np.save(file_path,self.valid_scores)

    def load_dataset(self):
        train_seqs = []
        valid_seqs = []
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_train_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                train_seqs.append(line.split('\n')[0])
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_valid_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                valid_seqs.append(line.split('\n')[0])
        self.train = train_seqs
        self.valid = valid_seqs
        self.current = len(self.train)
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_train_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        self.train_scores = np.load(file_path)
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_valid_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        self.valid_scores = np.load(file_path)

    def readseqs(self):
        self.seqs = []
        file_path = os.path.join(self.script_dir,'..','true_aff','muts.txt')
        with open(file_path,'r') as f:
            for line in f:
                self.seqs.append(line.split('\n')[0])
        self.ppost_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_ppost_score.npy'))
        self.sol_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_sol_score.npy'))
        self.aff_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_aff_score.npy'))
        self.var_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_var_score.npy'))
    
    def set_weights(self,w):
        self.weights = w
        self.train_scores = self.weights[0] * self.ppost_train_scores + self.weights[3] * (self.weights[1]*self.sasa_train_scores + self.weights[2]*(-self.aff_train_scores + self.weights[4]*self.var_train_scores))
        self.current = len(self.train_scores)
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.weights[0] * self.ppost_valid_scores + self.weights[3] * (self.weights[1]*self.sasa_valid_scores + self.weights[2]*(-self.aff_valid_scores + self.weights[4]*self.var_valid_scores))
        self.get_mcmc_weights(self.weights)

    def get_mcmc_weights(self,weights):
        self.mcmc_seqs = []
        file_path = os.path.join(self.script_dir,'gen_seqs','mcmc','true_aff','mcmc_true_aff_exp_lim_burn_dupl_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:1.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                self.mcmc_seqs.append(line.split('\n')[0])
        self.mcmc_length = len(self.mcmc_seqs)
        file_path = os.path.join(self.script_dir,'gen_seqs','mcmc','true_aff','mcmc_true_aff_exp_lim_burn_dupl_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:1.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],self.replicate))
        self.mcmc_score = np.load(file_path)
        sol_r,aff_r,var_r = self.oracle.return_indiv_scores(random.choices(self.mcmc_seqs,k = min(self.mcmc_length,500)))
        self.expected_aff = np.mean(aff_r + weights[4] * var_r)
        self.expected_sol = np.mean(sol_r)
        split = int(0.8*self.mcmc_length)
        train_indices = sorted(np.random.randint(0, len(self.mcmc_seqs), split).tolist())
        idx = 0
        valid_indices = []
        for i in range(len(self.mcmc_seqs)):
            if not i == train_indices[idx]:
                valid_indices.append(i)
            else:
                idx = idx + 1
            
        #valid_indices = [i for i in range(len(self.mcmc_seqs)) if i not in train_indices]
        self.mcmc_seqs_train = [self.mcmc_seqs[i] for i in train_indices]
        self.mcmc_seqs_valid = [self.mcmc_seqs[i] for i in valid_indices]
        self.mcmc_score_train = self.mcmc_score[train_indices]
        self.mcmc_score_valid = self.mcmc_score[valid_indices]

    def sample(self, n):
        n = min(n,len(self.train))
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def sample_valid(self,n):
        n = min(n,len(self.valid))
        indices = np.random.randint(0, len(self.valid), n)
        return ([self.valid[i] for i in indices],
                [self.valid_scores[i] for i in indices])

    def sample_mcmc(self,n):
        n = min(n,len(self.mcmc_seqs))
        indices = np.random.randint(0, len(self.mcmc_seqs), n)
        return ([self.mcmc_seqs[i] for i in indices],
                [self.mcmc_score[i] for i in indices])

    def sample_mcmc_valid(self,n):
        n = min(n,len(self.mcmc_seqs_valid))
        indices = np.random.randint(0, len(self.mcmc_seqs_valid), n)
        return ([self.mcmc_seqs_valid[i] for i in indices],
                [self.mcmc_score_valid[i] for i in indices])

    def sample_first_n(self,n):
        return(self.train[0:n],self.train_scores[0:n])

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/self.nfold):
                self.valid.append(x)
                val.append(score)
            else:
                self.train.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)

    def validation_set(self):
        return self.valid, self.valid_scores

    def _top_k(self, data, k):
        topk_scores, topk_prots = [], []
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.concatenate((topk_scores, data[1][indices]))
        topk_prots = np.concatenate((topk_prots, np.array(data[0])[indices]))
        return topk_prots.tolist(), topk_scores

    def top_k(self, k):
        data = (self.train + self.valid, np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]))
        data = (seqs, scores)
        return self._top_k(data, k)

class TrueAffHardDataset(Dataset):
    def __init__(self,split,nfold,args,oracle):
        #super().__init__(args,oracle)
        print('using True Aff Dataset')
        self.script_dir = os.path.dirname(__file__)
        self.readseqs()
        self.oracle = oracle
        self.replicate = args.seed 
        #x,ppost_score,sasa_score,aff_score = self.remove_bad_aff(x,ppost_score,sasa_score,aff_score)
        print(len(self.seqs[0]))
        #y = [math.exp((i+75)) for i in y]
        self.current = len(self.seqs)
        self.max_size = 10000
        split = int(0.2*self.current)
        self.train = self.seqs[:split]
        self.valid = self.seqs[split:]
        self.weights = np.array([1.0,0.0,1.0,20.0,1.0,1.0])
        self.current = len(self.train)
        self.ppost_train_scores = self.ppost_score[:split]
        self.ppost_valid_scores = self.ppost_score[split:]
        self.sasa_train_scores = self.sol_score[:split]
        self.sasa_valid_scores = self.sol_score[split:]
        self.aff_train_scores = self.aff_score[:split]
        self.aff_valid_scores = self.aff_score[split:]
        self.var_train_scores = self.var_score[:split]
        self.var_valid_scores = self.var_score[split:]
        self.train_scores = self.ppost_train_scores+self.weights[3] * (self.weights[1]*self.sasa_train_scores + self.weights[2]*(-self.aff_train_scores + self.weights[4]*self.var_train_scores))
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.ppost_valid_scores+self.weights[3] * (self.weights[1]*self.sasa_valid_scores + self.weights[2]*(-self.aff_valid_scores + self.weights[4]*self.var_valid_scores))
        self.nfold = 5
        
    def add_seq(self,seq,r):
        if seq[-1] == '%':
            seq = seq[:-1]
        if len(seq) == 33:
            if self.current < self.max_size:
                self.train.append(seq)
            else:
                self.train[self.current % self.max_size] = seq
            self.train_scores[self.current % self.max_size] = r
            self.current += 1

    def save_dataset(self):
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_train_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'w') as f:
            for seq in self.train:
                f.write(seq+'\n')
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_valid_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'w') as f:
            for seq in self.valid:
                f.write(seq+'\n')
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_train_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        np.save(file_path,self.train_scores)
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_valid_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        np.save(file_path,self.valid_scores)

    def load_dataset(self):
        train_seqs = []
        valid_seqs = []
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_train_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                train_seqs.append(line.split('\n')[0])
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_valid_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                valid_seqs.append(line.split('\n')[0])
        self.train = train_seqs
        self.valid = valid_seqs
        self.current = len(self.train)
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_train_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        self.train_scores = np.load(file_path)
        file_path = os.path.join(self.script_dir, 'saved_datasets','true_aff_hard_valid_scores_seqs_aff:{}_sol:{}_gw:{}_beta:{}_rep:{}.txt'.format(self.weights[2],self.weights[1],self.weights[3],self.weights[4],self.replicate))
        self.valid_scores = np.load(file_path)

    def readseqs(self):
        self.seqs = []
        file_path = os.path.join(self.script_dir,'..','true_aff','muts.txt')
        with open(file_path,'r') as f:
            for line in f:
                self.seqs.append(line.split('\n')[0])
        self.ppost_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_ppost_score.npy'))
        self.sol_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_sol_score.npy'))
        self.aff_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_aff_score.npy'))
        self.var_score = np.load(os.path.join(self.script_dir,'..','true_aff','true_aff_var_score.npy'))
    
    def set_weights(self,w):
        self.weights = w
        self.train_scores = self.weights[0] * self.ppost_train_scores + self.weights[3] * (self.weights[1]*self.sasa_train_scores + self.weights[2]*(-self.aff_train_scores + self.weights[4]*self.var_train_scores))
        self.current = len(self.train_scores)
        self.train_scores = np.concatenate((self.train_scores,np.zeros((self.max_size-self.current))))
        self.valid_scores = self.weights[0] * self.ppost_valid_scores + self.weights[3] * (self.weights[1]*self.sasa_valid_scores + self.weights[2]*(-self.aff_valid_scores + self.weights[4]*self.var_valid_scores))
        self.get_mcmc_weights(self.weights)

    def get_mcmc_weights(self,weights):
        self.mcmc_seqs = []
        file_path = os.path.join(self.script_dir,'gen_seqs','mcmc','true_aff_hard','mcmc_true_aff_hard_exp_lim_burn_dupl_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:1.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],self.replicate))
        with open(file_path,'r') as f:
            for line in f:
                self.mcmc_seqs.append(line.split('\n')[0])
        self.mcmc_length = len(self.mcmc_seqs)
        file_path = os.path.join(self.script_dir,'gen_seqs','mcmc','true_aff_hard','mcmc_true_aff_hard_exp_lim_burn_dupl_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:1.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],self.replicate))
        self.mcmc_score = np.load(file_path)
        sol_r,aff_r,var_r = self.oracle.return_indiv_scores(random.choices(self.mcmc_seqs,k = min(self.mcmc_length,500)))
        self.expected_aff = np.mean(aff_r + weights[4] * var_r)
        self.expected_sol = np.mean(sol_r)
        split = int(0.8*self.mcmc_length)
        train_indices = sorted(np.random.randint(0, len(self.mcmc_seqs), split).tolist())
        idx = 0
        valid_indices = []
        for i in range(len(self.mcmc_seqs)):
            if not i == train_indices[idx]:
                valid_indices.append(i)
            else:
                idx = idx + 1
            
        #valid_indices = [i for i in range(len(self.mcmc_seqs)) if i not in train_indices]
        self.mcmc_seqs_train = [self.mcmc_seqs[i] for i in train_indices]
        self.mcmc_seqs_valid = [self.mcmc_seqs[i] for i in valid_indices]
        self.mcmc_score_train = self.mcmc_score[train_indices]
        self.mcmc_score_valid = self.mcmc_score[valid_indices]

    def sample(self, n):
        n = min(n,len(self.train))
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def sample_valid(self,n):
        n = min(n,len(self.valid))
        indices = np.random.randint(0, len(self.valid), n)
        return ([self.valid[i] for i in indices],
                [self.valid_scores[i] for i in indices])

    def sample_mcmc(self,n):
        n = min(n,len(self.mcmc_seqs))
        indices = np.random.randint(0, len(self.mcmc_seqs), n)
        return ([self.mcmc_seqs[i] for i in indices],
                [self.mcmc_score[i] for i in indices])

    def sample_mcmc_valid(self,n):
        n = min(n,len(self.mcmc_seqs_valid))
        indices = np.random.randint(0, len(self.mcmc_seqs_valid), n)
        return ([self.mcmc_seqs_valid[i] for i in indices],
                [self.mcmc_score_valid[i] for i in indices])

    def sample_first_n(self,n):
        return(self.train[0:n],self.train_scores[0:n])

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/self.nfold):
                self.valid.append(x)
                val.append(score)
            else:
                self.train.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)

    def validation_set(self):
        return self.valid, self.valid_scores

    def _top_k(self, data, k):
        topk_scores, topk_prots = [], []
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.concatenate((topk_scores, data[1][indices]))
        topk_prots = np.concatenate((topk_prots, np.array(data[0])[indices]))
        return topk_prots.tolist(), topk_scores

    def top_k(self, k):
        data = (self.train + self.valid, np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]))
        data = (seqs, scores)
        return self._top_k(data, k)


if __name__ == '__main__':
    dataset = CovidAllDataset()