import numpy as np
from Levenshtein import distance
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm
from Levenshtein import distance
from torch import nn
from torch import from_numpy
import torch
import os

class delta_nn(torch.nn.Module):
    def __init__(self):
        super(delta_nn,self).__init__()
        #self.size = args.hidden_size*args.gen_max_len*2
        self.dense = torch.nn.Linear(2, 8)
        self.RELU = torch.nn.ReLU()
        self.dense_2 = torch.nn.Linear(8, 2)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = self.RELU(x)
        x = self.dense_2(x)
        return x

class true_aff_oracle:
    def __init__(self):
        self.amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
        self.reverse_dict = {0:'A', 1:'C', 2:'D',3:'E', 4:'F', 5:'G', 6:'H', 7:'I', 8:'K', 9:'L', 10:'M', 11:'N', 12:'P', 13:'Q', 14:'R', 15:'S', 16:'T', 17:'V', 18:'W', 19:'Y'}
        self.aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        self.seed = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
        self.best_seq = self.seed
        self.seq_length = 33
        self.method = 'simple'
        self.script_dir = os.path.dirname(__file__)
        if self.method == 'gamma':
            self.setup_gamma_pwm()
        elif self.method == 'simple':
            #self.setup_simple_pwm()
            self.load_simple_pwm()
        elif self.method == 'hard':
            #self.setup_hard_pwm()
            self.load_hard_pwm()
        elif self.method == 'mix':
            self.setup_mix_pwm()
        elif self.method == 'delta':
            self.setup_delta()
        else:
            self.setup_norm_pwm()
        #self.pwm = np.load('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/model_params/pwm.npy')
        #self.sopwm = np.load('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/model_params/sopwm.npy')
    
    def setup_simple_pwm(self):
        self.pwm_loc = 0.5
        self.pwm_scale = 0.5
        self.dpwm_loc = 0
        self.dpwm_scale = 0.5
        self.bias = 1.606332142168377
        self.pwm = norm.rvs(loc = self.pwm_loc, scale = self.pwm_scale, size = (self.seq_length,20)) 
        self.sopwm = norm.rvs(loc = self.dpwm_loc, scale = self.dpwm_scale, size = (self.seq_length,self.seq_length,20,20))
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_simple.npy')
        np.save(file_path,self.pwm)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_simple.npy')
        np.save(file_path,self.sopwm)

    def load_simple_pwm(self):
        self.method = 'simple'
        self.bias = 1.606332142168377
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_simple.npy')
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_simple.npy')
        self.pwm = np.load(file_path)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_simple.npy')
        self.sopwm = np.load(file_path)

    def setup_hard_pwm(self):
        self.pwm_loc = 0.0
        self.pwm_scale = 0.5
        self.dpwm_loc = 0.5
        self.dpwm_scale = 0.5
        self.bias = 1.606332142168377
        self.pwm = norm.rvs(loc = self.pwm_loc, scale = self.pwm_scale, size = (self.seq_length,20)) 
        self.sopwm = norm.rvs(loc = self.dpwm_loc, scale = self.dpwm_scale, size = (self.seq_length,self.seq_length,20,20))
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_hard.npy')
        np.save(file_path,self.pwm)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_hard.npy')
        np.save(file_path,self.sopwm)

    def load_hard_pwm(self):
        self.method = 'hard'
        self.bias = 1.606332142168377
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_hard.npy')
        self.pwm = np.load(file_path)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_hard.npy')
        self.sopwm = np.load(file_path)

    def setup_gamma_pwm(self):
        self.loc = 0
        self.scale = 1.2710445342639454
        self.gamma = 1.0
        self.bias = 3.606332142168377
        self.pwm = norm.rvs(loc = self.loc, scale = self.scale, size = (self.seq_length,20)) 
        self.best_seq = self.seed
        self.sopwm = norm.rvs(loc = self.loc, scale = self.scale, size = (self.seq_length,self.seq_length,20,20))
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_gamma.npy')
        np.save(file_path,self.pwm)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_gamma.npy') 
        np.save(file_path,self.sopwm)

    def setup_norm_pwm(self):
        self.mut1_loc = 0.5833171
        self.mut2_loc = 0.013485198
        self.mut1_scale = 0.89960486
        self.mut2_scale = 0.69764745
        self.bias = 2.13
        self.pwm = norm.rvs(loc = self.mut1_loc, scale = self.mut1_scale, size = (self.seq_length,20))
        self.sopwm = norm.rvs(loc = self.mut2_loc, scale = self.mut2_scale, size = (self.seq_length,self.seq_length,20,20))
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_norm.npy')
        np.save(file_path,self.pwm)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_norm.npy')
        np.save(file_path,self.sopwm)

    def setup_mix_pwm(self):
        self.bias = 2.86
        self.mut1_loc = 0.3230
        self.mut1_scale = 0.4063
        self.mut2_loc = 0.001
        self.mut2_scale = 1.075
        self.pwm = norm.rvs(loc = self.mut1_loc, scale = self.mut1_scale, size = (self.seq_length,20))
        self.sopwm = norm.rvs(loc = self.mut2_loc/3.0, scale = self.mut2_scale/3.0, size = (self.seq_length,self.seq_length,20,20))
        file_path = os.path.join(self.script_dir,'model_params', 'pwm_mix.npy')
        np.save(file_path,self.pwm)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_mix.npy')
        np.save(file_path,self.sopwm)

    def setup_delta(self):
        self.bias = 2.4343119855513278 - 1.5
        self.first_order_a = 3.156848845533767
        self.first_order_loc = -1.4985397389985127
        self.first_order_scale = 0.812597297971915
        self.gamma = 1.0
        self.pwm = gamma.rvs(self.first_order_a , loc = self.first_order_loc, scale = self.first_order_scale, size = (self.seq_length,20))
        self.delta_model = delta_nn().cuda()
        file_path = os.path.join(self.script_dir,'..','Covid','model_params', 'deltann_model_param.pt')
        self.delta_model.load_state_dict(torch.load(file_path))
        self.delta_model = self.delta_model.cpu()
        self.delta_model.eval()

    def compare_pwm(self):
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_simple.npy')
        simple_pwm = np.load(file_path)
        print(simple_pwm)
        file_path = os.path.join(self.script_dir,'model_params', 'sopwm_hard.npy')
        hard_pwm = np.load(file_path)
        print(hard_pwm)
        print(simple_pwm - hard_pwm)

    def score_delta(self,seqs):
        seqs_i = []
        seed_i = [self.amino_acid_dic[k] for k in self.seed]
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        scores = []
        distances = []
        for seq in seqs_i:
            score = 0
            dist = 0
            for pos in range(len(seq)):
                if seq[pos] != seed_i[pos]:
                    score += self.pwm[pos][seq[pos]]
                    dist += 1
            distances.append(dist)
            scores.append(score)
        distances2 = torch.Tensor(distances).reshape((-1,1))
        scores2 = torch.Tensor(scores).reshape((-1,1))
        inp = torch.cat((scores2,distances2),dim = 1)
        print(inp.shape)
        with torch.no_grad():
            output = self.delta_model(inp)
            mean = output[:,0].numpy()
            var = torch.exp(output[:,1]).numpy()
            deltas = np.random.normal(mean,var/2)
            print(np.mean(mean))
            print(np.max(var))
        print(deltas)
        deltas = self.gamma * deltas# * distances
        first_order = np.array(scores)
        scores = self.bias + first_order + deltas
        return scores,first_order,deltas

    def score_without_noise(self,seqs):
        seqs_i = []
        seed_i = [self.amino_acid_dic[k] for k in self.seed]
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        s = []
        first_orders = []
        second_orders = []
        for seq in seqs_i:
            p = 0
            fo = 0
            so = 0
            for pos in range(len(seq)):
                if seq[pos] != seed_i[pos]:
                    p += self.pwm[pos][seq[pos]]
                    fo += 1
                for pos2 in range(pos + 1,len(seq)):
                    if seq[pos] != seed_i[pos] and seq[pos2] != seed_i[pos2]:
                        p+= self.sopwm[pos,pos2,seq[pos],seq[pos2]]
                        so += 1
            p = p/(max(1,np.sqrt(fo + so)))
            #epistasis = norm.rvs(loc = (fo - 3),scale = max(0.0001,0.3*(2*fo - 3)))
            epistasis_mean = self.gamma * ((fo - 3)*0.5)
            epistasis_std = self.gamma * fo * 0.4
            score = (p/np.sqrt(self.scale))*epistasis_std + epistasis_mean + self.bias
            s.append(score)
            #s.append(p + self.bias)
        return s

    def score_without_noise_simple(self,seqs):
        seqs_i = []
        seed_i = [self.amino_acid_dic[k] for k in self.seed]
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        s = []
        first_orders = []
        second_orders = []
        for seq in seqs_i:
            score = 0
            for pos in range(len(seq)):
                if seq[pos] != seed_i[pos]:
                    score += self.pwm[pos][seq[pos]]
                for pos2 in range(pos + 1,len(seq)):
                    if seq[pos] != seed_i[pos] and seq[pos2] != seed_i[pos2]:
                        score += self.sopwm[pos,pos2,seq[pos],seq[pos2]]
            #s.append(np.min((score + self.bias,7)))
            s.append(self.bias + score)
        return s

    def score_without_noise_hard(self,seqs):
        seqs_i = []
        seed_i = [self.amino_acid_dic[k] for k in self.seed]
        for seq in seqs:
            seqs_i.append([self.amino_acid_dic[k] for k in seq])
        s = []
        first_orders = []
        second_orders = []
        for seq in seqs_i:
            score = 0
            for pos in range(len(seq)):
                if seq[pos] != seed_i[pos]:
                    score += self.pwm[pos][seq[pos]]
                for pos2 in range(pos + 1,len(seq)):
                    if seq[pos] != seed_i[pos] and seq[pos2] != seed_i[pos2]:
                        score += self.sopwm[pos,pos2,seq[pos],seq[pos2]]
            #s.append(np.min((score + self.bias,7)))
            s.append(self.bias + score)
        return s

    def score_with_noise(self,seqs,var):
        if self.method == 'delta':
            s,fo,d = self.score_delta(seqs)
            s = s + np.random.normal(loc = 0.0, scale = var, size = len(seqs))
            return s,fo,d
        elif self.method == 'simple':
            s = self.score_without_noise_simple(seqs)
            s = s + np.random.normal(loc = 0.0, scale = var, size = len(seqs))
            return s
        elif self.method == 'hard':
            s = self.score_without_noise_hard(seqs)
            s = s + np.random.normal(loc = 0.0, scale = var, size = len(seqs))
            return s
        else:
            s = self.score_without_noise(seqs)
            s = s + np.random.normal(loc = 0.0, scale = var, size = len(seqs))
            return s

    def deep_mutational_scan(self):
        mutants = []
        for pos in range(len(self.best_seq)):
            for aa in self.aa_list:
                seq = [i for i in self.best_seq]
                seq[pos] = aa
                mutants.append(''.join(seq))
        return mutants


    def generate_mutants(self,n_mut,n_mutations):
        mutants = []
        while len(mutants) < n_mut:
            if n_mutations == 1:
                break
            seq = [i for i in self.best_seq]
            if n_mutations == 0:
                n = np.random.choice(3, 1, p=[0, 0.15, (1 - 0.15)]) + 1
                n = n[0]
            else:
                n = n_mutations
            #n = 3
            pos = np.random.choice(range(self.seq_length),size = n,replace = False)
            muts = np.random.choice(self.aa_list,size = n, replace = True)
            for p in range(n):
                seq[pos[p]] = muts[p]
            if ''.join(seq) not in mutants:
                mutants.append(''.join(seq))
        if n_mutations == 0:
            mutants = mutants + self.deep_mutational_scan()
            file_path = os.path.join(self.script_dir, 'muts_txt')
            with open(file_path,'w') as f:
                for m in mutants:
                    f.write(m+'\n')
        elif n_mutations == 1:
            mutants = self.deep_mutational_scan()
            file_path = os.path.join(self.script_dir, 'muts_{}.txt'.format(n_mutations))
            with open(file_path,'w') as f:
                for m in mutants:
                    f.write(m+'\n')
        else:
            file_path = os.path.join(self.script_dir, 'muts_{}.txt'.format(n_mutations))
            with open(file_path,'w') as f:
                for m in mutants:
                    f.write(m+'\n')


    def score_mutants(self,n_mutations = 0):
        mutants = []
        if n_mutations == 0:
            file_path = os.path.join(self.script_dir, 'muts.txt'.format(n_mutations))
            with open(file_path,'r') as f:
                for line in f:
                    mutants.append(line.split('\n')[0])
        else:
            file_path = os.path.join(self.script_dir, 'muts_{}.txt'.format(n_mutations))
            with open(file_path,'r') as f:
                for line in f:
                    mutants.append(line.split('\n')[0])
        for v in [0.0,0.5,1.0]:
            if self.method == 'delta':
                s,fo,d = self.score_with_noise(mutants,v)
                np.save('./muts_score_noise:{}_method:{}_n_mutations:{}'.format(v,self.method,n_mutations),s)
                np.save('./muts_fo_noise:{}_method:{}_n_mutations:{}'.format(v,self.method,n_mutations),fo)
                np.save('./muts_deltas_noise:{}_method:{}_n_mutations:{}'.format(v,self.method,n_mutations),d)
            else:
                s = self.score_with_noise(mutants,v)
                np.save('./muts_score_noise:{}_method:{}_n_mutations:{}'.format(v,self.method,n_mutations),s)


    def get_mut_idx(self,mut_count):
        mutants = []
        file_path = os.path.join(self.script_dir, 'muts.txt')
        with open(file_path,'r') as f:
            for line in f:
                mutants.append(line.split('\n')[0])
        idx = [i for i in range(len(mutants)) if distance(mutants[i],self.seed) == mut_count]
        return idx

    def get_mutants(self):
        seqs = []
        file_path = os.path.join(self.script_dir, 'muts.txt')
        with open('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts.txt','r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs

    def plot_score(self,n_mutations):
        for v in [0.0,0.5,1.0]:
            score = np.load('./muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(v,self.method,n_mutations))
            #fo = np.load('./muts_fo_noise:{}_method:{}_n_mutations:{}.npy'.format(v,self.method,n_mutations))
            #deltas = np.load('./muts_deltas_noise:{}_method:{}_n_mutations:{}.npy'.format(v,self.method,n_mutations))

            mean_score = np.round(np.mean(score),decimals = 2)
            var_score = np.round(np.var(score),decimals = 2)
            counts, bins = np.histogram(score, bins = [i * 0.5 for i in range(-5,28)])
            file_path = os.path.join(self.script_dir,'..','Covid','data','covid_scores.npy')
            true_score = np.load(file_path)
            counts2, bins2 = np.histogram(true_score, bins = [i * 0.5 for i in range(-5,28)])
            plt.stairs(counts,bins,label = 'generated data')
            plt.stairs(counts2,bins2,label = 'true_data')
            plt.xlabel('fake score of generated sequences')
            plt.ylabel('count')
            plt.title('{} method/noise_{}/mean:{}/var:{}'.format(self.method,v,mean_score,var_score))
            plt.legend()
            plt.savefig('./figures/distrib_fake_aff_scores_{}_noise:{}_n_mutations:{}.png'.format(self.method,v,n_mutations))
            plt.clf()

            mean_score = np.round(np.mean(score),decimals = 2)
            var_score = np.round(np.var(score),decimals = 2)
            counts, bins = np.histogram(score, bins = [i * 0.5 for i in range(-5,3)])
            file_path = os.path.join(self.script_dir,'..','Covid','data','covid_scores.npy')
            true_score = np.load(file_path)
            counts2, bins2 = np.histogram(true_score, bins = [i * 0.5 for i in range(-5,3)])
            plt.stairs(counts,bins,label = 'generated data')
            plt.stairs(counts2,bins2,label = 'true_data')
            plt.xlabel('fake score of generated sequences')
            plt.ylabel('count')
            plt.title('{} method/noise_{}/mean:{}/var:{}'.format(self.method,v,mean_score,var_score))
            plt.legend()
            plt.savefig('./figures/distrib_top_fake_aff_scores_{}_noise:{}_n_mutations:{}.png'.format(self.method,v,n_mutations))
            plt.clf()

            '''
            plt.scatter(score,deltas)
            plt.xlabel('score')
            plt.ylabel('delta')
            plt.savefig('./score_deltas_scatter_method:{}_noise:{}_n_mutations:{}.png'.format(self.method,v,n_mutations))
            plt.clf()

            plt.scatter(score,fo)
            plt.xlabel('score')
            plt.ylabel('fo')
            plt.savefig('./score_fo_scatter_method:{}_noise:{}_n_mutations:{}.png'.format(self.method,v,n_mutations))
            plt.clf()
            '''

    def compare_distributions_per_mut(self):
        plt.rcParams.update({'font.size': 16})
        fig,axs = plt.subplots(1,2,figsize = (16,6))
        noise = [0.0]
        file_path = os.path.join(self.script_dir,'..','Covid','data','covid_scores.npy')
        true_score = np.load(file_path)
        file_path = os.path.join(self.script_dir,'..','Covid','data','covid_mut.npy')
        mut_count = np.load(file_path)
        mut_1 = [i for i in range(len(true_score)) if mut_count[i] == 1]
        mut_2 = [i for i in range(len(true_score)) if mut_count[i] == 2]
        mut_3 = [i for i in range(len(true_score)) if mut_count[i] == 3]
        scores = [-true_score,-true_score[mut_1],-true_score[mut_2],-true_score[mut_3]]
        labels = ['All mutants','# of mutations = 1','# of mutations = 2','# of mutations = 3']
        axs[0].boxplot(scores)
        axs[0].set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
        axs[0].set_ylabel(r'$log_{10}$ ($K_A$ (nM))')
        axs[0].set_ylim(-20,5)
        axs[0].set_title('Sars-Cov-2 dataset')
        scores = []
        labels = []
        for n_mut in range(7):
            var = noise[0]
            score = np.load('./muts_score_noise:{}_method:{}_n_mutations:{}.npy'.format(var,self.method,n_mut))
            scores.append(-score)
            if n_mut == 0:
                labels.append('GP Synthetic Dataset')
            else:
                labels.append('# of mutations = {}'.format(n_mut))
        axs[1].boxplot(scores)
        axs[1].set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
        axs[1].set_ylabel(r'$log_{10}$ (Synthetic $K_A$ (nM))')
        axs[1].set_ylim(-20,5)
        axs[1].set_title('{} epistasis dataset'.format(self.method))
        plt.tight_layout()
        plt.savefig('./figures/histo_all_muts_method:{}.png'.format(self.method))

    def plot_score_mut_count(self,mut_count):
        score = np.load('./muts_score_noise:0.0.npy')
        idx = self.get_mut_idx(mut_count)
        score = score[idx]
        counts, bins = np.histogram(score, bins = [i * 0.2 for i in range(-2,40)])
        plt.stairs(counts,bins)
        plt.savefig('distrib_fake_aff_scores_mut_count:{}.png'.format(mut_count))

if __name__ == '__main__':
    np.random.seed(0)
    oracle = true_aff_oracle()
    #oracle.compare_pwm()
    oracle.method = 'simple'
    oracle.compare_distributions_per_mut()
    oracle.method = 'hard'
    oracle.compare_distributions_per_mut()
    '''
    for n_mutations in range(7):
        print(n_mutations)
        #oracle = true_aff_oracle()
        #print(oracle.score_delta([oracle.seed,oracle.seed]))
        #idx = oracle.get_mut_idx(1)
        #print(len(idx))
        #oracle.generate_mutants(14000,n_mutations)
        #oracle.score_mutants(n_mutations = n_mutations)
        oracle.plot_score(n_mutations = n_mutations)
        #oracle.plot_score_mut_count(1)
        #print(oracle.score_without_noise(['GRAAGMFDL']))
    '''
