import argparse
import gzip
import pickle
import itertools
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from lib.oracle_wrapper import get_oracle

import matplotlib.pyplot as plt
from Levenshtein import distance

parser = argparse.ArgumentParser()

parser.add_argument("--saving", default = False, type = bool)
parser.add_argument("--loading", default = True, type = bool)
parser.add_argument("--saving_num", default = 102, type = int)
parser.add_argument("--loading_num", default = 102, type = int)
parser.add_argument("--gen_learning_rate", default=1e-3, type=float)
parser.add_argument("--gen_Z_learning_rate", default=5e-2, type=float)
parser.add_argument("--gen_num_iterations", default=6000, type=int) # Maybe this is too low?
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_data_sample_per_step", default=64, type=int)
parser.add_argument("--gen_model_type", default="cnn")

parser.add_argument("--save_path", default='results/test_mlp.pkl.gz')
parser.add_argument("--tb_log_dir", default='results/test_mlp')
parser.add_argument("--name", default='test_mlp')
parser.add_argument("--load_scores_path", default='.')

# Multi-round
parser.add_argument("--num_rounds", default=1, type=int)
parser.add_argument("--task", default="random", type=str)
parser.add_argument("--num_sampled_per_round", default=10, type=int) # 10k
parser.add_argument("--num_folds", default=5)
parser.add_argument("--vocab_size", default=22)
parser.add_argument("--max_len", default=65)
parser.add_argument("--gen_max_len", default=27)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--enable_tensorboard", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="none", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=80, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument("--oracle_split", default="D2_target", type=str)
parser.add_argument("--proxy_data_split", default="D1", type=str)
parser.add_argument("--oracle_type", default="MLP", type=str)
parser.add_argument("--oracle_features", default="AlBert", type=str)
parser.add_argument("--medoid_oracle_dist", default="edit", type=str)
parser.add_argument("--medoid_oracle_norm", default=1, type=int)
parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)


# Generator
#parser.add_argument("--gen_learning_rate", default=1e-4, type=float)
#parser.add_argument("--gen_Z_learning_rate", default=5e-3, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
#parser.add_argument("--gen_num_iterations", default=1, type=int) # Maybe this is too low?
#parser.add_argument("--gen_episodes_per_step", default=2, type=int)
parser.add_argument("--gen_num_hidden", default=16, type=int)
parser.add_argument("--hidden_size", default=64, type=int)
parser.add_argument("--layer_norm_eps", default=1e-12, type=float)
parser.add_argument("--num_attention_heads", default=4, type=int)
parser.add_argument("--hidden_act", default='relu', type=str)
parser.add_argument("--initializer_range", default=1.0, type=float)
parser.add_argument("--intermediate_size", default=128, type=int)
parser.add_argument("--gen_num_layers", default=2, type=int)
parser.add_argument("--gen_dropout", default=0.1, type=float)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=1, type=float)
parser.add_argument("--gen_reward_min", default=-8, type=float)
parser.add_argument("--gen_L2", default=0, type=float)
parser.add_argument("--gen_partition_init", default=100, type=float)
parser.add_argument("--pad_token_id", default=21, type=int)

#bytenet args
parser.add_argument("--gen_small_embedding", default=32, type=int)
parser.add_argument("--cnn_hidden_size", default=64, type=int)
parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
parser.add_argument("--cnn_max_r", default=4, type=int)
parser.add_argument("--cnn_n_layers", default=8, type=int)


# Soft-QLearning/GFlownet gen
parser.add_argument("--gen_reward_exp_ramping", default=1, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=1, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument("--gen_random_action_prob", default=0.05, type=float)
parser.add_argument("--gen_sampling_temperature", default=1., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
#parser.add_argument("--gen_data_sample_per_step", default=0, type=int)
# PG gen
parser.add_argument("--gen_do_pg", default=0, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=1e-2, type=float)
# learning partition Z explicitly
parser.add_argument("--gen_do_explicit_Z", default=1, type=int)
#parser.add_argument("--gen_model_type", default="cnn")

# Proxy
parser.add_argument("--proxy_learning_rate", default=1e-4)
parser.add_argument("--proxy_type", default="regression")
parser.add_argument("--proxy_arch", default="mlp")
parser.add_argument("--proxy_num_layers", default=4)
parser.add_argument("--proxy_dropout", default=0.1)

parser.add_argument("--proxy_num_hid", default=64, type=int)
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
parser.add_argument("--proxy_early_stop_to_best_params", default=0, type=int)
parser.add_argument("--proxy_num_iterations", default=30000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)


# Oracle PGen

parser.add_argument("--small_embedding", default=16, type=int)
parser.add_argument("--pgen_hidden_size", default=64, type=int)

amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}


def sample(seq):
    pos = np.random.randint(0,high=33)
    aa = amino_acid[np.random.randint(0,high = 20)]
    if pos == 0:
        new_seq = aa+seq[1:]
    elif pos == 32:
        new_seq = seq[0:32]+aa
    else:
        new_seq = seq[0:pos]+aa+seq[pos+1:]
    return new_seq

def get_mutants():
    seqs = []
    with open('.lib/true_aff/muts.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs

def get_init_true_aff():
    sol_r = np.load('./lib/dataset/preprocess/init_true_aff_sol_preprocess.npy')
    perf_r = np.load('./lib/dataset/preprocess/init_true_aff_performance_preprocess.npy')
    seqs = []
    with open('./lib/dataset/preprocess/init_true_aff_seqs_preprocess.npy','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,sol_r,perf_r

def get_best_simple_seed(n_chains,t_sol):
    seqs,sol,perf = get_init_true_aff()
    idx = [i for i in range(len(sol)) if sol[i] >= t_sol]
    perf = perf[idx]
    seqs = [seqs[i] for i in idx]
    top_idx = np.argmin(perf)
    print(perf[top_idx])
    top_seq = seqs[top_idx]
    print(top_seq)
    return [top_seq] * n_chains

def sample_para(seq):
    new_seq = copy.deepcopy(seq)
    for s in range(len(seq)):
        if len(seq[s]) != 33:
            print('error')
        pos = np.random.randint(0,high=33)
        aa = amino_acid[np.random.randint(0,high = 20)]
        if pos == 0:
            new_seq[s] = aa+seq[s][1:]
        elif pos == 32:
            new_seq[s] = seq[s][0:-1]+aa
        else:
            new_seq[s] = seq[s][0:pos]+aa+seq[s][pos+1:]
    return new_seq

def sample_para_lim(seq,ori_seed):
    new_seq = copy.deepcopy(seq)
    for s in range(len(seq)):
        if len(seq[s]) != 33:
            print('error')
        if distance(seq[s],ori_seed) < 6:
            pos = np.random.randint(0,high=33)
        else:
            list_of_pos = [i for i in range(len(seq[s])) if seq[s][i] != ori_seed[i]]
            pos = np.random.choice(list_of_pos)
        aa = amino_acid[np.random.randint(0,high = 20)]
        if pos == 0:
            new_seq[s] = aa+seq[s][1:]
        elif pos == 32:
            new_seq[s] = seq[s][0:-1]+aa
        else:
            new_seq[s] = seq[s][0:pos]+aa+seq[s][pos+1:]
    return new_seq

def one_hot_vec_para(seqs,n_chains):
    one_hot_seqs = np.zeros([n_chains,len(seqs),33,20])
    for s in range(len(seqs)):
        for c in range(n_chains):
            seq = seqs[s][c]
            for pos in range(len(seq)):
                one_hot_seqs[c,s,pos,amino_acid_dic[seq[pos]]] = 1
    return one_hot_seqs

def mean_chain(seqs):
    p = np.zeros([33,20])
    for s in seqs:
        for pos in range(len(s)):
            p[pos,amino_acid.index(s[pos])] += 1
    p = p/len(seqs)
    print(p)

def var_chain(seqs,mean):
    p = np.zeros([33,20])
    for s in seqs:
        for pos in range(len(s)):
            p[pos,amino_acid.index(s[pos])] += 1
    p = p/len(seqs)
    print(p)

def get_seeds(n_chains):
    seqs = []
    with open('./lib/dataset/IGLM_seqs.txt','r') as f:
        for i in range(n_chains):
            seqs.append(next(f).split('\n')[0])
    return seqs

def get_seeds_gp(n_chains):
    seqs = ['CAKGRAAGTFDSW']*n_chains
    return seqs

def get_seeds_gp_2(n_chains):
    seqs = ['GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS']*n_chains
    return seqs

def get_seeds_max_sol(n_chains):
    seqs = []
    with open('./lib/dataset/aff_sol_glow_10.0_0.0.txt','r') as f:
        for i in range(n_chains):
            seqs.append(next(f).split('\n')[0])
    return seqs

def get_neighbours(seq):
    neigh = []
    for pos in range(len(seq)):
        for aa in amino_acid:
            if aa != seq[pos]:
                if pos == 0:
                    new_seq = aa+seq[1:]
                elif pos == 32:
                    new_seq = seq[0:32]+aa
                else:
                    new_seq = seq[0:pos]+aa+seq[pos+1:]
            neigh.append(new_seq)
    return neigh

def check_seed(oracle):
    seqs = get_seeds(8)
    print(seqs[0])
    neigh = get_neighbours(seqs[1])
    print(neigh)
    r = oracle.get_ppost_score([seqs[1]]).cpu().numpy()
    r_neigh = oracle.get_ppost_score(neigh).cpu().numpy()
    print(r)
    print(np.max(r_neigh))
    print(np.mean(r_neigh))
    print(np.exp(np.max(r_neigh))/np.exp(r))

def autocorr(l_chain,chains):
    res = []
    for time in range(10,1000,5):
        a = []
        for j in range(1000,l_chain-time):
            seqs_1 = chains[0,j]
            seqs_2 = chains[0,j+time]
            a.append(np.sum(np.logical_and(seqs_1,seqs_2)))
        res.append(np.mean(a))
    return res

def autocorr_2(chains,l_chains):
    corr = []
    chains = chains[0]
    l_chains = chains.shape[0]
    print(l_chains)
    for step in range(10,300,5):
        res = np.zeros([33,20])
        for i in range(l_chains-step):
            res += chains[i]*chains[i+step]
        res = res / (l_chains-step)
        corr.append(np.sum(res))
    return corr

def gelman_rubin(chains,n_chains,l_chain):
    print(chains.shape)
    chains_mean = np.mean(chains,axis = 1)
    chains_var = np.var(chains,axis = 1)
    total_mean = np.mean(chains_mean,axis = 0)

    W = np.mean(chains_var,axis = 0)
    
    B = np.copy(chains_mean)
    for i in range(n_chains):
        B[i] = (B[i]-total_mean)**2
    
    B = np.sum(B,axis = 0)
    B = B*l_chain/(n_chains-1)
    
    V = (l_chain-1)/(l_chain) * W + ((n_chains+1)/(n_chains*l_chain)) * B
    #print(np.sqrt((V+0.01)/(W+0.01)))
    V = V.reshape(-1)
    W = W.reshape(-1)
    print(V.shape)
    print(W.shape)
    V = np.array([V[i] * W[i] for i in range(len(V)) if W[i] > 0.001])
    if len(V) > 0:
        print(np.sqrt(np.max((V))))
        return np.sqrt(np.max((V)))
    else:
        return 0

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def fit_mcmc(chains,l_chain):
    res = autocorr_2(chains,l_chain)
    print(res)
    fit_exp(res,np.min(res)-0.001)

def para_antBO(oracle,seed,l_chain,n_chains,skip,dist_lim):
    ori_seed = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
    seq = copy.deepcopy(seed)
    seqs = []
    step = 1
    log_probs = np.zeros((l_chain,n_chains))
    logging = []
    ppost_logging = []
    sol_logging = []
    curr_r, _,_,_ = oracle.trust_region_true_aff(seed)
    for i in tqdm(range(l_chain)):
        new_seq = sample_para_lim(seq,ori_seed)
        new_r, valid,sol_r,ppost_r = oracle.trust_region_true_aff(new_seq)
        for j in range(n_chains):
            if distance(new_seq[j],ori_seed) >= 7:
                    print('very bad!')
            if curr_r[j] <= new_r[j] and valid[j] and (distance(new_seq[j],ori_seed) < 7 or not dist_lim):
                seq[j] = new_seq[j]
                curr_r[j] = new_r[j]
        if (i % 50 == 0):
            print(seq)
            print(curr_r)
            print([distance(i,ori_seed) for i in seq])
            print(sol_r)
            print(ppost_r)
            print(valid)
        if (i % step == 0 and i >= skip):
            seqs.append(copy.deepcopy(seq))
            log_probs[i] = curr_r
    bit_chains = one_hot_vec_para(seqs,n_chains)
    return seqs,log_probs,bit_chains

def get_prev_seeds(n_chains,weights,dist_lim,replicate):
    if not dist_lim:
        seqs = []
        with open('./lib/dataset/antBO_true_aff_exp_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs[-n_chains:]
    else:
        seqs = []
        with open('./lib/dataset/antBO_true_aff_exp_lim_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs[-n_chains:]


def sample_chains_para(oracle,thresholds,replicate):
    n_chains = 800
    l_chain = 200
    skip = 0

    dist_lim = True
    weights = np.array(thresholds)
    
    seed = get_best_simple_seed(n_chains,thresholds[0])
    
    #oracle = get_oracle(args)
    oracle.gen_all_cdr = True
    oracle.enc_type = 'esm_t6'
    oracle.use_hum = True
    oracle.set_antBO_thresholds(thresholds)
    print('hello')
    print(oracle.antBO_min_sol)
    print(oracle.antBO_min_hum)

    var = 1.0
    perc = 0.8
    method = 'simple'
    oracle.update_true_aff_gp(var,perc,method)
    print(oracle.true_aff_method)    
    
    seqs,r,bit_chains = para_antBO(oracle,seed,l_chain,n_chains,skip,dist_lim)
    print(bit_chains.shape)
    r = np.reshape(r, -1)
    if not dist_lim:
        with open('./lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate),'w') as f:
            for seq in seqs:
                for a in seq:
                    f.write('{}\n'.format(a))

        np.save('./lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_prob_sol_min:{}_hum_min:{}_beta:{}_rep:{}.npy'.format(weights[0],weights[1],weights[2],replicate),r)
    else:
        with open('./lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate),'w') as f:
            for seq in seqs:
                for a in seq:
                    f.write('{}\n'.format(a))

        np.save('./lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_prob_sol_min:{}_hum_min:{}_beta:{}_rep:{}.npy'.format(weights[0],weights[1],weights[2],replicate),r)
    
def process_results(wei,replicate):
    print(wei)
    n_chains = 800
    skip = 0
    weights = np.array(wei)
    seqs = []
    seqs_filename = './lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate)
    with open(seqs_filename,'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    l_chain = int(len(seqs)/n_chains)
    seqs_2 = [[] for i in range(l_chain)]
    idx = 0
    for i in range(l_chain):
        for j in range(n_chains):
            seqs_2[i].append(seqs[idx])
            idx += 1
    seqs = seqs_2
    seqs = seqs[-1]
    score_filename = './lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_prob_sol_min:{}_hum_min:{}_beta:{}_rep:{}.npy'.format(weights[0],weights[1],weights[2],replicate)
    scores = np.load(score_filename)
    scores = scores.reshape((l_chain,n_chains))
    scores = scores[-1,:]
    
    seqs_filename = './lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_burn_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate)

    with open(seqs_filename,'w') as f:
        for seq in seqs:
            f.write('{}\n'.format(seq))

    score_filename = './lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_burn_prob_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate)
    np.save(score_filename,scores)

def main(args):
    oracle = get_oracle(args)
    replicate = 2
    print('using replicate:{}'.format(replicate))
    torch.manual_seed(replicate - 1)
    np.random.seed(replicate - 1)
    args.device = torch.device('cuda')
    global_weights = [10.0]
    beta = [0.0]
    keep_going = False
    
    for j in global_weights:
        for b in beta:
            thresholds = [[-10,-120,b],[4,-120,b]]
            for t in thresholds:
                sample_chains_para(oracle,t,replicate)
                process_results(t,replicate)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)