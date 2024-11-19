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

def get_seeds_gp_2(n_chains):
    seqs = ['GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS']*n_chains
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
    print(np.sqrt(np.max((V)/(W+0.01))))
    return np.sqrt(np.max((V)/(W+0.01)))

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


def para_mcmc(oracle,seed,l_chain,n_chains,skip,dist_lim):
    ori_seed = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
    seq = copy.deepcopy(seed)
    seqs = []
    step = 1
    log_probs = np.zeros((l_chain,n_chains))
    logging = []
    ppost_logging = []
    sol_logging = []
    curr_r = oracle(seed)
    for i in tqdm(range(l_chain)):
        #new_seq = sample_para(seq)
        new_seq = sample_para_lim(seq,ori_seed)
        new_r,aff_r = oracle(new_seq,return_aff = True)
        alpha = np.exp((new_r-curr_r))
        prob = np.random.rand(n_chains)
        for j in range(n_chains):
            if distance(new_seq[j],ori_seed) >= 7:
                    print('very bad!')
            if prob[j] <= alpha[j] and (distance(new_seq[j],ori_seed) < 7 or not dist_lim):
                seq[j] = new_seq[j]
                curr_r[j] = new_r[j]
        if (i % 50 == 0):
            print(seq)
            print(curr_r)
        if (i % step == 0 and i >= skip):
            seqs.append(copy.deepcopy(seq))
            log_probs[i] = curr_r
    bit_chains = one_hot_vec_para(seqs,n_chains)
    return seqs,log_probs,bit_chains

def get_prev_seeds(n_chains,weights,dist_lim,replicate):
    if not dist_lim:
        seqs = []
        with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs[-n_chains:]
    else:
        seqs = []
        with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
            for line in f:
                seqs.append(line.split('\n')[0])
        return seqs[-n_chains:]

def sample_chains_para(oracle,keep_going,wei,replicate):
    n_chains = 8
    l_chain = 20000
    skip = 0
    use_annealing = False
    dist_lim = True
    weights = np.array(wei)
    
    if not keep_going and not use_annealing:
        seed = get_seeds_gp_2(n_chains)
    elif not keep_going and use_annealing:
        print('using annealing')
        annealing_gw = wei[3] - 5.0
        annealing_w = np.array(wei)
        annealing_w[3] = annealing_gw
        seed = get_prev_seeds(n_chains,annealing_w,dist_lim,replicate)
    else:
        seed = get_prev_seeds(n_chains,weights,dist_lim,replicate)

    #oracle = get_oracle(args)
    oracle.set_weights(weights)
    oracle.gen_all_cdr = True
    oracle.enc_type = 'esm_t6'
    oracle.use_hum = True
        
    seqs,r,bit_chains = para_mcmc(oracle,seed,l_chain,n_chains,skip,dist_lim)
    print(bit_chains.shape)
    r = np.reshape(r, -1)
    if not keep_going:
        if not dist_lim:
            with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'w') as f:
                for seq in seqs:
                    for a in seq:
                        f.write('{}\n'.format(a))

            np.save('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),r)
        else:
            with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'w') as f:
                for seq in seqs:
                    for a in seq:
                        f.write('{}\n'.format(a))

            np.save('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),r)
    else:
        if not dist_lim:
            with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'a') as f:
                for seq in seqs:
                    for a in seq:
                        f.write('{}\n'.format(a))
            r_prev = np.load('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno_rep:{}:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
            r = np.concatenate((r_prev,r))
            np.save('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),r)
        else:
            with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'a') as f:
                for seq in seqs:
                    for a in seq:
                        f.write('{}\n'.format(a))
            r_prev = np.load('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
            r = np.concatenate((r_prev,r))
            np.save('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),r)
    

def burn_in(wei,replicate):
    print(wei)
    n_chains = 8
    skip = 0
    weights = np.array(wei)
    seqs = []
    seqs_filename = './lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate)
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

    score_filename = './lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate)
    scores = np.load(score_filename)
    scores = scores.reshape((l_chain,n_chains))
    min_gr = 5.0
    min_skip = 0
    bit_chains = one_hot_vec_para(seqs,n_chains)
    print(bit_chains.shape)
    if bit_chains.shape[1] >= 300000:
        bit_chains = bit_chains[:,-80000:,:,:]
        l_chain = 80000
        seqs = seqs[-80000:]
        scores = scores[-80000:]
        print(bit_chains.shape)
        print(scores.shape)
    for k in tqdm(range(int(l_chain/1000))):
        skip = k * 1000
        gr = gelman_rubin(bit_chains[:,skip:,:,:],n_chains ,l_chain - skip)
        if gr < min_gr:
            min_skip = skip
            min_gr = gr
    seqs_filename = './lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate)

    with open(seqs_filename,'w') as f:
        for seq in seqs[min_skip:]:
            for a in seq:
                f.write('{}\n'.format(a))
    
    scores = scores[min_skip:].reshape(-1)

    score_filename = './lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate)
    np.save(score_filename,scores)
    return (min_skip,min_gr)
      

def main(args):
    oracle = get_oracle(args)
    replicate = 1
    print('using replicate:{}'.format(replicate))
    torch.manual_seed(replicate - 1)
    np.random.seed(replicate - 1)
    #np.random.seed()
    args.device = torch.device('cuda')
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    keep_going = False
    gelman_rubin = {}
    for j in global_weights:
        for b in beta:
            weights = [[1.0,0.15,0.85,j,b,1.0],[1.0,0.125,0.875,j,b,1.0],[1.0,0.1,0.9,j,b,1.0],[1.0,0.05,0.95,j,b,1.0],[1.0,0.0,1.0,j,b,1.0]]
            for w in weights:
                if keep_going:
                    gr = burn_in(w,replicate)
                    if gr[1] > 1.1:
                        sample_chains_para(oracle,keep_going,w,replicate)
                        gr = burn_in(w,replicate)
                        gelman_rubin['sol_wei:{}_gw:{}_beta:{}'.format(w[1],w[3],w[4])] = gr
                else:
                    sample_chains_para(oracle,keep_going,w,replicate)
                    gr = burn_in(w,replicate)
                    gelman_rubin['sol_wei:{}_gw:{}_beta:{}'.format(w[1],w[3],w[4])] = gr
    for key in gelman_rubin:
        print(key)
        print(gelman_rubin[key])

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)