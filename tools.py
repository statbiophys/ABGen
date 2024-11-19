import argparse
import gzip
import pickle
import itertools
import time
from Levenshtein import distance
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import logomaker

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.oracle_wrapper import get_oracle
#from lib.proxy import get_proxy_model
from lib.utils.env import get_tokenizer
#from lib.model.tmodel import AbRep
import matplotlib.pyplot as plt
from gen_iglm import get_thera_seqs
from gen_iglm import get_IGG_seqs
from gen_iglm import get_thera_CDR
from gen_iglm import get_IGLM_seqs_2

parser = argparse.ArgumentParser()

parser.add_argument("--saving", default = False, type = bool)
parser.add_argument("--loading", default = True, type = bool)
parser.add_argument("--saving_num", default = 102, type = int)
parser.add_argument("--loading_num", default = 102, type = int)
parser.add_argument("--gen_learning_rate", default=1e-4, type=float)
parser.add_argument("--gen_Z_learning_rate", default=5e-3, type=float)
parser.add_argument("--gen_num_iterations", default=1, type=int) # Maybe this is too low?
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_data_sample_per_step", default=64, type=int)
parser.add_argument("--gen_model_type", default="cnn")
parser.add_argument("--use_replay_buffer", default=False, type=bool)
parser.add_argument("--log_id", default=103, type=int)
parser.add_argument("--log_results", default=True, type=bool)
parser.add_argument("--restrict_aff", default=False, type=float)

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


def compute_score(oracle):
    seqs = []
    scores = []
    batch_size = 128
    with open('./lib/dataset/OLGA_seqs_full.csv','r') as f:
        for line in f:
            a = line.split(',')
            if len(a[0]) == 13:
                seqs.append(a[0])
    
    num_batches = int(len(seqs)/batch_size)
    for i in tqdm(range(num_batches)):
        aff, var = oracle.get_ppost_score(seqs[i*batch_size:(i+1)*batch_size])
        scores = scores+(aff.tolist())

    with open('./lib/dataset/ppost_len13_2.csv','w') as f:
        for i in range(len(scores)):
            f.write(seqs[i]+','+str(scores[i])+'\n')

def compute_score_2(oracle):
    seqs = []
    scores = []
    batch_size = 128
    oracle.gen_all_cdr = True

    with open('./lib/Covid/data/cdrlist.txt','r') as f:
        for line in f:
            a = line.split('\n')[0]
            seqs.append(a)

    print(len(seqs))
    print(len(seqs[0]))
    score = np.zeros(len(seqs))
    ppost_score = np.zeros(len(seqs))
    sol_score = np.zeros(len(seqs))
    aff_score = np.zeros(len(seqs))
    var_score = np.zeros(len(seqs))

    num_batches = int(len(seqs)/batch_size) + 1
    for i in tqdm(range(num_batches)):
        score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        ppost_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle.get_ppost_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        sol_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle.get_sasa_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        aff, var = oracle.get_aff_score_gp_3(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        aff_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = aff
        var_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = var
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/dataset/full_covid_scores.npy',score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/dataset/full_covid_ppost_score.npy',ppost_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/dataset/full_covid_sol_score.npy',sol_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/dataset/full_covid_aff_score.npy',aff_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/dataset/full_covid_var_score.npy',var_score)
    print(score.shape)
    print(ppost_score.shape)
    print(sol_score.shape)
    print(aff_score.shape)
    print(var_score.shape)


def compute_score_true_aff(oracle):
    seqs = []
    scores = []
    batch_size = 128
    oracle.gen_all_cdr = True

    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            a = line.split('\n')[0]
            seqs.append(a)

    print(len(seqs))
    print(len(seqs[0]))
    score = np.zeros(len(seqs))
    ppost_score = np.zeros(len(seqs))
    sol_score = np.zeros(len(seqs))
    aff_score = np.zeros(len(seqs))
    var_score = np.zeros(len(seqs))
    
    num_batches = int(len(seqs)/batch_size) + 1
    for i in tqdm(range(num_batches)):
        ppost_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle.get_ppost_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        sol_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle.get_sasa_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        aff, var = oracle.get_trueaff_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        aff_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = aff
        var_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = var
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_ppost_score.npy',ppost_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_sol_score.npy',sol_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_aff_score.npy',aff_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_var_score.npy',var_score)
    print(ppost_score.shape)
    print(sol_score.shape)
    print(aff_score.shape)
    print(var_score.shape)

def compute_score_true_aff_hard(oracle):
    seqs = []
    scores = []
    batch_size = 128
    oracle.gen_all_cdr = True
    oracle.update_true_aff_gp(1.0,0.8,'hard')
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            a = line.split('\n')[0]
            seqs.append(a)

    print(len(seqs))
    print(len(seqs[0]))
    score = np.zeros(len(seqs))
    ppost_score = np.zeros(len(seqs))
    sol_score = np.zeros(len(seqs))
    aff_score = np.zeros(len(seqs))
    var_score = np.zeros(len(seqs))
    
    num_batches = int(len(seqs)/batch_size) + 1
    for i in tqdm(range(num_batches)):
        ppost_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle.get_ppost_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        sol_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = oracle.get_sasa_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        aff, var = oracle.get_trueaff_score(seqs[i*batch_size:min((i+1)*batch_size,len(seqs))])
        aff_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = aff
        var_score[i*batch_size:min((i+1)*batch_size,len(seqs))] = var
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_hard_ppost_score.npy',ppost_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_hard_sol_score.npy',sol_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_hard_aff_score.npy',aff_score)
    np.save('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_hard_var_score.npy',var_score)
    print(ppost_score.shape)
    print(sol_score.shape)
    print(aff_score.shape)
    print(var_score.shape)

def distribution_of_scores_true_aff(oracle):
    sol_score = np.load('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_sol_score.npy')
    hum_score = np.load('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/true_aff_ppost_score.npy')

    counts,bins = np.histogram(sol_score)
    plt.stairs(counts,bins)
    plt.savefig('./pareto_final/distrib_sol_true_aff_original.png')
    plt.clf()
    print(np.quantile(sol_score,0.7))
    print(np.quantile(sol_score,0.9))
    counts,bins = np.histogram(hum_score)
    plt.stairs(counts,bins)
    plt.savefig('./pareto_final/distrib_hum_true_aff_original.png')
    print(np.quantile(hum_score,0.7))
    print(np.quantile(hum_score,0.9))

    muts_1 = []
    with open('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts_{}.txt'.format(1),'r') as f:
        for line in f:
            muts_1.append(line.split('\n')[0])
    print(len(muts_1))
    print(muts_1[0])
    print(np.quantile(oracle.get_ppost_score(muts_1).numpy(),0.9))
    print(np.quantile(oracle.get_sasa_score(muts_1),0.9))

def verify_oracle(oracle):
    seqs = []
    true_scores = []
    scores = []
    count = 0
    num_seqs = 10000
    with open('./lib/dataset/OLGA_seqs_full.csv','r') as f:
        for line in f:
            a = line.split(',')
            if len(a[0]) < args.gen_max_len-2:
                if float(a[2]) > -75:
                    seqs.append(a[0])
                    true_scores.append(float(a[2]))
                    count += 1
                    if count > num_seqs:
                        break
    
    num_batches = int(num_seqs/8)
    for i in tqdm(range(num_batches)):
        scores = scores+(oracle(seqs[i*8:(i+1)*8]).tolist())

    print(true_scores[1000:1020])
    print(scores[1000:1020])
    print(len(true_scores[0:len(scores)]))
    print(len(scores))
    print(pearsonr(true_scores[0:len(scores)],scores))
    plt.plot(true_scores[0:len(scores)],scores,'bo')
    plt.savefig('./test.png')

def histo():
    scores = []
    with open('./lib/dataset/aff_len13.csv','r') as f:
        for line in f:
            s = line.split(',')[1].split('\n')[0]
            if float(s) > -50:
                scores.append(float(s))
    counts, bins = np.histogram(scores)
    plt.stairs(counts, bins)
    plt.savefig('./lib/dataset/aff13.png')

def remove_dominated(aff_r,sol_r):
    scores = list(set([(aff_r[i],sol_r[i]) for i in range(len(aff_r))]))
    pareto = []
    not_pareto = []
    for i in tqdm(range(len(scores))):
        dom = True
        for j in range(len(scores)):
            if (scores[i][0] < scores[j][0] and scores[i][1] < scores[j][1]) or (scores[i][0] == scores[j][0] and scores[i][1] < scores[j][1]) or (scores[i][0] < scores[j][0] and scores[i][1] == scores[j][1]):
                not_pareto.append(scores[i])
                dom = False
                break
        if dom:
            pareto.append(scores[i])

    dist_from_pareto = []
    for i in not_pareto:
        min_dist = 10000
        for j in pareto:
            dist = np.sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2)
            min_dist = min(min_dist,dist)
        dist_from_pareto.append(min_dist)

    aff_pareto = np.array([i[0] for i in pareto])
    sol_pareto = np.array([i[1] for i in pareto])
    aff_not_pareto = np.array([i[0] for i in not_pareto])
    sol_not_pareto = np.array([i[1] for i in not_pareto])
    dist_from_pareto = np.array(dist_from_pareto)
    return aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist_from_pareto

def check_scores(oracle):
    global_weights = [9.0]
    oracle.gen_all_cdr = True
    for i in global_weights:
        print(i)
        #w = [[1.0,0.4,1.6,i,1.0,1.0],[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        w = [[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_exp_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            scores = np.load('./lib/dataset/mcmc_all_exp_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5]))
            s = oracle(seqs[-20:])
            ppost_r,sasa_r,aff_r = oracle.return_indiv_scores(seqs[-20:])
            aff_r_2 = np.load('./lib/dataset/mcmc_all_exp_aff_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5]))
            print(aff_r_2[-20:])
            print(aff_r)
            print(s)
            print(scores[-20:])
        
def compare_gflowmcmc_mcmc():
    global_weights = [12.0]
    amino_acid_dic = {'A':0, 'C':1, 'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    for i in global_weights:
        print(i)
        w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        #w = [[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            gflow_seqs = []
            mcmc_seqs = []
            with open('./lib/dataset/gflowmcmc_sol:{}_aff:{}_global:{}.txt'.format(weights[1],weights[2],weights[3]),'r') as f:
                for line in f:
                    gflow_seqs.append(line.split('\n')[0])
            with open('./lib/dataset/mcmc_all_exp_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    mcmc_seqs.append(line.split('\n')[0])
            gflow_stats = np.zeros((33,20))
            mcmc_stats = np.zeros((33,20))
            for seq in gflow_seqs:
                for i in range(len(seq)):
                    gflow_stats[i][amino_acid_dic[seq[i]]] += 1
            gflow_stats = gflow_stats / len(gflow_seqs)
            for seq in mcmc_seqs:
                for i in range(len(seq)):
                    mcmc_stats[i][amino_acid_dic[seq[i]]] += 1
            mcmc_stats = mcmc_stats / len(mcmc_seqs)
            print(weights)
            print(np.mean(np.abs(gflow_stats - mcmc_stats)))
            nov = novelty(gflow_seqs,mcmc_seqs)
            print(nov)

def plot_pareto_gflow(oracle):
    global_weights = [20.0]
    beta = [0.0]
    oracle.gen_all_cdr = True
    for gw in global_weights:
        for b in beta:
            print(gw)
            w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for weights in w:
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/gflow_sol:{}_aff:{}_global:{}_beta:{}.txt'.format(weights[1],weights[2],weights[3],weights[4]),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                seqs = [i for i in seqs if len(i) == 33]
                print(len(seqs))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
                aff_score = aff_r + weights[4] * var_r
                plt.scatter(aff_r,sasa_r,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]),marker = '+')

            plt.xlabel('Affinity Score')
            plt.ylabel('Solubility Score')
            plt.title('GFLOW ALL CDRS Global Weight:{}/Beta:{}'.format(weights[3],weights[4]))
            plt.legend()
            plt.savefig('./figures_final/pareto_gflow_all_global_weight:{}_beta:{}.png'.format(weights[3],weights[4]))
            plt.clf()


        for gw in global_weights:
            for b in beta:
                w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                #w = [[1.0,0.0,2.0,i,1.0,1.0]]
                for weights in w:
                    r = np.load('./lib/dataset/gflow_scores_sol:{}_aff:{}_gw:{}_beta:{}.npy'.format(weights[1],weights[2],weights[3],weights[4]))
                    counts, bins = np.histogram(r)
                    plt.stairs(counts, bins, label = 'aff:{},sol:{}'.format(weights[2],weights[1])) 
                plt.legend()
                plt.title('Distribution of scores for Gflow seqs')   
                plt.savefig('./figures_final/pareto_gflow_all_histo_global_weights:{}_beta:{}.png'.format(weights[3],weights[4]))
                plt.clf()

def plot_pareto_gflow_mcmc(oracle):
    global_weights = [12.0]
    oracle.gen_all_cdr = True
    for i in global_weights:
        print(i)
        w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        #w = [[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/gflowmcmc_sol:{}_aff:{}_global:{}.txt'.format(weights[1],weights[2],weights[3]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            seqs = [i for i in seqs if len(i) == 33]
            print(len(seqs))
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
            aff_score = aff_r + w[4] * var_r
            plt.scatter(aff_r,sasa_r,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]),marker = '+')

        plt.xlabel('Affinity Score')
        plt.ylabel('Solubility Score')
        plt.title('GFLOWMCMC ALL CDRS Global Weight:{}'.format(weights[3]))
        plt.legend()
        plt.savefig('./figures_final/pareto_gflowmcmc_all_global_weight:{}.png'.format(weights[3]))
        plt.clf()
    
    for i in global_weights:
        print(i)
        w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        #w = [[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            r = np.load('./lib/dataset/gflowmcmc_scores_sol:{}_aff:{}_gw:{}.npy'.format(weights[1],weights[2],weights[3]))
            counts, bins = np.histogram(r)
            plt.stairs(counts, bins, label = 'aff:{},sol:{}'.format(weights[2],weights[1])) 
        plt.legend()   
        plt.title('Distribution of scores for GflowMCMC seqs')  
        plt.savefig('./figures_final/pareto_gflowmcmc_all_histo_global_weights:{}.png'.format(weights[3]))
        plt.clf()
    
def plot_pareto_mcmc_true_aff(oracle):
    global_weights = [10.0]
    oracle.gen_all_cdr = True
    replicate = 1
    beta = [1.0,0.0,-1.0]
    for i in global_weights:
        print(i)
        for b in beta:
        #w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            #w = [[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                seqs = [i for i in seqs if len(i) == 33]
                print(len(seqs))
                sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(random.choices(seqs,k = 500))
                aff_score = aff_r + weights[4] * var_r
                plt.scatter(aff_r,sasa_r,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]),marker = '+')

            plt.xlabel('Affinity Score')
            plt.ylabel('Solubility Score')
            plt.title('TRUE AFF MCMC ALL CDRS Global Weight:{}'.format(weights[3]))
            plt.legend()
            plt.savefig('./figures_final/pareto_true_aff_mcmc_all_global_weight:{}_beta:{}.png'.format(weights[3],b))
            plt.clf()
        
        '''
        for i in global_weights:
            print(i)
            w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
            #w = [[1.0,0.0,2.0,i,1.0,1.0]]
            for weights in w:
                r = np.load('./lib/dataset/gflowmcmc_scores_sol:{}_aff:{}_gw:{}.npy'.format(weights[1],weights[2],weights[3]))
                counts, bins = np.histogram(r)
                plt.stairs(counts, bins, label = 'aff:{},sol:{}'.format(weights[2],weights[1])) 
            plt.legend()   
            plt.title('Distribution of scores for GflowMCMC seqs')  
            plt.savefig('./figures_final/pareto_gflowmcmc_all_histo_global_weights:{}.png'.format(weights[3]))
            plt.clf()
        '''

def avg_dist():
    global_weights = [10.01]
    seed = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
    for i in global_weights:
        print(i)
        w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0],[1.0,0.3,1.7,i,0.0,1.0],[1.0,0.2,1.8,i,0.0,1.0],[1.0,0.0,2.0,i,0.0,1.0]]
        #w = [[1.0,0.55,1.45,i,1.0,1.0]]
        for weights in w:
            seqs = []
            with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            d = [distance(seed,i) for i in seqs]
            counts, bins = np.histogram(d)
            plt.stairs(counts, bins, label = 'aff:{},sol:{},beta:{},mean distance:{}'.format(weights[2],weights[1],weights[4],np.mean(d)))
        plt.legend()
        plt.title('Distribution of distance to WT seq')
        plt.savefig('./figures_final/histo_dist_lim_{}.png'.format(i)) 

def remove_duplicates():
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    replicate = 3
    for gw in global_weights:
        for b in beta:
            print(gw)
            w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for weights in w:
                print(weights)
                seqs = []
                with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                score = np.load('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
                score = score[len(score) - len(seqs):]
                pair = [(seqs[i],score[i]) for i in range(len(seqs))]
                pair = set(pair)
                [seqs,score] = list(zip(*pair))
                visited = []
                v_score = []
                for i in range(len(seqs)):
                    if seqs[i] not in visited:
                        visited.append(seqs[i])
                        v_score.append(score[i])
                print(len(visited))
                print(len(v_score))
                with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_dupl_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'w') as f:
                    for s in visited:
                        f.write('{}\n'.format(s))
                np.save('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_dupl_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),np.array(v_score))

def remove_duplicates_true_aff():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    replicate = 3
    for gw in global_weights:
        for b in beta:
            print(gw)
            #w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for weights in w:
                print(weights)
                seqs = []
                with open('./lib/dataset/gen_seqs/mcmc/true_aff/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                score = np.load('./lib/dataset/gen_seqs/mcmc/true_aff/mcmc_true_aff_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
                score = score[len(score) - len(seqs):]
                pair = [(seqs[i],score[i]) for i in range(len(seqs))]
                pair = set(pair)
                [seqs,score] = list(zip(*pair))
                visited = []
                v_score = []
                for i in range(len(seqs)):
                    if seqs[i] not in visited:
                        visited.append(seqs[i])
                        v_score.append(score[i])
                print(len(visited))
                print(len(v_score))
                with open('./lib/dataset/gen_seqs/mcmc/true_aff/mcmc_true_aff_exp_lim_burn_dupl_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'w') as f:
                    for s in visited:
                        f.write('{}\n'.format(s))
                np.save('./lib/dataset/gen_seqs/mcmc/true_aff/mcmc_true_aff_exp_lim_burn_dupl_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),np.array(v_score))

def remove_duplicates_true_aff_hard():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    replicate = 3
    for gw in global_weights:
        for b in beta:
            print(gw)
            #w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for weights in w:
                print(weights)
                seqs = []
                with open('./lib/dataset/gen_seqs/mcmc/true_aff_hard/mcmc_true_aff_hard_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                score = np.load('./lib/dataset/gen_seqs/mcmc/true_aff_hard/mcmc_true_aff_hard_exp_lim_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
                score = score[len(score) - len(seqs):]
                pair = [(seqs[i],score[i]) for i in range(len(seqs))]
                pair = set(pair)
                [seqs,score] = list(zip(*pair))
                visited = []
                v_score = []
                for i in range(len(seqs)):
                    if seqs[i] not in visited:
                        visited.append(seqs[i])
                        v_score.append(score[i])
                print(len(visited))
                print(len(v_score))
                with open('./lib/dataset/gen_seqs/mcmc/true_aff_hard/mcmc_true_aff_hard_exp_lim_burn_dupl_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'w') as f:
                    for s in visited:
                        f.write('{}\n'.format(s))
                np.save('./lib/dataset/gen_seqs/mcmc/true_aff_hard/mcmc_true_aff_hard_exp_lim_burn_dupl_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),np.array(v_score))

def plot_histo_scores(oracle):
    global_weights = [9.0]
    oracle.gen_all_cdr = True
    for i in global_weights:
        print(i)
        w = [[1.0,0.3,1.7,i,1.0,1.0]]
        #w = [[1.0,0.3,1.7,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_exp_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            ppost_r,sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
            score = ppost_r + weights[3] * (weights[1]*sasa_r + weights[2] * aff_r+ weights[4]*var_r)
            counts, bins = np.histogram(var_r)
            plt.stairs(counts, bins, label = 'no lim,aff:{},sol:{}'.format(weights[2],weights[1])) 
            print('mean hum no lim {}'.format(np.mean(ppost_r.numpy())))
            print('mean sol no lim {}'.format(np.mean(sasa_r)))
            print('mean aff no lim {}'.format(np.mean(aff_r)))
            print('mean var no lim {}'.format(np.mean(var_r)))
            print('mean score{}'.format(np.mean(score.numpy())))
            with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            ppost_r_lim,sasa_r_lim,aff_r_lim,var_r_lim = oracle.return_indiv_scores(random.choices(seqs,k = 500))
            score_lim = ppost_r_lim + weights[3] * (weights[1]*sasa_r_lim + weights[2] * aff_r_lim+ weights[4]*var_r_lim)
            counts, bins = np.histogram(var_r_lim)
            plt.stairs(counts, bins, label = 'lim,aff:{},sol:{}'.format(weights[2],weights[1]))
            print('mean hum {}'.format(np.mean(ppost_r_lim.numpy())))
            print('mean sol {}'.format(np.mean(sasa_r_lim)))
            print('mean aff {}'.format(np.mean(aff_r_lim)))
            print('mean var {}'.format(np.mean(var_r_lim)))
            print('mean score{}'.format(np.mean(score_lim.numpy())))
    plt.legend()   
    plt.savefig('./figures_final/pareto_var_score_histo:{}.png'.format(weights[3]))
    plt.clf()

def plot_pareto(oracle):
    global_weights = [30.0]
    oracle.gen_all_cdr = True
    enc_type = 'esm'
    for i in global_weights:
        print(i)
        #w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        w = [[1.0,0.6,1.4,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_exp_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(seqs,k = 1500))
            aff_score = aff_r + w[4] * var_r
            plt.scatter(aff_score,sasa_r,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]),marker = '+')

        plt.xlabel('Affinity Score')
        plt.ylabel('Solubility Score')
        plt.title('MCMC ALL CDRS Global Weight:{}'.format(weights[3]))
        plt.legend()
        plt.savefig('./figures_final/pareto_mcmc_all_global_weight2:{}.png'.format(weights[3]))
        plt.clf()
    '''
    for i in global_weights:
        print(i)
        #w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        w = [[1.0,0.55,1.55,i,1.0,1.0]]
        for weights in w:
            r = np.load('./lib/dataset/mcmc_all_exp_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5]))
            counts, bins = np.histogram(r)
            plt.stairs(counts, bins, label = 'aff:{},sol:{}'.format(weights[2],weights[1])) 
        plt.legend()
        plt.title('Distribution of scores for MCMC seqs')
        plt.savefig('./figures_final/pareto_mcmc_all_histo_global_weights:{}.png'.format(weights[3]))
        plt.clf()

        r = r.reshape(60000,8)
        plt.plot(r)
        plt.savefig('./figures_final/pareto_mcmc_all_chains_global_weights:{}.png'.format(weights[3]))
    '''

def plot_pareto_lim(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0]
    oracle.gen_all_cdr = True
    replicate = 1
    for i in global_weights:
        for b in beta:
            print(i)
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            #w = [[1.0,0.15,0.85,i,0.0,1.0],[1.0,0.1,0.9,i,0.0,1.0],[1.0,0.0,1.0,i,0.0,1.0]]
            #w = [[1.0,0.0,1.0,i,1.0,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]),marker = '+')

            plt.xlabel('Affinity Score')
            plt.ylabel('Solubility Score')
            plt.title('MCMC ALL CDRS Global Weight:{}'.format(weights[3]))
            plt.legend()
            plt.savefig('./pareto_final/pareto_mcmc_all_lim_global_weight:{}_beta:{}_rep:{}.png'.format(weights[3],weights[4],replicate))
            plt.clf()
        '''
        for i in global_weights:
            print(i)
            w = [[1.0,0.4,1.6,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
            #w = [[1.0,0.0,2.0,i,1.0,1.0]]
            for weights in w:
                r = np.load('./lib/dataset/mcmc_all_exp_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5]))
                counts, bins = np.histogram(r)
                plt.stairs(counts, bins, label = 'aff:{},sol:{}'.format(weights[2],weights[1])) 
            plt.legend()   
            plt.savefig('./figures_final/pareto_mcmc_all_histo_global_weights:{}.png'.format(weights[3]))
            plt.clf()
        '''

def plot_pareto_all(oracle):
    global_weights = [1.0,3.0,5.0]
    oracle.gen_all_cdr = True
    for i in global_weights:
        print(i)
        w = [[1.0,0.4,1.6,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            ppost_r,sasa_r,aff_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
            plt.scatter(aff_r,sasa_r,label = 'SOL:{} AFF:{} TOTAL:{}'.format(weights[1],weights[2],weights[3]))

    plt.xlabel('Affinity Score')
    plt.ylabel('Solubility Score')
    plt.title('MCMC ALL CDRS'.format(weights[3]))
    plt.legend()
    plt.savefig('./figures_final/pareto_mcmc_all_combine.png')

def plot_pareto_all_sample(oracle):
    #global_weights = [20.0,25.0,30.0]
    global_weights = [20.0]
    beta = [1.0,0.0,-1.0]
    #beta = [0.0]
    replicate = 1
    oracle.gen_all_cdr = True
    for i in global_weights:
        for b in beta:
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,std_r = oracle.return_indiv_scores(sampled)
                sample_aff = torch.normal(torch.Tensor(aff_r),torch.Tensor(std_r)).cpu().numpy()
                pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(sample_aff,sasa_r)
                plt.scatter(not_pareto_aff,not_pareto_sol,color = 'g',label = 'MCMC generated seqs')
                plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
                idx = np.argsort(pareto_aff).tolist()
                pareto_aff = [pareto_aff[i] for i in idx]
                pareto_sol = [pareto_sol[i] for i in idx]
                plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front')
                plt.xlabel('affinity score')
                plt.ylabel('solubility score')
                plt.legend()
                plt.title('MCMC Covid Sample')
                plt.savefig('./pareto_final/pareto_mcmc_lim_sample_sol:{}_gw:{}_beta:{}_rep:{}.png'.format(weights[1],i,b,replicate))
                plt.clf()

def plot_pareto_all_color(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [1.0,0.0,-1.0]
    replicate = 1
    oracle.gen_all_cdr = True
    add_initial = False
    for i in global_weights:
        for b in beta:
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            all_aff = np.array([])
            all_sol = np.array([])
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(sampled)
                aff_r = aff_r + b * var_r
                all_aff = np.concatenate((all_aff,aff_r))
                all_sol = np.concatenate((all_sol,sasa_r))
            pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(all_aff,all_sol)
            min_d = np.min(dist)
            max_d = np.max(dist)
            dist = [1 - (i - min_d)/(max_d - min_d) for i in dist]
            plt.scatter(not_pareto_aff,not_pareto_sol,alpha = dist,color = 'r', marker = '+',label = 'MCMC generated seqs')
            plt.scatter(pareto_aff,pareto_sol,color = 'g',marker = '+',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front')
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                            for line in f:
                                cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            plt.xlabel('affinity score')
            plt.ylabel('solubility score')
            plt.legend()
            plt.title('MCMC Covid')
            plt.savefig('./pareto_final/pareto_mcmc_lim_all_combine_color_gw:{}_beta:{}_initial:{}_rep:{}.png'.format(i,b,add_initial,replicate))
            plt.clf()

def plot_pareto_compare_all(oracle):
    global_weights = [20.0]
    beta = [0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    for i in global_weights:
        for b in beta:
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            all_aff_mcmc = np.array([])
            all_sol_mcmc = np.array([])
            all_aff_gflow = np.array([])
            all_sol_gflow = np.array([])
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(sampled)
                aff_r = aff_r + b * var_r
                all_aff_mcmc = np.concatenate((all_aff_mcmc,aff_r))
                all_sol_mcmc = np.concatenate((all_sol_mcmc,sasa_r))

                seqs = []
                with open('./lib/dataset/gflow_sol:{}_aff:{}_global:{}_beta:{}.txt'.format(weights[1],weights[2],weights[3],weights[4])) as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(sampled)
                aff_r = aff_r + b * var_r
                all_aff_gflow = np.concatenate((all_aff_gflow,aff_r))
                all_sol_gflow = np.concatenate((all_sol_gflow,sasa_r))
            #pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(all_aff,all_sol)
            #min_d = np.min(dist)
            #max_d = np.max(dist)
            #dist = [1 - (i - min_d)/(max_d - min_d) for i in dist]
            #plt.scatter(not_pareto_aff,not_pareto_sol,alpha = dist,color = 'r', marker = '+',label = 'MCMC generated seqs')
            #plt.scatter(pareto_aff,pareto_sol,color = 'g',marker = '+',label = 'pareto front seqs')
            plt.scatter(all_aff_mcmc,all_sol_mcmc,color = 'r',marker = '+',label = 'mcmc seqs')
            plt.scatter(all_aff_gflow,all_sol_gflow,color = 'g',marker = '+',label = 'gflow seqs')
            #idx = np.argsort(pareto_aff).tolist()
            #pareto_aff = [pareto_aff[i] for i in idx]
            #pareto_sol = [pareto_sol[i] for i in idx]
            #plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front')
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                            for line in f:
                                cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            plt.xlabel('affinity score')
            plt.ylabel('solubility score')
            plt.legend()
            plt.title('MCMC Covid')
            plt.savefig('./figures_final/pareto_mcmc_gflow_compare_all_gw:{}_beta_{}_initial:{}.png'.format(i,b,add_initial))
            plt.clf()

def get_color_plot(aff,sol):
    aff = np.array(aff)
    sol = np.array(sol)
    xy = np.vstack([aff,sol])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = aff[idx], sol[idx], z[idx]
    return x,y,z

def plot_pareto_all_color_density(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    for i in global_weights:
        for b in beta:
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            all_aff = np.array([])
            all_sol = np.array([])
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(sampled)
                aff_r = aff_r + b * var_r
                all_aff = np.concatenate((all_aff,aff_r))
                all_sol = np.concatenate((all_sol,sasa_r))
            pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(all_aff,all_sol)
            min_d = np.min(dist)
            max_d = np.max(dist)
            dist = [1 - (i - min_d)/(max_d - min_d) for i in dist]
            x,y,z = get_color_plot(not_pareto_aff,not_pareto_sol)

            plt.scatter(x,y,alpha = dist,c = z,label = 'MCMC generated seqs')
            plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                            for line in f:
                                cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            plt.xlabel('affinity score')
            plt.ylabel('solubility score')
            plt.legend()
            plt.title('MCMC Covid/GW:{}/beta:{}'.format(i,b))
            plt.savefig('./pareto_final/pareto_mcmc_lim_all_combine_color_density_gw:{}_beta:{}_initial:{}_rep:{}.png'.format(i,b,add_initial,replicate))
            plt.clf()

def plot_pareto_all_gw_density(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 2
    for b in beta:
        all_aff = np.array([])
        all_sol = np.array([])
        for i in global_weights:
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(sampled)
                aff_r = aff_r + b * var_r
                all_aff = np.concatenate((all_aff,aff_r))
                all_sol = np.concatenate((all_sol,sasa_r))
        pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(all_aff,all_sol)
        min_d = np.min(dist)
        max_d = np.max(dist)
        dist = [1 - (i - min_d)/(max_d - min_d) for i in dist]
        x,y,z = get_color_plot(not_pareto_aff,not_pareto_sol)

        plt.scatter(x,y,alpha = dist,c = z,label = 'MCMC generated seqs')
        plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        plt.xlabel('affinity score')
        plt.ylabel('solubility score')
        plt.legend()
        plt.title('MCMC Covid/beta:{}'.format(b))
        plt.savefig('./pareto_final/pareto_mcmc_all_lim_combinegw_density_beta:{}_initial:{}_rep:{}.png'.format(b,add_initial,replicate))
        plt.clf()

def plot_pareto_true_aff_gw_density(oracle):
    global_weights = [10.0]
    beta = [1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    for b in beta:
        all_aff = np.array([])
        all_sol = np.array([])
        for i in global_weights:
            #w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(sampled)
                aff_r = aff_r + b * var_r
                all_aff = np.concatenate((all_aff,aff_r))
                all_sol = np.concatenate((all_sol,sasa_r))
        pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(all_aff,all_sol)
        min_d = np.min(dist)
        max_d = np.max(dist)
        dist = [1 - (i - min_d)/(max_d - min_d) for i in dist]
        x,y,z = get_color_plot(not_pareto_aff,not_pareto_sol)

        plt.scatter(x,y,alpha = dist,c = z,label = 'MCMC generated seqs')
        plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        if add_initial:
            cdrlist = []
            with open('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        plt.xlabel('affinity score = mu + {} x sigma'.format(b))
        plt.ylabel('solubility score')
        plt.legend()
        plt.title('MCMC True Aff Covid/beta:{}'.format(b))
        plt.savefig('./pareto_final/pareto_mcmc_true_aff_lim_combinegw_density_beta:{}_initial:{}_rep:{}.png'.format(b,add_initial,replicate))
        plt.clf()
        print('done')

def plot_pareto_true_aff_gw_density_2(oracle):
    cdrlist = []
    with open('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts.txt','r') as f:
        for line in f:
            cdrlist.append(line.split('\n')[0])
    global_weights = [10.0]
    beta = [1.0,-1.0,0.0]
    budget = 500
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    for b in beta:
        seqs = []
        for i in global_weights:
            #w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
        seqs = list(set(seqs))
        print(len(seqs))
        sasa_r = oracle.get_sasa_score(seqs)
        aff_r = 12 - np.array(oracle.true_aff_ora.score_without_noise_simple(seqs))

        pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(aff_r,sasa_r)
        sampled_idx = random.sample(range(len(not_pareto_aff)),k = min(len(not_pareto_aff),budget))
        not_pareto_aff = not_pareto_aff[sampled_idx]
        not_pareto_sol = not_pareto_sol[sampled_idx]
        dist = dist[sampled_idx]

        min_d = np.min(dist)
        max_d = np.max(dist)
        dist = [1 - (i - min_d)/(max_d - min_d) for i in dist]
        x,y,z = get_color_plot(not_pareto_aff,not_pareto_sol)

        plt.scatter(x,y,alpha = dist,c = z,label = 'MCMC generated seqs')
        plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        
        if add_initial:
            sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        plt.xlabel('true_affinity')
        plt.ylabel('solubility score')
        plt.legend()
        plt.title('MCMC True Aff Covid/beta:{}'.format(b))
        plt.savefig('./pareto_final/pareto_mcmc_true_aff_lim_combinegw_density_beta:{}_initial:{}_rep:{}_trueaffinity_budget:{}.png'.format(b,add_initial,replicate,budget))
        plt.clf()
        print('done')

def plot_pareto_true_aff_pareto_sampled_seqs(oracle):
    cdrlist = []
    with open('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts.txt','r') as f:
        for line in f:
            cdrlist.append(line.split('\n')[0])
    global_weights = [10.0]
    beta = [1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    replicate = 1
    add_initial = False
    budget = 500
    for b in beta:
        sampled_seqs = []
        seqs = []
        for i in global_weights:
            #w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled_seqs += random.choices(seqs,k = budget)
        sampled_seqs = list(set(sampled_seqs))
        sasa_r = oracle.get_sasa_score(sampled_seqs)
        aff_r = 12 - np.array(oracle.true_aff_ora.score_without_noise_simple(sampled_seqs))

        pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(aff_r,sasa_r)
        plt.scatter(pareto_aff,pareto_sol,label = 'beta = {}'.format(b))
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol,label = 'pareto front')
        
        if add_initial:
            sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

    plt.xlabel('true_affinity')
    plt.ylabel('solubility score')
    plt.legend()
    plt.title('MCMC True Aff Pareto All Seqs')
    plt.savefig('./pareto_final/pareto_mcmc_true_aff_sampled_seqs_initial:{}_rep:{}_trueaffinity_budget:{}.png'.format(add_initial,replicate,budget))
    plt.clf()

def plot_pareto_true_aff_pareto_all_seqs(oracle):
    cdrlist = []
    with open('/root/workdir/ABGen/BioSeq-GFN-AL/lib/true_aff/muts.txt','r') as f:
        for line in f:
            cdrlist.append(line.split('\n')[0])
    global_weights = [10.0]
    beta = [1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    replicate = 1
    add_initial = False
    for b in beta:
        seqs = []
        for i in global_weights:
            #w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
        seqs = list(set(seqs))
        print(len(seqs))
        sasa_r = oracle.get_sasa_score(seqs)
        aff_r = 12 - np.array(oracle.true_aff_ora.score_without_noise_simple(seqs))

        pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(aff_r,sasa_r)
        plt.scatter(pareto_aff,pareto_sol,label = 'beta = {}'.format(b))
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol,label = 'pareto front')
        
        if add_initial:
            sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

    plt.xlabel('true_affinity')
    plt.ylabel('solubility score')
    plt.legend()
    plt.title('MCMC True Aff Pareto All Seqs')
    plt.savefig('./pareto_final/pareto_mcmc_true_aff_all_seqs_initial:{}_rep:{}_trueaffinity.png'.format(add_initial,replicate))
    plt.clf()

def plot_pareto_all_fronts(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    for i in global_weights:
        for b in beta:
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            all_aff = np.array([])
            all_sol = np.array([])
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                sampled = list(set(random.choices(seqs,k = 500)))
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(sampled)
                aff_r = aff_r + b * var_r
                all_aff = np.concatenate((all_aff,aff_r))
                all_sol = np.concatenate((all_sol,sasa_r))
            pareto_aff,pareto_sol,not_pareto_aff,not_pareto_sol,dist = remove_dominated(all_aff,all_sol)
            plt.scatter(pareto_aff,pareto_sol,label = 'pareto front seqs/gw:{}/beta:{}'.format(i,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front/gw:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                            for line in f:
                                cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            plt.xlabel('affinity score')
            plt.ylabel('solubility score')
            plt.legend()
            plt.title('MCMC Covid/GW:{}/beta:{}'.format(i,b))
            plt.savefig('./figures_final/all_pareto_fronts.png')
            plt.clf()

def plot_initial_muts(oracle):
    cdrlist = []
    beta = [1.0,0.0,-1.0]
    for b in beta:
        w = [1.0,1.0,1.0,1.0,b,1.0]
        oracle.set_weights(w)
        with open('./lib/Covid/data/cdrlist.txt','r') as f:
            for line in f:
                cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k=50))
            aff_r = aff_r + b * var_r
            print(sasa_r)
            print(aff_r)
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
        plt.xlabel('affinity score')
        plt.ylabel('solubility score')
        plt.legend()
        plt.title('MCMC Covid')
        plt.savefig('./figures_final/pareto_initial_muts_beta:{}.png'.format(b))
        plt.clf()
    

def plot_histo_pareto(oracle):
    global_weights = [1.0,5.0,10.0]
    for i in global_weights:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        print(i)
        w = [[1.0,2.0,0.0,i,1.0,1.0],[1.0,1.0,1.0,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            aff,var = oracle.get_aff_score_gp_2(random.choices(seqs,k = 50))
            counts, bins = np.histogram(aff)
            ax1.stairs(counts, bins,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]))
            counts, bins = np.histogram(var)
            ax2.stairs(counts, bins,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]))

        ax1.set(xlabel = 'affinity score', ylabel = 'density')
        ax2.set(xlabel = 'variance score', ylabel = 'density')
        fig.suptitle('Global Weight:{}'.format(weights[3]))
        fig.savefig('./figures_final/histo_mcmc_all_global_weight:{}.png'.format(weights[3]))
        fig.clf()

def plot_pareto_ppost(oracle):
    global_weights = [1.0,5.0,10.0]
    for i in global_weights:
        print(i)
        w = [[1.0,2.0,0.0,i,1.0,1.0],[1.0,1.5,0.5,i,1.0,1.0],[1.0,1.0,1.0,i,1.0,1.0],[1.0,0.5,1.5,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            ppost_r,sasa_r,aff_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
            plt.scatter(ppost_r,aff_r,label = 'PPOST:{} AFF:{} BETA:{}'.format(1.0,weights[2],weights[4]))

        plt.xlabel('ppost score')
        plt.ylabel('affinity score')
        plt.title('MCMC ALL CDRS Global Weight:{}'.format(weights[3]))
        plt.legend()
        plt.savefig('./figures_final/pareto_mcmc_all_ppost_global_weight:{}.png'.format(weights[3]))
        plt.clf()

def plot_pareto_ppost_sol(oracle):
    global_weights = [1.0,5.0,10.0]
    for i in global_weights:
        print(i)
        w = [[1.0,2.0,0.0,i,1.0,1.0],[1.0,1.5,0.5,i,1.0,1.0],[1.0,1.0,1.0,i,1.0,1.0],[1.0,0.5,1.5,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/mcmc_all_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            ppost_r,sasa_r,aff_r = oracle.return_indiv_scores(random.choices(seqs,k = 500))
            plt.scatter(ppost_r,sasa_r,label = 'PPOST:{} SOL:{} BETA:{}'.format(1.0,weights[2],weights[3]))

        plt.xlabel('ppost score')
        plt.ylabel('solubility score')
        plt.title('MCMC ALL CDRS Global Weight:{}'.format(weights[3]))
        plt.legend()
        plt.savefig('./figures_final/pareto_mcmc_all_ppost_sol_global_weight:{}.png'.format(weights[3]))
        plt.clf()

def plot_pareto_antBO(oracle):
    sasa_min = [2.0,3.0]
    ppost_min = [-30,-35,-40]
    for s in sasa_min:
        for p in ppost_min:
            seqs = []
            with open('./lib/true_aff/antBO_sol_min:{}_ppost_min:{}.txt'.format(s,p),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            print(seqs[0:10])
            sasa_r,aff_r,ppost_r = oracle.score_true_aff_seqs_indiv(random.choices(seqs,k = 500))
            plt.scatter(aff_r,sasa_r,label = 'PPOST_MIN:{} SASA_MIN:{}'.format(p,s))
    plt.xlabel('aff score')
    plt.ylabel('solubility score')
    plt.title('antBO')
    plt.legend()
    plt.savefig('./figures_final/pareto_antBO_aff_sol.png')
    plt.clf()

def compare_true_aff_var(oracle):
    global_weights = [3.0,5.0,10.0,20.0]
    var = 0.1
    #oracle.update_true_aff_gp(var)
    for i in global_weights:
        w = [[1.0,0.1,1.9,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/gen_seqs/sim_anneal_true_aff_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            mean,var = oracle.get_mean_var_true_aff(random.choices(seqs,k = 4000))
            plt.scatter(mean,var,label = 'Global:{}'.format(i))

    plt.xlabel('affinity')
    plt.ylabel('variance')
    plt.title('Mean variance comp')
    plt.legend()
    plt.savefig('./figures_final/compare_mean_var_3.png')
    plt.clf()

def compare_true_aff_var_mean_ori(oracle):
    var = 0.1
    seed = oracle.true_aff_seed
    dist = []
    oracle.update_true_aff_gp(var)
    seqs = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    seqs = random.choices(seqs,k=4000)
    dist = [distance(i,seed) for i in seqs]
    mean,var = oracle.get_mean_var_true_aff(seqs)
    for i in range(6):
        mean_k = [mean[k] for k in range(len(mean)) if dist[k] == i]
        var_k = [var[k] for k in range(len(var)) if dist[k] == i]
        print(len(mean_k))
        print(len(var_k))
        plt.scatter(mean_k,var_k,label = 'distance {}'.format(i))

    plt.xlabel('affinity')
    plt.ylabel('variance')
    plt.title('Mean variance comp')
    plt.legend()
    plt.savefig('./figures_final/compare_mean_var_original_muts_0.1.png')
    plt.clf()

def plot_sim_anneal(oracle):
    global_weights = [5.0,8.0,9.0,10.0]
    for i in global_weights:
        w = [[1.0,0.4,1.6,i,1.0,1.0],[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.1,1.9,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/gen_seqs/sim_anneal_true_aff_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            sasa_r,aff_r,ppost_r = oracle.score_true_aff_seqs_indiv(random.choices(seqs,k = 500))
            print(weights)
            print(np.mean(aff_r))
            plt.scatter(aff_r,sasa_r,label = 'SOL:{} AFF:{} BETA:{}'.format(weights[1],weights[2],weights[4]))

    plt.xlabel('affinity score')
    plt.ylabel('solubility score')
    plt.title('MCMC TRUE AFF')
    plt.legend()
    plt.savefig('./figures_final/pareto_sim_anneal_true_aff.png')
    plt.clf()

def plot_sim_anneal_2(oracle):
    global_weights = [3.0,5.0,10.0,20.0]
    min_SASA = -1
    max_SASA = 6.0
    min_aff = 1.9
    max_aff = 4.0
    for i in global_weights:
        w = [[1.0,0.4,1.6,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.1,1.9,i,1.0,1.0],[1.0,0.3,1.7,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/gen_seqs/sim_anneal_true_aff_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            sasa_r,aff_r,ppost_r = oracle.score_true_aff_seqs_indiv(random.choices(seqs,k = 500))
            sasa_r_c = (sasa_r - min_SASA)/(max_SASA - min_SASA)
            aff_r_c = (aff_r - min_aff)/(max_aff - min_aff)
            sasa_r_c = np.clip(sasa_r_c,0,1)
            aff_r_c = np.clip(aff_r_c,0,1)
            color = [[sasa_r_c[i],0,aff_r_c[i]] for i in range(len(sasa_r_c))]
            print(color[0:5])
            print(weights)
            print(np.mean(aff_r))
            plt.scatter(aff_r,sasa_r, c = color)

    plt.xlabel('affinity score')
    plt.ylabel('solubility score')
    plt.title('MCMC TRUE AFF')
    plt.savefig('./figures_final/pareto_sim_anneal_true_aff_2.png')
    plt.clf()

def find_best_dev_mut_true_aff(oracle):
    seqs = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    random_seqs = random.choices(seqs,k = 4000)
    sasa_r,aff_r,ppost_r = oracle.score_true_aff_seqs_indiv(random_seqs)
    pareto_seqs = []
    for i in range(500):
        dom = False
        for j in range(500):
            if sasa_r[i] < sasa_r[j] and ppost_r[i] < ppost_r[j]:
                dom = True
                break
        if not dom:
            pareto_seqs.append(random_seqs[i])
    with open('./lib/true_aff/pareto_seqs_dev.txt','w') as f:
        for s in pareto_seqs:
            f.write(s+'\n')
    sasa_r,aff_r,ppost_r = oracle.score_true_aff_seqs_indiv(pareto_seqs)
    plt.scatter(sasa_r,ppost_r)
    plt.savefig('./lib/true_aff/figures/pareto_dev.png')


def diversity():
    seqs = []
    with open('./lib/dataset/gflow_ppost_13.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    div = 0
    for i in seqs:
        for j in seqs:
            div += distance(i,j)
    print(div/len(seqs))

def process_length():
    seqs = []
    with open('./lib/dataset/aff_sol_glow_0.0_3.0.txt','r') as f:
        for line in f:
            seq = line.split('\n')[0]
            if len(seq) == 13:
                seqs.append(seq)
    with open('./lib/dataset/aff_sol_glow_0.0_3.0.txt','w') as f:
        for seq in seqs:
            f.write('{}\n'.format(seq))

def compute_mean():
    data = []
    with open('pareto.txt','r') as f:
        for line in f:
            a = line.split(',')
            data.append((float(a[0]),float(a[1]),int(a[2])))
    mean_sol = 0
    mean_aff = 0
    count = 0
    for i in data:
        if i[2] == 6:
            mean_sol += i[1]
            mean_aff += i[0]
            count += 1
    print(mean_sol/count)
    print(mean_aff/count)


def plot_pareto_dom():
    data = []
    with open('pareto.txt','r') as f:
        for line in f:
            a = line.split(',')
            data.append((float(a[0]),float(a[1]),int(a[2])))
    #dom = remove_dominated(data)
    for i in data:
        if i[2] == 0:
            plt.plot(i[0],i[1],'b+')
        elif i[2] == 1:
            plt.plot(i[0],i[1],'r+')
        else:
            plt.plot(i[0],i[1],'g+')
    plt.xlabel('affinity score')
    plt.ylabel('solubility score')
    plt.savefig('./figures/pareto.png')

class cov_args:
    def __init__(self):
        self.task = 'random'
        self.gen_max_len = 37
        self.num_tokens = 22
        self.task_train = 'nmasked'
        self.model = 'cnn'

def get_covid_seqs():
    cdrlist = []
    with open('./lib/Covid/data/cdrlist.txt','r') as f:
        for line in f:
            cdrlist.append(line.split('\n')[0])
    return cdrlist

def get_true_aff_seqs():
    true_aff_seqs = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            true_aff_seqs.append(line.split('\n')[0])
    return true_aff_seqs

def get_IGLM_seqs():
    seqs = []
    with open('./lib/dataset/IGLM_seqs.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs

def covid_histo(oracle):
    seqs = get_covid_seqs()
    humaness_scores = oracle.get_ppost_score(seqs).numpy()
    solubility_scores = oracle.get_sasa_score(seqs)
    counts,bins = np.histogram(humaness_scores)
    plt.stairs(counts,bins)
    plt.savefig('./pareto_final/histo_ppost_initial_covid.png')
    plt.clf()
    counts,bins = np.histogram(solubility_scores)
    plt.stairs(counts,bins)
    plt.savefig('./pareto_final/histo_sol_initial_covid.png')
    plt.clf()
       
def compare_thera_IGLM_sol(oracle):
    thera_seqs = get_thera_seqs()
    IGLM_seqs = get_IGLM_seqs_2()
    IGG_seqs = get_IGG_seqs()
    len_IGLM = [len(i) for i in IGLM_seqs]
    len_thera = [len(i) for i in thera_seqs]
    len_IGG = [len(i) for i in IGG_seqs]
    print(np.mean(len_IGLM))
    print(np.mean(len_thera))
    print(np.mean(len_IGG))
    
    thera_sol = oracle.get_full_sasa_score(thera_seqs)
    IGLM_sol = oracle.get_full_sasa_score(IGLM_seqs)
    IGG_sol = oracle.get_full_sasa_score(IGG_seqs)
    print(np.mean(thera_sol))
    print(np.mean(IGLM_sol))
    print(np.mean(IGG_sol))
    counts, bins = np.histogram(thera_sol, bins = range(-7, 7), density =  True)
    plt.stairs(counts, bins, label = 'therapeutic antibodies')
    counts, bins = np.histogram(IGLM_sol, bins = range(-7, 7), density =  True)
    plt.stairs(counts, bins, label = 'IGLM antibodies')
    counts, bins = np.histogram(IGG_sol, bins = range(-7, 7), density =  True)
    plt.stairs(counts, bins, label = 'IGG antibodies')
    plt.xlabel('solubility score')
    plt.ylabel('density')
    plt.legend()
    plt.savefig('./figures/compare_thera_IGLM_IGG_sol.png')


def compare_thera_IGLM_ppost(oracle):
    thera_seqs = get_thera_seqs()
    IGLM_seqs = get_IGLM_seqs_2()
    IGG_seqs = get_IGG_seqs()
    len_IGLM = [len(i) for i in IGLM_seqs]
    len_thera = [len(i) for i in thera_seqs]
    len_IGG = [len(i) for i in IGG_seqs]
    print(np.mean(len_IGLM))
    print(np.mean(len_thera))
    print(np.mean(len_IGG))
    
    thera_sol = oracle.get_ppost_full_score(thera_seqs).numpy()
    IGLM_sol = oracle.get_ppost_full_score(IGLM_seqs).numpy()
    IGG_sol = oracle.get_ppost_full_score(IGG_seqs).numpy()
    print(np.mean(thera_sol))
    print(np.mean(IGLM_sol))
    print(np.mean(IGG_sol))
    counts, bins = np.histogram(thera_sol, bins = range(0, 6), density =  True)
    plt.stairs(counts, bins, label = 'therapeutic antibodies')
    counts, bins = np.histogram(IGLM_sol, bins = range(0, 6), density =  True)
    plt.stairs(counts, bins, label = 'IGLM antibodies')
    counts, bins = np.histogram(IGG_sol, bins = range(0, 6), density =  True)
    plt.stairs(counts, bins, label = 'IGG antibodies')
    plt.xlabel('perplexity')
    plt.ylabel('density')
    plt.legend()
    plt.savefig('./figures_final/compare_thera_IGLM_IGG_ppost.png')

def true_aff_histo(oracle):
    seqs = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    oracle.gen_all_cdr = False
    full_seqs = ['CAK'+i+'W' for i in seqs]
    batch_size = 50
    n_batch = int(len(seqs)/batch_size)
    sasa_score = []
    for i in tqdm(range(n_batch)):
        sasa_score += oracle.get_ppost_score(full_seqs[batch_size*i:batch_size*(i+1)]).tolist()
    counts, bins = np.histogram(sasa_score,bins = range(-50,-25), density =  True)
    plt.stairs(counts, bins)
    plt.xlabel('humaness')
    plt.ylabel('density')
    plt.savefig('./lib/true_aff/figures/ppost_histo.png')

def diversity(seqs):
    div = []
    for i in range(len(seqs)-1):
        for j in range(i+1,len(seqs)):
            div.append(distance(seqs[i],seqs[j]))
    return np.mean(div)

def performance(seqs,oracle):
    trueaff_score = oracle.true_aff_ora.score_without_noise_simple(seqs)
    #sort_score = np.sort(trueaff_score)[-100:]
    return np.mean(trueaff_score)
    
def novelty(init,gen):
    nov = []
    for i in tqdm(range(len(gen))):
        min_d = 33
        for j in init:
            dist = distance(gen[i],j)
            if dist < min_d:
                min_d = dist
        nov.append(min_d)
    return np.mean(nov)

def analyze_covid(oracle):
    oracle.gen_all_cdr = True
    initial_muts = []
    with open('./lib/Covid/data/cdrlist.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])

    global_weights = [20.0,25.0,30.0]
    beta = [1.0,0.0,-1.0]
    '''
    div = diversity(initial_muts)
    nov = novelty(initial_muts,initial_muts)
    instability,hydro,charge = oracle.get_dev_score(initial_muts)
    print('div:{},nov:{}'.format(div,nov))
    print('insta:{},hydro:{},charge:{}'.format(np.mean(instability),np.mean(hydro),np.mean(charge)))
    '''
    for i in global_weights:
        for b in beta:
            print(i)
            w = [[1.0,0.15,0.85,i,b,1.0],[1.0,0.125,0.875,i,b,1.0],[1.0,0.1,0.9,i,b,1.0],[1.0,0.05,0.95,i,b,1.0],[1.0,0.0,1.0,i,b,1.0]]
            for weights in w:
                print(weights)
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                seqs = random.choices(seqs,k = 1500) 
                div = diversity(seqs)
                nov = novelty(initial_muts,seqs)
                instability,hydro,charge = oracle.get_dev_score(seqs)
                print('div:{},nov:{}'.format(div,nov))
                print('insta:{},hydro:{},charge:{}'.format(np.mean(instability),np.mean(hydro),np.mean(charge)))

def analyze_covid_gflowmcmc(oracle):
    oracle.gen_all_cdr = True
    initial_muts = []
    with open('./lib/Covid/data/cdrlist.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])
    initial_muts = random.choices(initial_muts,k = 4000)

    global_weights = [12.0]
    '''
    div = diversity(initial_muts)
    nov = novelty(initial_muts,initial_muts)
    instability,hydro,charge = oracle.get_dev_score(initial_muts)
    print('div:{},nov:{}'.format(div,nov))
    print('insta:{},hydro:{},charge:{}'.format(np.mean(instability),np.mean(hydro),np.mean(charge)))
    '''
    for i in global_weights:
        print(i)
        w = [[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        for weights in w:
            oracle.set_weights(weights)
            seqs = []
            with open('./lib/dataset/gflowmcmc_sol:{}_aff:{}_global:{}.txt'.format(weights[1],weights[2],weights[3]),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            seqs = random.choices(seqs,k = 230) 
            div = diversity(seqs)
            nov = novelty(initial_muts,seqs)
            instability,hydro,charge = oracle.get_dev_score(seqs)
            print('div:{},nov:{}'.format(div,nov))
            print('insta:{},hydro:{},charge:{}'.format(np.mean(instability),np.mean(hydro),np.mean(charge)))

def analyze_true_aff(oracle):
    initial_muts = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])
    replicate = 1
    global_weights = [10.0]
    beta = [1.0,0.0,-1.0]
    fig, axs = plt.subplots(4,figsize = (8,12))
    solus = [[],[]]
    perfs = [[],[]]
    divs = [[],[]]
    novs = [[],[]]
    budget = 100
    for gw in global_weights:
        for b in beta:
            w = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            idx = 0
            for weights in w:
                oracle.set_weights(weights)
                seqs = []
                with open('./lib/dataset/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
                    for line in f:
                        seqs.append(line.split('\n')[0])
                seqs = list(set(random.choices(seqs,k = budget)))
                seqs = [s for s in seqs if s not in initial_muts] 
                div = diversity(seqs)
                perf = performance(seqs,oracle)
                nov = novelty(initial_muts,seqs)
                sasa_r,aff_r,var_r = oracle.score_true_aff_seqs_indiv(seqs)
                perfs[idx].append(perf)
                divs[idx].append(div)
                novs[idx].append(nov)
                solus[idx].append(np.mean(sasa_r))
                idx += 1
    axs[0].plot(beta,perfs[0],label = 'sol weight 0.15')
    axs[0].plot(beta,perfs[1],label = 'sol weight 0.0')
    axs[1].plot(beta,divs[0],label = 'sol weight 0.15')
    axs[1].plot(beta,divs[1],label = 'sol weight 0.0')
    axs[2].plot(beta,novs[0],label = 'sol weight 0.15')
    axs[2].plot(beta,novs[1],label = 'sol weight 0.0')
    axs[3].plot(beta,solus[0],label = 'sol weight 0.15')
    axs[3].plot(beta,solus[1],label = 'sol weight 0.0')
    axs[0].set_xlabel('beta')
    axs[1].set_xlabel('beta')
    axs[2].set_xlabel('beta')
    axs[3].set_xlabel('beta')
    axs[0].set_ylabel('performance')
    axs[1].set_ylabel('diversity')
    axs[2].set_ylabel('novelty')
    axs[3].set_ylabel('solubility')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/analyze_true_aff_budget:{}.png'.format(budget))



def analyze_ant_BO_true_aff(oracle):
    initial_muts = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])
    initial_muts = random.choices(initial_muts,k = 4000)

    sol_min = [2.0,2.5]
    ppost_min = [-35,-30]
    '''
    div = diversity(initial_muts)
    perf = performance(initial_muts,oracle)
    nov = novelty(initial_muts,initial_muts)
    full_seqs = ['CAK'+i+'W' for i in initial_muts]
    instability,hydro,charge = oracle.get_dev_score(full_seqs)
    print('global_weight:{},div:{},performance:{},nov:{}'.format(4.0,div,perf,nov))
    print('insta:{},hydro:{},charge:{}'.format(np.mean(instability),np.mean(hydro),np.mean(charge)))
    '''
    for s in sol_min:
        for p in ppost_min:
            seqs = []
            with open('./lib/gen_seqs/antBO_sol_min:{}_ppost_min:{}.txt'.format(s,p),'r') as f:
                for line in f:
                    seqs.append(line.split('\n')[0])
            print('sasa_min:{}'.format(s))
            print('ppost_min:{}'.format(p))
            seqs = random.choices(seqs,k = 4000) 
            div = diversity(seqs)
            perf = performance(seqs,oracle)
            nov = novelty(initial_muts,seqs)
            full_seqs = ['CAK'+i+'W' for i in seqs]
            instability,hydro,charge = oracle.get_dev_score(full_seqs)
            print('div:{},performance:{},nov:{}'.format(div,perf,nov))
            print('insta:{},hydro:{},charge:{}'.format(np.mean(instability),np.mean(hydro),np.mean(charge)))

def make_logos():
    '''
    initial_muts = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])
    initial_muts = random.choices(initial_muts,k = 4000)
    '''
    global_weights = [20.0]
    for gw in global_weights:
        #w = [[1.0,0.5,1.5,i,1.0,1.0],[1.0,0.3,1.7,i,1.0,1.0],[1.0,0.2,1.8,i,1.0,1.0],[1.0,0.0,2.0,i,1.0,1.0]]
        w = [[1.0,0.55,1.45,gw,1.0,1.0]]
        for weights in w:
            seqs = []
            with open('./lib/dataset/mcmc_all_exp_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:                
                for line in f:
                    seqs.append(line.split('\n')[0])
            seqs = random.choices(seqs,k = 4000)
            aa_alphabet = 'RHKDESTNQCGPAVILMFYW'
            logo_matrix = np.zeros((len(seqs[0]), 20))
            logo_df = pd.DataFrame(logo_matrix, columns=[l for l in aa_alphabet])
            for s in seqs:
                for i in range(len(s)):
                    logo_df.loc[i, s[i]] += 1/len(seqs)
                
            fig, ax = plt.subplots(figsize=(len(seqs[0])*0.6, 2))
            logomaker.Logo(logo_df, ax=ax)
            plt.savefig('./figures_final/logo_gw:{}_aff:{}_sol:{}.png'.format(gw,weights[2],weights[1]))
            plt.clf()

def extract_mid_seqs(oracle):
    oracle.gen_all_cdr = True
    weights = [1.0,0.55,1.55,15.0,1.0,1.0]
    seqs = []
    with open('./lib/dataset/mcmc_all_exp_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5]),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    seqs = random.choices(seqs,k = 3000)
    sasa_r,aff_r,var_r = oracle.return_indiv_scores(seqs)
    good_seqs = list(set([seqs[i] for i in range(len(aff_r)) if aff_r[i] > 4.1 and aff_r[i] < 5]))
    print(len(good_seqs))
    with open('./lib/dataset/mcmc_middle_seeds.txt','w') as f:
        for s in good_seqs:
            f.write('{}\n'.format(s))

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device('cuda')
    #oracle = get_oracle(args)
    remove_duplicates()
    remove_duplicates_true_aff()
    remove_duplicates_true_aff_hard()
    #compute_score_true_aff(oracle)
    #compute_score_true_aff_hard(oracle)
    #distribution_of_scores_true_aff(oracle)
    #oracle = get_oracle(args)
    #covid_histo(oracle)
    #remove_duplicates_true_aff()
    #plot_pareto_mcmc_true_aff(oracle)
    #plot_pareto_true_aff_gw_density(oracle)
    #plot_pareto_all_gw_density(oracle)
    #plot_pareto_all_color_density(oracle)
    #plot_pareto_compare_all(oracle)
    #extract_mid_seqs(oracle)
    #make_logos()
    #avg_dist()
    #analyze_covid(oracle)
    #remove_duplicates()
    #oracle = get_oracle(args)
    #plot_pareto(oracle)
    #check_scores(oracle)
    #plot_pareto_true_aff(oracle)
    #plot_pareto_lim(oracle)
    #plot_pareto_true_aff(oracle)
    #plot_histo_scores(oracle)
    #plot_pareto_gflow(oracle)
    #plot_pareto_all_color(oracle)
    #plot_pareto_all_color_density(oracle)
    #plot_initial_muts(oracle)
    #compare_true_aff_var_mean_ori(oracle)
    #analyze_true_aff(oracle)
    #plot_pareto_true_aff_gw_density_2(oracle)
    #plot_pareto_true_aff_pareto_all_seqs(oracle)
    #plot_pareto_true_aff_pareto_sampled_seqs(oracle)
    #analyze_ant_BO_true_aff(oracle)
    #compare_gflowmcmc_mcmc()
    #analyze_covid_gflowmcmc(oracle)

    #plot_pareto_antBO(oracle)
    #plot_sim_anneal(oracle)
    #true_aff_histo(oracle)
    #a = ['KKPPLEDLF','KAIPYEDLF','KRMPLEDLF','KAPPDEDLY','LAPVDEDLF','KSPWREDLF','KAPPLELLF','MAPIHEDLP','KAHPLERLI','KMQPLLPLF']
    #plot_pareto_true_aff(oracle)
    #compare_true_aff_var(oracle)
    #find_best_dev_mut_true_aff(oracle)
    #plot_mean_gp_IGLM(oracle)
    #compute_score_2(oracle)
    #verify_oracle(oracle)
    #plot_pareto(oracle)
    #plot_histo_pareto(oracle)
    #plot_pareto_ppost(oracle)
    #plot_pareto_ppost_sol(oracle)
    #get_covid_seqs(oracle)
    #compute_mean()
    #diversity()
    #process_length()
    #compare_thera_IGLM_ppost(oracle)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)