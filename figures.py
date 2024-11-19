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
import seaborn as sns
from scipy.integrate import simpson
from matplotlib.lines import Line2D

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
#from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
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

def novelty(init,gen,return_std = False):
    nov = []
    for i in range(len(gen)):
        min_d = 33
        for j in init:
            dist = distance(gen[i],j)
            if dist < min_d:
                min_d = dist
        nov.append(min_d)
    if not return_std:
        return np.mean(nov)
    else:
        return np.mean(nov),np.std(nov)/np.sqrt(len(nov))

def diversity(seqs,return_std = False):
    div = []
    for i in range(len(seqs)):
        dist = [distance(seqs[i],seqs[j]) for j in range(len(seqs)) if i != j]
        div.append(np.mean(dist))
    if not return_std:
        return np.mean(div)
    else:
        return np.mean(div),np.std(div)/np.sqrt(len(div))

def remove_dominated(aff_r,sol_r):
    norm_aff_r = (aff_r - np.mean(aff_r))/np.std(aff_r)
    norm_sol_r = (sol_r - np.mean(sol_r))/np.std(sol_r)
    scores = [(aff_r[i],sol_r[i]) for i in range(len(aff_r))]
    norm_scores = [(norm_aff_r[i],norm_sol_r[i]) for i in range(len(aff_r))]
    pareto = []
    norm_pareto = []
    not_pareto = []
    norm_not_pareto = []
    for i in tqdm(range(len(scores))):
        dom = True
        for j in range(len(scores)):
            if (scores[i][0] < scores[j][0] and scores[i][1] < scores[j][1]) or (scores[i][0] == scores[j][0] and scores[i][1] < scores[j][1]) or (scores[i][0] < scores[j][0] and scores[i][1] == scores[j][1]):
                not_pareto.append(scores[i])
                norm_not_pareto.append(norm_scores[i])
                dom = False
                break
        if dom:
            pareto.append(scores[i])
            norm_pareto.append(norm_scores[i])

    dist_from_pareto = []
    for i in norm_scores:
        min_dist = 10000
        for j in norm_pareto:
            dist = np.sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2)
            min_dist = min(min_dist,dist)
        dist_from_pareto.append(min_dist)

    aff_pareto = np.array([i[0] for i in pareto])
    sol_pareto = np.array([i[1] for i in pareto])
    aff_not_pareto = np.array([i[0] for i in not_pareto])
    sol_not_pareto = np.array([i[1] for i in not_pareto])
    dist_from_pareto = np.array(dist_from_pareto)
    return aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist_from_pareto

def find_pareto_idx(seqs,aff_r,sol_r):
    norm_aff_r = (aff_r - np.mean(aff_r))/np.std(aff_r)
    norm_sol_r = (sol_r - np.mean(sol_r))/np.std(sol_r)
    scores = [(aff_r[i],sol_r[i]) for i in range(len(seqs))]
    norm_scores = [(norm_aff_r[i],norm_sol_r[i]) for i in range(len(aff_r))]
    pareto = []
    not_pareto = []
    norm_pareto = []
    norm_not_pareto = []
    for i in range(len(scores)):
        dom = True
        for j in range(len(scores)):
            if (scores[i][0] < scores[j][0] and scores[i][1] < scores[j][1]) or (scores[i][0] == scores[j][0] and scores[i][1] < scores[j][1]) or (scores[i][0] < scores[j][0] and scores[i][1] == scores[j][1]):
                not_pareto.append(scores[i])
                norm_not_pareto(norm_scores[i])
                dom = False
                break
        if dom:
            pareto.append(scores[i])
            norm_pareto(norm_scores[i])
    dist_from_pareto = []
    for i in range(len(seqs)):
        min_dist = 10000
        for j in range(len()):
            dist = np.sqrt((norm_aff_r[i] - norm_aff_r[j])**2 + (norm_sol_r[i] - norm_sol_r[j])**2)
            min_dist = min(min_dist,dist)
        dist_from_pareto.append(min_dist)
    return pareto,not_pareto,dist_from_pareto

## Preprocess initial covid seqs

def preprocess_pareto_init_covid(oracle):
    seqs = get_covid_initials()
    seqs = list(set(seqs))
    batch_size = 128
    n_batches = int(len(seqs)/batch_size) + 1
    sol_r = np.array([])
    aff_r = np.array([])
    mu_r = np.array([])
    ppost_r = np.array([])
    perf_r = np.array([])
    insta_r = np.array([])
    hydro_r = np.array([])
    charge_r = np.array([])
    for j in tqdm(range(n_batches)):
        s_r,a_r,v_r = oracle.return_indiv_scores(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r = -a_r
        instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        sol_r = np.concatenate((sol_r,s_r))
        aff_r = np.concatenate((aff_r,a_r))
        insta_r = np.concatenate((insta_r,instability))
        hydro_r = np.concatenate((hydro_r,hydrophobicity))
        charge_r = np.concatenate((charge_r,charge))
        with open('./lib/dataset/preprocess/init_covid_seqs_preprocess.txt','w') as f:
            for seq in seqs:
                f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/init_covid_aff_preprocess.npy',aff_r)
        np.save('./lib/dataset/preprocess/init_covid_sol_preprocess.npy',sol_r)
        np.save('./lib/dataset/preprocess/init_covid_insta_preprocess.npy',insta_r)
        np.save('./lib/dataset/preprocess/init_covid_hydro_preprocess.npy',hydro_r)
        np.save('./lib/dataset/preprocess/init_covid_charge_preprocess.npy',charge_r)

def get_init_covid_no_dupl():
    aff_r = np.load('./lib/dataset/preprocess/init_covid_aff_preprocess.npy')
    sol_r = np.load('./lib/dataset/preprocess/init_covid_sol_preprocess.npy')
    insta_r = np.load('./lib/dataset/preprocess/init_covid_insta_preprocess.npy')
    hydro_r = np.load('./lib/dataset/preprocess/init_covid_hydro_preprocess.npy')
    charge_r = np.load('./lib/dataset/preprocess/init_covid_charge_preprocess.npy')
    seqs = []
    with open('./lib/dataset/preprocess/init_covid_seqs_preprocess.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,insta_r,hydro_r,charge_r

def get_mcmc_covid_seqs(weights,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/mcmc/covid/mcmc_all_exp_lim_burn_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
    init_seqs,_,_,_,_,_ = get_init_covid_no_dupl()
    seqs = [seq for seq in seqs if seq not in init_seqs]
    return seqs,score

def get_all_mcmc_covid_seqs(gw,beta,replicate):
    weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.125,0.875,gw,beta,1.0],[1.0,0.1,0.9,gw,beta,1.0],[1.0,0.05,0.95,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    mcmc_seqs = []
    mcmc_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_mcmc_covid_seqs(w,replicate)
        mcmc_seqs += seqs
        mcmc_score = np.concatenate((mcmc_score,score))
    return mcmc_seqs, mcmc_score

def preprocess_pareto_mcmc(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [2.0,1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for i in global_weights:
        for b in beta:
            seqs,score = get_all_mcmc_covid_seqs(i,b,replicate)
            seqs = list(set(seqs))
            n_batches = int(len(seqs)/batch_size) + 1
            sol_r = np.array([])
            aff_r = np.array([])
            insta_r = np.array([])
            hydro_r = np.array([])
            charge_r = np.array([])
            for j in tqdm(range(n_batches)):
                s_r,a_r,v_r = oracle.return_indiv_scores(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = a_r + b * v_r
                instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                sol_r = np.concatenate((sol_r,s_r))
                aff_r = np.concatenate((aff_r,a_r))
                insta_r = np.concatenate((insta_r,instability))
                hydro_r = np.concatenate((hydro_r,hydrophobicity))
                charge_r = np.concatenate((charge_r,charge))
            with open('./lib/dataset/preprocess/mcmc_all_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),'w') as f:
                for seq in seqs:
                    f.write(seq + '\n')
            np.save('./lib/dataset/preprocess/mcmc_all_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),aff_r)
            np.save('./lib/dataset/preprocess/mcmc_all_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),sol_r)
            np.save('./lib/dataset/preprocess/mcmc_covid_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),insta_r)
            np.save('./lib/dataset/preprocess/mcmc_covid_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),hydro_r)
            np.save('./lib/dataset/preprocess/mcmc_covid_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),charge_r)

def preprocess_pareto_mcmc_combine(oracle):
    global_weights = [20.0,25.0,30.0]
    beta = [2.0,1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for b in beta:
        all_seqs = []
        for i in global_weights:
            seqs,score = get_all_mcmc_covid_seqs(i,b,replicate)
            all_seqs = all_seqs + seqs
        seqs = list(set(all_seqs))
        n_batches = int(len(seqs)/batch_size) + 1
        sol_r = np.array([])
        aff_r = np.array([])
        insta_r = np.array([])
        hydro_r = np.array([])
        charge_r = np.array([])
        for j in tqdm(range(n_batches)):
            s_r,a_r,v_r = oracle.return_indiv_scores(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            a_r = a_r + b * v_r
            instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            sol_r = np.concatenate((sol_r,s_r))
            aff_r = np.concatenate((aff_r,a_r))
            insta_r = np.concatenate((insta_r,instability))
            hydro_r = np.concatenate((hydro_r,hydrophobicity))
            charge_r = np.concatenate((charge_r,charge))
        with open('./lib/dataset/preprocess/mcmc_all_seqs_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),'w') as f:
            for seq in seqs:
                f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/mcmc_all_aff_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),aff_r)
        np.save('./lib/dataset/preprocess/mcmc_all_sol_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),sol_r)
        np.save('./lib/dataset/preprocess/mcmc_covid_insta_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),insta_r)
        np.save('./lib/dataset/preprocess/mcmc_covid_hydro_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),hydro_r)
        np.save('./lib/dataset/preprocess/mcmc_covid_charge_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),charge_r)

def get_mcmc_covid_dev_no_dupl(g,beta,replicate):
    charge_r = np.load('./lib/dataset/preprocess/mcmc_covid_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    insta_r = np.load('./lib/dataset/preprocess/mcmc_covid_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    hydro_r = np.load('./lib/dataset/preprocess/mcmc_covid_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    return charge_r,insta_r,hydro_r

def get_mcmc_covid_dev_no_dupl_combine(beta,replicate):
    charge_r = np.load('./lib/dataset/preprocess/mcmc_covid_charge_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    insta_r = np.load('./lib/dataset/preprocess/mcmc_covid_insta_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    hydro_r = np.load('./lib/dataset/preprocess/mcmc_covid_hydro_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    return charge_r,insta_r,hydro_r

def get_mcmc_covid_no_dupl(g,beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/mcmc_all_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/mcmc_all_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    seqs = []
    aff_r = aff_r - 7
    with open('./lib/dataset/preprocess/mcmc_all_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r

def get_mcmc_covid_no_dupl_combine(beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/mcmc_all_aff_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/mcmc_all_sol_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    aff_r = aff_r - 7
    seqs = []
    with open('./lib/dataset/preprocess/mcmc_all_seqs_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r

def get_mcmc_covid_dict(g,beta,replicate):
    seqs,aff_r,sol_r = get_mcmc_covid_no_dupl(g,beta,replicate)
    aff_r = aff_r - 7
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i])
    return d

def get_mcmc_covid_dict_combine(beta,replicate):
    seqs,aff_r,sol_r = get_mcmc_covid_no_dupl_combine(beta,replicate)
    aff_r = aff_r - 7
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i])
    return d

def get_mcmc_covid_pareto_front(gw,beta,replicate):
    seqs,aff_r,sol_r = get_mcmc_covid_no_dupl(gw,beta,replicate)
    charge_r,insta_r,hydro_r = get_mcmc_covid_dev_no_dupl(gw,beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,charge_r,insta_r,hydro_r

def get_mcmc_covid_pareto_front_combine(beta,replicate):
    seqs,aff_r,sol_r = get_mcmc_covid_no_dupl_combine(beta,replicate)
    charge_r,insta_r,hydro_r = get_mcmc_covid_dev_no_dupl_combine(beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,charge_r,insta_r,hydro_r

def make_logos_mcmc_covid_combine():
    beta = [-1.0,0.0,1.0,2.0]
    global_weight = [20.0]
    replicate = 1
    fig, ax = plt.subplots(4,figsize=(33*0.6, 12))
    for gw in global_weight:
        for j in range(len(beta)):
            b = beta[j]
            seqs,aff_r,sol_r = get_mcmc_covid_no_dupl_combine(b,replicate)
            aa_alphabet = 'RHKDESTNQCGPAVILMFYW'
            logo_matrix = np.zeros((len(seqs[0]), 20))
            logo_df = pd.DataFrame(logo_matrix, columns=[l for l in aa_alphabet])
            for s in seqs:
                for i in range(len(s)):
                    logo_df.loc[i, s[i]] += 1/len(seqs)
                
            logomaker.Logo(logo_df, ax=ax[j])
            ax[j].set_title(r'Metropolis Hastings / $\beta$ = {}'.format(b))
    plt.tight_layout()
    plt.savefig('./pareto_final/logo_mcmc_covid_replicate_{}.png'.format(replicate))

def get_chain_length_mcmc_covid():
    beta = [-1.0,0.0,1.0,2.0]
    global_weight = [20.0]
    replicate = 1
    table_array = [[r'$\beta$',r'$w_{aff}$','Chain Length after burn-in']]
    fig, ax = plt.subplots(figsize = (8,6),dpi = 160)

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    for gw in global_weight:
        for b in beta:
            weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for w in weights:
                seqs,score = get_mcmc_covid_seqs(w,replicate)
                print(score.shape)
                table_array.append([b,w[2],int(len(score)/8)])
    print(len(table_array))
    plt.table(table_array,cellLoc = 'center',loc = 'center')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_covid_table_chain_length.png')

def get_chain_length_mcmc_true_aff():
    beta = [-1.0,0.0,1.0,2.0]
    global_weight = [10.0]
    replicate = 1
    table_array = [[r'$\beta$',r'$w_{aff}$','Chain Length after burn-in']]
    fig, ax = plt.subplots(figsize = (8,6),dpi = 160)

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    for gw in global_weight:
        for b in beta:
            weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for w in weights:
                seqs,score = get_mcmc_true_aff_seqs(w,replicate)
                print(score.shape)
                table_array.append([b,w[2],int(len(score)/8)])
    print(len(table_array))
    plt.table(table_array,cellLoc = 'center',loc = 'center')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_table_chain_length.png')

def get_chain_length_mcmc_true_aff_hard():
    beta = [-1.0,0.0,1.0]
    global_weight = [10.0]
    replicate = 1
    table_array = [[r'$\beta$',r'$w_{aff}$','Chain Length after burn-in']]
    fig, ax = plt.subplots(figsize = (8,6),dpi = 160)

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    for gw in global_weight:
        for b in beta:
            weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for w in weights:
                seqs,score = get_mcmc_true_aff_hard_seqs(w,replicate)
                print(score.shape)
                table_array.append([b,w[2],int(len(score)/8)])
    print(len(table_array))
    plt.table(table_array,cellLoc = 'center',loc = 'center')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_table_chain_length.png')

def plot_mcmc_pareto_covid_top():
    beta = [-1.0,0.0,1.0,2.0]
    global_weight = [20.0,25.0,30.0]
    replicate = 1
    fig, ax = plt.subplots(1,4,figsize = (24,6))
    for i in range(len(beta)):
        b = beta[i]
        for g in global_weight:
            seqs,aff_r,sol_r = get_mcmc_covid_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            ax[i].scatter(pareto_aff,pareto_sol,label = r'$T^{-1}$' + ' = {}'.format(g))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            ax[i].plot(pareto_aff,pareto_sol)
        ax[i].set_ylabel(r'$\hat f_{\rm sol}$')
        ax[i].set_ylim(2,7)
        ax[i].set_xlabel(r'$\hat f_{\rm aff}$')
        ax[i].set_title(r'$\beta$ = {}'.format(b))
        ax[i].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_all_compare_pareto_temp_rep:{}.png'.format(replicate))

def get_color_plot(aff,sol):
    aff = np.array(aff)
    sol = np.array(sol)
    xy = np.vstack([aff,sol])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = aff[idx], sol[idx], z[idx]
    return x,y,z

def plot_mcmc_pareto_covid_density():
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    for i in range(len(global_weights)):
        gw = global_weights[i]
        fig,axs = plt.subplots(1,4,figsize = (32,6))
        for m in range(len(beta)):
            b = beta[m]
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist_from_pareto,charge_r,insta_r,hydro_r = get_mcmc_covid_pareto_front(gw,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[m].scatter(x,y,c = z,label = 'MCMC generated seqs')
            plt.colorbar(z,ax = axs[m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
            axs[m].set_xlabel('affinity score')
            axs[m].set_ylabel('solubility score')
            axs[m].legend()
            axs[m].set_title('MCMC/GW:{}/beta:{}'.format(gw,b))
        plt.tight_layout()
        plt.savefig('./pareto_final/pareto_mcmc_covid_density_gw:{}_initial:{}_rep:{}.png'.format(gw,add_initial,replicate))

def plot_mcmc_pareto_covid_density_top():
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    for i in range(len(global_weights)):
        gw = global_weights[i]
        fig,axs = plt.subplots(1,4,figsize = (32,6))
        for m in range(len(beta)):
            b = beta[m]
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_mcmc_covid_pareto_front(gw,b,replicate)
            top_idx = np.argsort(dist)[:1000]
            top_seqs = [seqs[i] for i in top_idx]
            top_aff = all_aff[top_idx]
            top_sol = all_sol[top_idx]
            x,y,z = get_color_plot(top_aff,top_sol)
            z = axs[m].scatter(x,y,c = z,label = 'MCMC generated seqs')
            plt.colorbar(z,ax = axs[m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
            axs[m].set_xlabel('Affinity Score')
            axs[m].set_ylabel('Solubility Score')
            axs[m].legend()
            axs[m].set_title(r'MCMC / $\beta$:' + '{}'.format(b))
        plt.tight_layout()
        plt.savefig('./pareto_final/pareto_mcmc_covid_density_top_gw:{}_initial:{}_rep:{}.png'.format(gw,add_initial,replicate))

def plot_mcmc_pareto_covid_density_combine():
    # 3 plots, one for each beta, global weight combined
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,4,figsize = (32,6))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist_from_pareto,charge_r,insta_r,hydro_r = get_mcmc_covid_pareto_front_combine(b,replicate)
        idx = random.choices(range(len(seqs)),k = 1000)
        seqs = [seqs[i] for i in idx]
        all_aff = all_aff[idx]
        all_sol = all_sol[idx]
        x,y,z = get_color_plot(all_aff,all_sol)
        z = axs[m].scatter(x,y,c = z,label = 'MCMC generated seqs')
        plt.colorbar(z,ax = axs[m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
        axs[m].set_xlabel('affinity score')
        axs[m].set_ylabel('solubility score')
        axs[m].legend()
        axs[m].set_title('MCMC/beta:{}'.format(b))
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_mcmc_covid_density_combine_initial:{}_rep:{}.png'.format(add_initial,replicate))

def plot_mcmc_pareto_covid_density_combine_top():
    global_weights = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,3,figsize = (32,6))
    solubility_ticks = [2,3,4,5,6,7]
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_mcmc_covid_pareto_front_combine(b,replicate)
        top_idx = np.argsort(dist)[:1000]
        top_seqs = [seqs[i] for i in top_idx]
        top_aff = all_aff[top_idx]
        top_sol = all_sol[top_idx]
        x,y,z = get_color_plot(top_aff,top_sol)
        z = axs[m].scatter(x,y,c = z,label = 'MCMC generated seqs')
        plt.colorbar(z,ax = axs[m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[m].set_ylim(2,7)
        axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
        axs[m].set_xlabel(r'$\mu_{aff}$ + $\beta_{epi}$ x $\sigma_{aff}$')
        axs[m].set_ylabel('solubility score')
        axs[m].legend()
        axs[m].set_title(r'MCMC / $\beta_{epi}$' + ' : {}'.format(b))
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_mcmc_covid_density_top_combine_initial:{}_rep:{}.png'.format(add_initial,replicate))

### END OF MCMC SECTION
### START OF GFLOW SECTION

def plot_spearmanr_losses():
    beta = [-1.0,0.0,1.0,2.0]
    gw = 20.0
    replicate = 1
    plt.figure(figsize = (8,6))
    for b in beta:
        #weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
        weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
        for global_weight in weights:
            loss = np.load('./losses/spearmanr_mcmc_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate))
            plt.plot(np.array(range(len(loss))) * 100,loss,label = r'w' + r' : {} / '.format(global_weight[2]) + r'$\beta$:{}'.format(global_weight[4],replicate))
            plt.xlabel('iterations')
            plt.ylabel('spearmanr correlation')
            plt.title('CB-119 peptide')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('./losses/spearman_gflow_covid_rep{}.png'.format(replicate))

def plot_spearmanr_losses_true_aff():
    beta = [-1.0,0.0,1.0,2.0]
    gw = 10.0
    replicate = 1
    plt.figure(figsize = (8,6))
    for b in beta:
        weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
        for global_weight in weights:
            loss = np.load('./losses/true_aff_spearmanr_mcmc_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate))
            plt.plot(np.array(range(len(loss))) * 100,loss,label = r'w' + r' : {} / '.format(global_weight[2]) + r'$\beta$:{}'.format(global_weight[4],replicate))
            plt.xlabel('iterations')
            plt.ylabel('spearmanr correlation')
            plt.title('Simple')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('./losses/spearman_gflow_true_aff_rep:{}.png'.format(replicate))

def plot_spearmanr_losses_true_aff_hard():
    beta = [-1.0,0.0,1.0,2.0]
    gw = 10.0
    replicate = 1
    plt.figure(figsize = (11,6))
    for b in beta:
        weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
        for global_weight in weights:
            loss = np.load('./losses/true_aff_hard_spearmanr_mcmc_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate))
            plt.plot(np.array(range(len(loss))) * 100,loss,label = r'w' + r' : {} / '.format(global_weight[2]) + r'$\beta$:{}'.format(global_weight[4],replicate))
            plt.xlabel('iterations')
            plt.ylabel('Spearmanr correlation')
            plt.ylim(-0.4,1.0)
            plt.title('Hard')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.2))
    plt.tight_layout()
    plt.savefig('./losses/spearman_gflow_true_aff_hard_rep:{}.png'.format(replicate))

def get_gflow_covid_seqs(global_weight,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/gflownet/covid/gflow_sol:{}_aff:{}_global:{}_beta:{}_rep:{}.txt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/gflownet/covid/gflow_scores_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate))
    init_seqs,_,_,_,_,_ = get_init_covid_no_dupl()
    seqs = [seq for seq in seqs if seq not in init_seqs]
    return seqs,score

def get_all_gflow_covid_seqs(gw,beta,replicate):
    weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.125,0.875,gw,beta,1.0],[1.0,0.1,0.9,gw,beta,1.0],[1.0,0.05,0.95,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    gflow_seqs = []
    gflow_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_gflow_covid_seqs(w,replicate)
        gflow_seqs += seqs
        gflow_score = np.concatenate((gflow_score,score))
    return gflow_seqs, gflow_score

def preprocess_pareto_gflow(oracle):
    #global_weights = [20.0,25.0,30.0]
    global_weights = [20.0]
    beta = [2.0,1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for i in global_weights:
        for b in beta:
            seqs,score = get_all_gflow_covid_seqs(i,b,replicate)
            seqs = list(set(seqs))
            n_batches = int(len(seqs)/batch_size) + 1
            sol_r = np.array([])
            aff_r = np.array([])
            insta_r = np.array([])
            hydro_r = np.array([])
            charge_r = np.array([])
            for j in tqdm(range(n_batches)):
                s_r,a_r,v_r = oracle.return_indiv_scores(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = a_r + b * v_r
                instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                sol_r = np.concatenate((sol_r,s_r))
                aff_r = np.concatenate((aff_r,a_r))
                insta_r = np.concatenate((insta_r,instability))
                hydro_r = np.concatenate((hydro_r,hydrophobicity))
                charge_r = np.concatenate((charge_r,charge))
            with open('./lib/dataset/preprocess/gflow_all_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),'w') as f:
                for seq in seqs:
                    f.write(seq + '\n')
            np.save('./lib/dataset/preprocess/gflow_all_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),aff_r)
            np.save('./lib/dataset/preprocess/gflow_all_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),sol_r)
            np.save('./lib/dataset/preprocess/gflow_covid_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),insta_r)
            np.save('./lib/dataset/preprocess/gflow_covid_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),hydro_r)
            np.save('./lib/dataset/preprocess/gflow_covid_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),charge_r)

def preprocess_pareto_gflow_combine(oracle):
    #global_weights = [20.0,25.0,30.0]
    global_weights = [20.0]
    beta = [2.0,1.0,-1.0,0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for b in beta:
        all_seqs = []
        for i in global_weights:
            seqs,score = get_all_gflow_covid_seqs(i,b,replicate)
            all_seqs = all_seqs + seqs
        seqs = list(set(all_seqs))
        n_batches = int(len(seqs)/batch_size) + 1
        sol_r = np.array([])
        aff_r = np.array([])
        insta_r = np.array([])
        hydro_r = np.array([])
        charge_r = np.array([])
        for j in tqdm(range(n_batches)):
            s_r,a_r,v_r = oracle.return_indiv_scores(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            a_r = a_r + b * v_r
            instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            sol_r = np.concatenate((sol_r,s_r))
            aff_r = np.concatenate((aff_r,a_r))
            insta_r = np.concatenate((insta_r,instability))
            hydro_r = np.concatenate((hydro_r,hydrophobicity))
            charge_r = np.concatenate((charge_r,charge))
        with open('./lib/dataset/preprocess/gflow_all_seqs_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),'w') as f:
            for seq in seqs:
                f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/gflow_all_aff_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),aff_r)
        np.save('./lib/dataset/preprocess/gflow_all_sol_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),sol_r)
        np.save('./lib/dataset/preprocess/gflow_covid_insta_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),insta_r)
        np.save('./lib/dataset/preprocess/gflow_covid_hydro_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),hydro_r)
        np.save('./lib/dataset/preprocess/gflow_covid_charge_preprocess_combine_beta:{}_rep_{}.npy'.format(b,replicate),charge_r)

def get_gflow_covid_dev_no_dupl(g,beta,replicate):
    charge_r = np.load('./lib/dataset/preprocess/gflow_covid_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    insta_r = np.load('./lib/dataset/preprocess/gflow_covid_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    hydro_r = np.load('./lib/dataset/preprocess/gflow_covid_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    return charge_r,insta_r,hydro_r

def get_gflow_covid_dev_no_dupl_combine(beta,replicate):
    charge_r = np.load('./lib/dataset/preprocess/gflow_covid_charge_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    insta_r = np.load('./lib/dataset/preprocess/gflow_covid_insta_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    hydro_r = np.load('./lib/dataset/preprocess/gflow_covid_hydro_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    return charge_r,insta_r,hydro_r

def get_gflow_covid_no_dupl(g,beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/gflow_all_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/gflow_all_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    seqs = []
    with open('./lib/dataset/preprocess/gflow_all_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r

def get_gflow_covid_no_dupl_combine(beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/gflow_all_aff_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/gflow_all_sol_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate))
    seqs = []
    with open('./lib/dataset/preprocess/gflow_all_seqs_preprocess_combine_beta:{}_rep_{}.npy'.format(beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r

def get_gflow_covid_dict(g,beta,replicate):
    seqs,aff_r,sol_r = get_gflow_covid_no_dupl(g,beta,replicate)
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i])
    return d

def get_gflow_covid_dict_combine(beta,replicate):
    seqs,aff_r,sol_r = get_gflow_covid_no_dupl_combine(beta,replicate)
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i])
    return d

def get_gflow_covid_pareto_front(gw,beta,replicate):
    seqs,aff_r,sol_r = get_gflow_covid_no_dupl(gw,beta,replicate)
    aff_r = aff_r - 7
    charge_r,insta_r,hydro_r = get_gflow_covid_dev_no_dupl(gw,beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,charge_r,insta_r,hydro_r

def get_gflow_covid_pareto_front_combine(beta,replicate):
    seqs,aff_r,sol_r = get_gflow_covid_no_dupl_combine(beta,replicate)
    aff_r = aff_r - 7
    charge_r,insta_r,hydro_r = get_gflow_covid_dev_no_dupl_combine(beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,charge_r,insta_r,hydro_r

def make_logos_gflow_covid_combine():
    beta = [-1.0,0.0,1.0,2.0]
    global_weight = [20.0]
    replicate = 1
    fig, ax = plt.subplots(3,figsize=(33*0.6, 12))
    for gw in global_weight:
        for j in range(len(beta)):
            b = beta[j]
            seqs,aff_r,sol_r = get_gflow_covid_no_dupl_combine(b,replicate)
            aa_alphabet = 'RHKDESTNQCGPAVILMFYW'
            logo_matrix = np.zeros((len(seqs[0]), 20))
            logo_df = pd.DataFrame(logo_matrix, columns=[l for l in aa_alphabet])
            for s in seqs:
                for i in range(len(s)):
                    logo_df.loc[i, s[i]] += 1/len(seqs)
                
            logomaker.Logo(logo_df, ax=ax[j])
            ax[j].set_title(r'GFlowNet / $\beta$ = {}'.format(b))
    plt.tight_layout()
    plt.savefig('./pareto_final/logo_gflow_covid_replicate_{}.png'.format(replicate))

def plot_gflow_pareto_covid_top():
    beta = [2.0,1.0,-1.0,0.0]
    #global_weight = [20.0,25.0,30.0]
    global_weights = [20.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r = get_gflow_covid_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('GFLOW Covid Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_all_compare_pareto_temp_rep:{}.png'.format(replicate))

def plot_gflow_pareto_covid_density():
    #global_weights = [20.0,25.0,30.0]
    global_weights = [20.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,4,figsize = (32,6))
    for m in range(len(beta)):
        b = beta[m]
        for i in range(len(global_weights)):
            gw = global_weights[i]
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_gflow_covid_pareto_front(gw,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[m].scatter(x,y,c = z,label = 'gflow generated seqs')
            plt.colorbar(z,ax = axs[m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[m].set_xlabel('affinity score')
            axs[m].set_ylabel('solubility score')
            axs[m].legend()
            axs[m].set_title('gflow/GW:{}/beta:{}'.format(gw,b))


            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_covid_density_initial:{}_rep:{}.png'.format(add_initial,replicate))

def plot_gflow_pareto_covid_density_combine():
    # 3 plots, one for each beta, global weight combined
    #global_weights = [20.0,25.0,30.0]
    global_weights = [20.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,4,figsize = (32,6))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_gflow_covid_pareto_front_combine(b,replicate)
        idx = random.choices(range(len(seqs)),k = 1000)
        seqs = [seqs[i] for i in idx]
        all_aff = all_aff[idx]
        all_sol = all_sol[idx]
        x,y,z = get_color_plot(all_aff,all_sol)
        z = axs[m].scatter(x,y,c = z,label = 'gflow generated seqs')
        plt.colorbar(z,ax = axs[m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
        axs[m].set_xlabel('affinity score')
        axs[m].set_ylabel('solubility score')
        axs[m].legend()
        axs[m].set_title('gflow/beta:{}'.format(b))
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_covid_density_combine_initial:{}_rep:{}.png'.format(add_initial,replicate))

def plot_gflow_pareto_covid_density_combine_top():
    #global_weights = [20.0,25.0,30.0]
    global_weights = [20.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,4,figsize = (32,6))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_gflow_covid_pareto_front_combine(b,replicate)
        top_idx = np.argsort(dist)[:1000]
        top_seqs = [seqs[i] for i in top_idx]
        top_aff = all_aff[top_idx]
        top_sol = all_sol[top_idx]
        print(np.min(top_aff))
        print(top_aff)
        print(dist)
        x,y,z = get_color_plot(top_aff,top_sol)
        z = axs[m].scatter(x,y,c = z,label = 'gflow generated seqs')
        plt.colorbar(z,ax = axs[m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
        axs[m].set_xlabel('Affinity Score')
        axs[m].set_ylabel('Solubility Score')
        axs[m].legend()
        axs[m].set_title(r'GFLOW / $\beta$:' + '{}'.format(b))
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_covid_density_top_combine_initial:{}_rep:{}.png'.format(add_initial,replicate))

## END OF GFLOW
## START OF MCMC GFLOW COMPARE

def plot_mcmc_gflow_compare_pareto_covid_top():
    beta = [2.0,1.0,-1.0,0.0]
    global_weight = [20.0]
    replicate = 1
    fig,axs = plt.subplots(4,2,figsize = (16,6))
    for i in range(len(beta)):
        b = beta[i]
        for g in global_weight:
            seqs,aff_r,sol_r = get_gflow_covid_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'gflow pareto inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)

            seqs,aff_r,sol_r = get_mcmc_covid_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'mcmc pareto inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)

    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('MCMC/GFLOW Covid Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_covid_compare_pareto_temp_rep:{}.png'.format(replicate))


def plot_compare_pareto_covid_density():
    global_weights = [20.0]
    beta = [2.0,1.0,-1.0,0.0]
    add_initial = True
    replicate = 1
    for b in beta:
        for i in global_weights:
            plt.figure(figsize = (10,6))
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_gflow_covid_pareto_front(i,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            plt.scatter(all_aff,all_sol,color = 'r',label = 'GFlowNet generated seqs')
            plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')

            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,charge_r,insta_r,hydro_r = get_mcmc_covid_pareto_front(i,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            plt.scatter(all_aff,all_sol,color = 'g',label = 'MCMC generated seqs')
            plt.scatter(pareto_aff,pareto_sol,color = 'g',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'g')

            if add_initial:
                cdrlist = ['GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS']
                '''
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                '''
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')
            plt.xlabel('affinity score')
            plt.ylabel('solubility score')
            plt.legend()
            plt.title('GFLOW Covid/GW:{}/beta:{}'.format(i,b))
            plt.tight_layout()
            plt.savefig('./pareto_final/pareto_compare_density_global:{}_beta:{}_initial:{}_rep:{}.png'.format(i,b,add_initial,replicate))
            plt.clf()

def plot_compare_pareto_covid_density_top(oracle):
    global_weights = [20.0]
    beta = [2.0,1.0,-1.0,0.0]
    add_initial = True
    replicate = 1
    for b in beta:
        fig, axs = plt.subplots(1,2,figsize = (16,6))
        for i in global_weights:
            gflow_seqs,gflow_pareto_aff,gflow_pareto_sol,gflow_all_aff,gflow_all_sol,gflow_dist,gflow_charge_r,gflow_insta_r,gflow_hydro_r = get_gflow_covid_pareto_front(i,b,replicate)
            idx = np.argsort(gflow_dist)[:1000]
            gflow_seqs = [gflow_seqs[i] for i in idx]
            gflow_all_aff = gflow_all_aff[idx]
            gflow_all_sol = gflow_all_sol[idx]

            mcmc_seqs,mcmc_pareto_aff,mcmc_pareto_sol,mcmc_all_aff,mcmc_all_sol,mcmc_dist,mcmc_charge_r,mcmc_insta_r,mcmc_hydro_r = get_mcmc_covid_pareto_front(i,b,replicate)
            idx = np.argsort(mcmc_dist)[:1000]
            mcmc_seqs = [mcmc_seqs[i] for i in idx]
            mcmc_all_aff = mcmc_all_aff[idx]
            mcmc_all_sol = mcmc_all_sol[idx]

            idx = np.argsort(gflow_pareto_aff).tolist()
            gflow_pareto_aff = [gflow_pareto_aff[i] for i in idx]
            gflow_pareto_sol = [gflow_pareto_sol[i] for i in idx]

            idx = np.argsort(mcmc_pareto_aff).tolist()
            mcmc_pareto_aff = [mcmc_pareto_aff[i] for i in idx]
            mcmc_pareto_sol = [mcmc_pareto_sol[i] for i in idx]

            x,y,z = get_color_plot(gflow_all_aff,gflow_all_sol)
            axs[1].scatter(x,y,c = z,label = 'Sequences generated by GFlowNet')
            axs[1].plot(mcmc_pareto_aff,mcmc_pareto_sol,'o-',color = '#800080')
            axs[1].plot(gflow_pareto_aff,gflow_pareto_sol,'o-',color = 'r')
            axs[1].set_xlabel(r'$\hat f_{\rm aff}$')
            axs[1].set_ylabel(r'$\hat f_{\rm sol}$')
            axs[1].set_ylim(0,7)
            axs[1].legend(frameon=False, loc = 'lower left')
            leg = axs[1].get_legend()
            leg.legendHandles[0].set_color('black')
            axs[1].set_title(r'$T^{-1}$' + ' : {} /'.format(i) +  r'$\beta$' + ' : {}'.format(b))

            x,y,z = get_color_plot(mcmc_all_aff,mcmc_all_sol)
            axs[0].scatter(x,y,c = z,label = 'Sequences generated by MCMC')
            axs[0].plot(mcmc_pareto_aff,mcmc_pareto_sol,'o-',label = 'MCMC empirical pareto front',color = '#800080')
            axs[0].plot(gflow_pareto_aff,gflow_pareto_sol,'o-',label = 'GFlowNet empirical pareto front',color = 'r')
            axs[0].set_xlabel(r'$\hat f_{\rm aff}$')
            axs[0].set_ylabel(r'$\hat f_{\rm sol}$')
            axs[0].set_ylim(0,7)
            axs[0].legend(frameon=False, loc = 'lower left')
            leg = axs[0].get_legend()
            leg.legendHandles[0].set_color('black')
            axs[0].set_title(r'$T^{-1}$' + ' : {} /'.format(i) +  r'$\beta$' + ' : {}'.format(b))

            if add_initial:
                cdrlist = ['GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS']*2
                '''
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                '''
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(cdrlist)
                aff_r = aff_r + b * var_r
                aff_r = aff_r - 7
                print(sasa_r)
                print(aff_r)
                axs[0].scatter(aff_r[0],sasa_r[0], c = 'r', marker = '+')
                axs[0].text(aff_r[0], sasa_r[0], 'WT', fontsize=12, ha='right', va='top')
                axs[1].scatter(aff_r[0],sasa_r[0], c = 'r', marker = '+')
                axs[1].text(aff_r[0], sasa_r[0], 'WT', fontsize=12, ha='right', va='top')
            plt.tight_layout()
            plt.savefig('./pareto_final/pareto_compare_density_top_global:{}_beta:{}_initial:{}_rep:{}.png'.format(i,b,add_initial,replicate))
            plt.clf()

def get_covid_initials():
    initial_muts = []
    with open('./lib/Covid/data/cdrlist.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])
    return initial_muts

def compare_covid_methods():
    global_weights = [20.0]
    beta = [-1.0,0.0,1.0,2.0]
    budgets = [20,500]
    replicate = 1
    multiple = 10
    results = np.zeros((len(global_weights),len(beta),len(budgets),2,6,multiple))
    init = get_covid_initials()
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(global_weights)):
            gw = global_weights[j]
            mcmc_seqs_o,mcmc_pareto_aff_o,mcmc_pareto_sol_o,mcmc_aff_o,mcmc_sol_o,mcmc_dist_o,mcmc_charge_o,mcmc_insta_o,mcmc_hydro_o = get_mcmc_covid_pareto_front(gw,b,replicate)
            gflow_seqs_o,gflow_pareto_aff_o,gflow_pareto_sol_o,gflow_aff_o,gflow_sol_o,gflow_dist_o,gflow_charge_o,gflow_insta_o,gflow_hydro_o = get_gflow_covid_pareto_front(gw,b,replicate)
            for k in range(len(budgets)):
                budget = budgets[k]
                for n in range(multiple):
                    mcmc_idx = random.choices(range(len(mcmc_seqs_o)),k = budget)
                    mcmc_seqs = [mcmc_seqs_o[i] for i in mcmc_idx]
                    mcmc_sol = mcmc_sol_o[mcmc_idx]
                    mcmc_charge = mcmc_charge_o[mcmc_idx]
                    mcmc_insta = mcmc_insta_o[mcmc_idx]
                    mcmc_hydro = mcmc_hydro_o[mcmc_idx]
                    results[j,i,k,0,0,n] = np.mean(mcmc_sol)
                    results[j,i,k,0,1,n] = diversity(mcmc_seqs)
                    results[j,i,k,0,2,n] = novelty(init,mcmc_seqs)
                    results[j,i,k,0,3,n] = np.mean(mcmc_charge)
                    results[j,i,k,0,4,n] = np.mean(mcmc_insta)
                    results[j,i,k,0,5,n] = np.mean(mcmc_hydro)

                    gflow_idx = random.choices(range(len(gflow_seqs_o)),k = budget)
                    gflow_seqs = [gflow_seqs_o[i] for i in gflow_idx]
                    gflow_sol = gflow_sol_o[gflow_idx]
                    gflow_charge = gflow_charge_o[gflow_idx]
                    gflow_insta = gflow_insta_o[gflow_idx]
                    gflow_hydro = gflow_hydro_o[gflow_idx]
                    results[j,i,k,1,0,n] = np.mean(gflow_sol)
                    results[j,i,k,1,1,n] = diversity(gflow_seqs)
                    results[j,i,k,1,2,n] = novelty(init,gflow_seqs)
                    results[j,i,k,1,3,n] = np.mean(gflow_charge)
                    results[j,i,k,1,4,n] = np.mean(gflow_insta)
                    results[j,i,k,1,5,n] = np.mean(gflow_hydro)

    methods = ['mcmc','gflownet']
    task = ['solubility','diversity','novelty','charge','instability','biopython hydrophobicity']
    fig,axs = plt.subplots(len(task),len(budgets),figsize = (16,36))
    for i in range(len(global_weights)):
        for j in range(len(budgets)):
            for t in range(len(task)):
                axs[t,j].set_xlabel('beta')
                axs[t,j].set_ylabel(task[t])
                axs[t,j].set_title('gw:{}/budget:{}'.format(global_weights[i],budgets[j]))
                for k in range(len(methods)):
                    axs[t,j].plot(beta,np.mean(results[i,:,j,k,t],axis = 1),label = methods[k])
    plt.legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_methods_covid.png')

def compare_covid_methods_top():
    global_weights = [20.0]
    beta = [-1.0,0.0,1.0,2.0]
    budgets = [5,500]
    replicate = 1
    multiple = 10
    results = np.zeros((len(global_weights),len(beta),len(budgets),2,12))
    init = get_covid_initials()
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(global_weights)):
            gw = global_weights[j]
            mcmc_seqs_o,mcmc_pareto_aff_o,mcmc_pareto_sol_o,mcmc_aff_o,mcmc_sol_o,mcmc_dist_o,mcmc_charge_o,mcmc_insta_o,mcmc_hydro_o = get_mcmc_covid_pareto_front(gw,b,replicate)
            gflow_seqs_o,gflow_pareto_aff_o,gflow_pareto_sol_o,gflow_aff_o,gflow_sol_o,gflow_dist_o,gflow_charge_o,gflow_insta_o,gflow_hydro_o = get_gflow_covid_pareto_front(gw,b,replicate)
            for k in range(len(budgets)):
                budget = budgets[k]
                mcmc_idx = np.argsort(mcmc_dist_o)[:budget]
                mcmc_seqs = [mcmc_seqs_o[i] for i in mcmc_idx]
                mcmc_sol = mcmc_sol_o[mcmc_idx]
                mcmc_charge = mcmc_charge_o[mcmc_idx]
                mcmc_insta = mcmc_insta_o[mcmc_idx]
                mcmc_hydro = mcmc_hydro_o[mcmc_idx]
                div_mean, div_std = diversity(mcmc_seqs,return_std = True)
                nov_mean, nov_std = novelty(init,mcmc_seqs,return_std = True)
                results[j,i,k,0,0] = np.mean(mcmc_sol)
                results[j,i,k,0,1] = div_mean
                results[j,i,k,0,2] = nov_mean
                results[j,i,k,0,3] = np.mean(mcmc_charge)
                results[j,i,k,0,4] = np.mean(mcmc_insta)
                results[j,i,k,0,5] = np.mean(mcmc_hydro)
                results[j,i,k,0,6] = np.std(mcmc_sol)
                results[j,i,k,0,7] = div_std
                results[j,i,k,0,8] = nov_std
                results[j,i,k,0,9] = np.std(mcmc_charge)
                results[j,i,k,0,10] = np.std(mcmc_insta)
                results[j,i,k,0,11] = np.std(mcmc_hydro)

                gflow_idx = np.argsort(gflow_dist_o)[:budget]
                gflow_seqs = [gflow_seqs_o[i] for i in gflow_idx]
                gflow_sol = gflow_sol_o[gflow_idx]
                gflow_charge = gflow_charge_o[gflow_idx]
                gflow_insta = gflow_insta_o[gflow_idx]
                gflow_hydro = gflow_hydro_o[gflow_idx]
                div_mean, div_std = diversity(gflow_seqs,return_std = True)
                nov_mean, nov_std = novelty(init,gflow_seqs,return_std = True)
                results[j,i,k,1,0] = np.mean(gflow_sol)
                results[j,i,k,1,1] = div_mean
                results[j,i,k,1,2] = nov_mean
                results[j,i,k,1,3] = np.mean(gflow_charge)
                results[j,i,k,1,4] = np.mean(gflow_insta)
                results[j,i,k,1,5] = np.mean(gflow_hydro)
                results[j,i,k,1,6] = np.std(gflow_sol)
                results[j,i,k,1,7] = div_std
                results[j,i,k,1,8] = nov_std
                results[j,i,k,1,9] = np.std(gflow_charge)
                results[j,i,k,1,10] = np.std(gflow_insta)
                results[j,i,k,1,11] = np.std(gflow_hydro)

    methods = ['MCMC','GFlownet']
    methods_shift = [-0.125,0.125]
    beta = np.array(beta)
    task = ['solubility','Diversity','Novelty','Charge','Instability','biopython hydrophobicity']
    fig,axs = plt.subplots(len(budgets),len(task),figsize = (48,12))
    for i in range(len(global_weights)):
        for j in range(len(budgets)):
            for t in range(len(task)):
                axs[j,t].set_xlabel(r'$\beta$')
                axs[j,t].set_ylabel(task[t])
                for k in range(len(methods)):
                    lower_error = np.zeros_like(results[i,:,j,k,t + 6])
                    axs[j,t].errorbar(beta + methods_shift[k],results[i,:,j,k,t],yerr = [lower_error,results[i,:,j,k,t + 6]],ecolor = 'k', fmt = 'none',capsize = 11,elinewidth = 3)
                    axs[j,t].bar(beta + methods_shift[k],results[i,:,j,k,t],width = 0.25, label = methods[k])
                if (t == 3 and j == 1) or t == 1:
                    axs[j,t].legend(loc="upper right",frameon = False)
                if t == 2 or t == 1:
                    axs[j,t].set_ylim(0,6.5)
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_methods_covid_top.png')

##END OF COMPARE COVID

def get_true_aff_initial():
    initial_muts = []
    with open('./lib/true_aff/muts.txt','r') as f:
        for line in f:
            initial_muts.append(line.split('\n')[0])
    return initial_muts

def preprocess_pareto_init_true_aff(oracle):
    seqs = get_true_aff_initial()
    seqs = list(set(seqs))
    batch_size = 128
    n_batches = int(len(seqs)/batch_size) + 1
    sol_r = np.array([])
    aff_r = np.array([])
    mu_r = np.array([])
    ppost_r = np.array([])
    perf_r = np.array([])
    insta_r = np.array([])
    hydro_r = np.array([])
    charge_r = np.array([])
    for j in tqdm(range(n_batches)):
        s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r = - a_r
        p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        print(oracle.true_aff_ora.method)
        performance = oracle.true_aff_ora.score_without_noise_simple(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        mu_r = np.concatenate((mu_r,a_r))
        sol_r = np.concatenate((sol_r,s_r))
        aff_r = np.concatenate((aff_r,a_r))
        ppost_r = np.concatenate((ppost_r,p_r))
        insta_r = np.concatenate((insta_r,instability))
        hydro_r = np.concatenate((hydro_r,hydrophobicity))
        charge_r = np.concatenate((charge_r,charge))
        perf_r = np.concatenate((perf_r,performance))
    print(np.min(perf_r))
    with open('./lib/dataset/preprocess/init_true_aff_seqs_preprocess.npy','w') as f:
        for seq in seqs:
            f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/init_true_aff_mu_preprocess.npy',mu_r)
        np.save('./lib/dataset/preprocess/init_true_aff_aff_preprocess.npy',aff_r)
        np.save('./lib/dataset/preprocess/init_true_aff_sol_preprocess.npy',sol_r)
        np.save('./lib/dataset/preprocess/init_true_aff_ppost_preprocess.npy',ppost_r)
        np.save('./lib/dataset/preprocess/init_true_aff_insta_preprocess.npy',insta_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hydro_preprocess.npy',hydro_r)
        np.save('./lib/dataset/preprocess/init_true_aff_charge_preprocess.npy',charge_r)
        np.save('./lib/dataset/preprocess/init_true_aff_performance_preprocess.npy',perf_r)

def get_init_true_aff_no_dupl():
    aff_r = np.load('./lib/dataset/preprocess/init_true_aff_aff_preprocess.npy')
    sol_r = np.load('./lib/dataset/preprocess/init_true_aff_sol_preprocess.npy')
    ppost_r = np.load('./lib/dataset/preprocess/init_true_aff_ppost_preprocess.npy')
    perf_r = np.load('./lib/dataset/preprocess/init_true_aff_performance_preprocess.npy')
    seqs = []
    with open('./lib/dataset/preprocess/init_true_aff_seqs_preprocess.npy','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_true_aff_random():
    muts = []
    with open('./lib/true_aff/mutants_of_best.txt','r') as f:
        for line in f:
            muts.append(line.split('\n')[0])
    return muts

def preprocess_pareto_random_true_aff(oracle):
    seqs = get_true_aff_random()
    seqs = list(set(seqs))
    batch_size = 128
    n_batches = int(len(seqs)/batch_size) + 1
    sol_r = np.array([])
    aff_r = np.array([])
    mu_r = np.array([])
    ppost_r = np.array([])
    perf_r = np.array([])
    insta_r = np.array([])
    hydro_r = np.array([])
    charge_r = np.array([])
    for j in tqdm(range(n_batches)):
        s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r = - a_r
        p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        print(oracle.true_aff_ora.method)
        performance = oracle.true_aff_ora.score_without_noise_simple(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        mu_r = np.concatenate((mu_r,a_r))
        sol_r = np.concatenate((sol_r,s_r))
        aff_r = np.concatenate((aff_r,a_r))
        ppost_r = np.concatenate((ppost_r,p_r))
        insta_r = np.concatenate((insta_r,instability))
        hydro_r = np.concatenate((hydro_r,hydrophobicity))
        charge_r = np.concatenate((charge_r,charge))
        perf_r = np.concatenate((perf_r,performance))
    print(np.min(perf_r))
    with open('./lib/dataset/preprocess/random_true_aff_seqs_preprocess.npy','w') as f:
        for seq in seqs:
            f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/random_true_aff_mu_preprocess.npy',mu_r)
        np.save('./lib/dataset/preprocess/random_true_aff_aff_preprocess.npy',aff_r)
        np.save('./lib/dataset/preprocess/random_true_aff_sol_preprocess.npy',sol_r)
        np.save('./lib/dataset/preprocess/random_true_aff_ppost_preprocess.npy',ppost_r)
        np.save('./lib/dataset/preprocess/random_true_aff_insta_preprocess.npy',insta_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hydro_preprocess.npy',hydro_r)
        np.save('./lib/dataset/preprocess/random_true_aff_charge_preprocess.npy',charge_r)
        np.save('./lib/dataset/preprocess/random_true_aff_performance_preprocess.npy',perf_r)

def get_random_true_aff_no_dupl():
    aff_r = np.load('./lib/dataset/preprocess/random_true_aff_aff_preprocess.npy')
    sol_r = np.load('./lib/dataset/preprocess/random_true_aff_sol_preprocess.npy')
    ppost_r = np.load('./lib/dataset/preprocess/random_true_aff_ppost_preprocess.npy')
    perf_r = np.load('./lib/dataset/preprocess/random_true_aff_performance_preprocess.npy')
    seqs = []
    with open('./lib/dataset/preprocess/random_true_aff_seqs_preprocess.npy','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r



##START OF TRUE_AFF_MCMC

def get_mcmc_true_aff_seqs(weights,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/mcmc/true_aff/mcmc_true_aff_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/mcmc/true_aff/mcmc_true_aff_exp_lim_burn_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
    seqs = list(set(seqs))
    init_seqs,_,_,_,_ = get_init_true_aff_hard_no_dupl()
    seqs = [i for i in seqs if i not in init_seqs]
    return seqs,score

def get_all_mcmc_true_aff_seqs(gw,beta,replicate):
    #weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.125,0.875,gw,beta,1.0],[1.0,0.1,0.9,gw,beta,1.0],[1.0,0.05,0.95,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    mcmc_seqs = []
    mcmc_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_mcmc_true_aff_seqs(w,replicate)
        mcmc_seqs += seqs
        mcmc_score = np.concatenate((mcmc_score,score))
    return mcmc_seqs, mcmc_score

def preprocess_pareto_mcmc_true_aff(oracle):
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for i in global_weights:
        for b in beta:
            seqs,score = get_all_mcmc_true_aff_seqs(i,b,replicate)
            seqs = list(set(seqs))
            n_batches = int(len(seqs)/batch_size) + 1
            sol_r = np.array([])
            aff_r = np.array([])
            mu_r = np.array([])
            ppost_r = np.array([])
            perf_r = np.array([])
            insta_r = np.array([])
            hydro_r = np.array([])
            charge_r = np.array([])
            for j in tqdm(range(n_batches)):
                s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = 12 - a_r
                p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                print(oracle.true_aff_ora.method)
                performance = oracle.true_aff_ora.score_without_noise_simple(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                mu_r = np.concatenate((mu_r,a_r))
                a_r = a_r + b * v_r
                sol_r = np.concatenate((sol_r,s_r))
                aff_r = np.concatenate((aff_r,a_r))
                ppost_r = np.concatenate((ppost_r,p_r))
                insta_r = np.concatenate((insta_r,instability))
                hydro_r = np.concatenate((hydro_r,hydrophobicity))
                charge_r = np.concatenate((charge_r,charge))
                perf_r = np.concatenate((perf_r,performance))
            with open('./lib/dataset/preprocess/mcmc_true_aff_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),'w') as f:
                for seq in seqs:
                    f.write(seq + '\n')
            np.save('./lib/dataset/preprocess/mcmc_true_aff_mu_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),mu_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),aff_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),sol_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),ppost_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),insta_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),hydro_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),charge_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),perf_r)

def get_mcmc_true_aff_no_dupl(g,beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    ppost_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    perf_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    seqs = []
    aff_r = aff_r - 12
    with open('./lib/dataset/preprocess/mcmc_true_aff_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_mcmc_true_aff_dict(g,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_no_dupl(g,beta,replicate)
    d = {}
    for i in range(len(seqs)):
        if seqs[i] not in d:
            d[seqs[i]] = (sol_r[i],aff_r[i],ppost_r[i],perf_r[i])
        else:
            print('error!!!!')
    return d

def get_mcmc_true_aff_error(g,beta,replicate):
    perf_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    mu_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_mu_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    mu_r = mu_r - 12
    return mu_r,perf_r

def get_mcmc_true_aff_pareto_front(gw,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_no_dupl(gw,beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,perf_r

def plot_mcmc_pareto_true_aff_top():
    beta = [1.0,-1.0,0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('MCMC True Aff Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_compare_pareto_temp_rep:{}.png'.format(replicate))

def plot_mcmc_true_aff_developability():
    beta = [0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    all_sol = np.array([])
    all_hum = np.array([])
    all_aff = np.array([])
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_no_dupl(g,b,replicate)
            all_aff = np.concatenate((all_aff,aff_r))
            all_sol = np.concatenate((all_sol,sol_r))
            all_hum = np.concatenate((all_hum,ppost_r))
    x,y,z = get_color_plot(all_sol,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('MCMC True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_developability_hum_sol_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('MCMC True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_developability_hum_aff_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_sol)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('MCMC True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_developability_aff_sol_rep:{}.png'.format(replicate))
    plt.clf()

def find_seqs_in_trust():
    beta = [1.0,0.0,-1.0]
    global_weight = [10.0]
    replicate = 1
    sol_min = 4
    hum_min = -105
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r = get_mcmc_true_aff_no_dupl(g,b,replicate)
            sol_mask = sol_r > sol_min
            hum_mask = ppost_r > hum_min
            aff_r = (aff_r * sol_mask) * hum_mask
            print(np.sum(aff_r > 0))
            print(np.quantile(aff_r,0.9))

def plot_mcmc_pareto_true_aff_density():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_mcmc_true_aff_pareto_front(i,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'MCMC generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel('affinity score')
            axs[0,m].set_ylabel('solubility score')
            axs[0,m].legend()
            axs[0,m].set_title('MCMC/GW:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('affinity score')
            axs[1,m].set_ylabel('true affinity')
            axs[1,m].legend()
            axs[1,m].set_title('MCMC/GW:{}/beta:{}/pear:{}'.format(i,b,pearson))

    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_mcmc_true_aff_density_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))

def plot_mcmc_pareto_true_aff_density_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,5,figsize = (40,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_mcmc_true_aff_pareto_front(i,b,replicate)
            idx = np.argsort(dist)[:1000]
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,4].scatter(all_perf,all_sol,label = r'$\beta$' + ' : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'MCMC generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel(r'$\hat f_{\rm aff}$')
            axs[0,m].set_ylabel(r'$\hat f_{\rm sol}$')
            axs[0,m].set_ylim(1,7)
            if m == 0: 
                axs[0,m].legend(frameon = False)
            axs[0,m].set_title(r'$\beta$ : ' + ' {}'.format(b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('Affinity Score')
            axs[1,m].set_ylabel('True affinity')
            axs[1,m].legend()
            axs[1,m].set_title(r'$\beta$ : ' + r' {} / $\rho$:{}'.format(b,pearson))
    axs[0,4].set_xlabel('True affinity')
    axs[0,4].set_ylabel('Solubility')
    axs[0,4].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_mcmc_true_aff_density_top_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))

def analyze_mcmc_true_aff():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    multiple = 10
    budgets = [20,500]
    fig,axs = plt.subplots(len(budgets),5,figsize = (40,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5,multiple))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_mcmc_true_aff_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    for m in range(multiple):
                        seqs, score = get_mcmc_true_aff_seqs(weights[l],replicate)
                        seqs = list(set(seqs))
                        seqs = random.choices(seqs,k = budget)
                        all_aff = np.zeros(len(seqs))
                        all_sol = np.zeros(len(seqs))
                        all_ppost = np.zeros(len(seqs))
                        all_perf = np.zeros(len(seqs))
                        for j in range(len(seqs)):
                            r = dupl_dict[seqs[j]]
                            all_sol[j] = r[0]
                            all_aff[j] = r[1]
                            all_ppost[j] = r[2]
                            all_perf[j] = r[3]
                        results[i,f,k,l,0,m] = np.mean(all_sol)
                        results[i,f,k,l,1,m] = np.mean(all_perf)
                        results[i,f,k,l,2,m] = np.mean(all_ppost)
                        results[i,f,k,l,3,m] = diversity(seqs)
                        results[i,f,k,l,4,m] = novelty(init,seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                p_mean = np.mean(results[i,:,k,l,1],axis = 1)
                p_var = np.std(results[i,:,k,l,1],axis = 1)
                s_mean = np.mean(results[i,:,k,l,0],axis = 1)
                s_var = np.std(results[i,:,k,l,0],axis = 1)
                axs[i,0].plot(beta,p_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,0].fill_between(beta, (p_mean-2*p_var), (p_mean+2*p_var), alpha=.1)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta, p_var,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,1].set_ylabel('performance standard deviation')
                axs[i,1].set_xlabel('beta')
                axs[i,2].plot(beta,s_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,2].set_ylabel('solubility')
                axs[i,2].set_xlabel('beta')
                axs[i,3].plot(beta,np.mean(results[i,:,k,l,3],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,3].set_ylabel('diversity')
                axs[i,3].set_xlabel('beta')
                axs[i,4].plot(beta,np.mean(results[i,:,k,l,4],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,4].set_ylabel('novelty')
                axs[i,4].set_xlabel('beta')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_analyze.png')

def analyze_mcmc_true_aff_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    budgets = [5,500]
    fig,axs = plt.subplots(len(budgets),4,figsize = (32,8*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,10))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_mcmc_true_aff_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    seqs, score = get_mcmc_true_aff_seqs(weights[l],replicate)
                    seqs = list(set(seqs))
                    all_aff = np.zeros(len(seqs))
                    all_sol = np.zeros(len(seqs))
                    all_ppost = np.zeros(len(seqs))
                    all_perf = np.zeros(len(seqs))
                    for j in range(len(seqs)):
                        r = dupl_dict[seqs[j]]
                        all_sol[j] = r[0]
                        all_aff[j] = r[1]
                        all_ppost[j] = r[2]
                        all_perf[j] = r[3]
                    aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist = remove_dominated(all_aff,all_sol)
                    dist = np.array(dist)
                    top_idx = np.argsort(dist)[:budget]

                    top_aff = all_aff[top_idx]
                    top_sol = all_sol[top_idx]
                    top_ppost = all_ppost[top_idx]
                    top_perf = all_perf[top_idx]
                    top_seqs = [seqs[i] for i in top_idx]
                    div_mean, div_std = diversity(top_seqs,return_std = True)
                    nov_mean, nov_std = novelty(init,top_seqs,return_std = True)
                    results[i,f,k,l,0] = np.mean(top_sol)
                    results[i,f,k,l,1] = np.mean(top_perf)
                    results[i,f,k,l,2] = np.mean(top_ppost)
                    results[i,f,k,l,3] = div_mean
                    results[i,f,k,l,4] = nov_mean
                    results[i,f,k,l,5] = np.std(top_sol)
                    results[i,f,k,l,6] = np.std(top_perf)
                    results[i,f,k,l,7] = np.std(top_ppost)
                    results[i,f,k,l,8] = div_std
                    results[i,f,k,l,9] = nov_std

    beta = np.array(beta)
    methods_shift = [-0.125,0.125]
    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                axs[i,0].errorbar(beta + methods_shift[l],results[i,:,k,l,1],yerr = results[i,:,k,l,6],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,0].bar(beta + methods_shift[l],results[i,:,k,l,1],label = r'budget:{} / $aff_w$:{}'.format(budgets[i],aff_w[l]),width = 0.25)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel(r'$\beta$')
                axs[i,0].set_xticks(beta)
                axs[i,0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
                axs[i,1].errorbar(beta + methods_shift[l],results[i,:,k,l,0],yerr = results[i,:,k,l,5],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,1].bar(beta + methods_shift[l],results[i,:,k,l,0],label = r'budget:{} / $aff_w$:{}'.format(budgets[i],aff_w[l]),width = 0.25)
                axs[i,1].set_ylabel('solubility')
                axs[i,1].set_xlabel(r'$\beta$')
                axs[i,1].set_xticks(beta)
                axs[i,2].errorbar(beta + methods_shift[l],results[i,:,k,l,3],yerr = results[i,:,k,l,8],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,2].bar(beta + methods_shift[l],results[i,:,k,l,3],label = r'$w$' + ': {}'.format(aff_w[l]),width = 0.25)
                axs[i,2].set_ylabel('Diversity')
                axs[i,2].set_xlabel(r'$\beta$')
                axs[i,2].set_xticks(beta)
                axs[i,2].set_ylim(0,7)
                axs[i,2].legend(loc='lower center', bbox_to_anchor=(0.15, 0.85), frameon = False)
                axs[i,3].errorbar(beta + methods_shift[l],results[i,:,k,l,4],yerr = results[i,:,k,l,9],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,3].bar(beta + methods_shift[l],results[i,:,k,l,4],label = r'$w$:' + ' {}'.format(aff_w[l]),width = 0.25)
                axs[i,3].set_ylabel('Novelty')
                axs[i,3].set_xlabel(r'$\beta$')
                axs[i,3].set_xticks(beta)
                #axs[i,3].legend(loc='lower center', bbox_to_anchor=(0.25, 0.8), frameon = False)
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_analyze_top.png')

def filter_seqs(sol_threshold,aff_threshold,aff,sol):
    idx = [i for i in range(len(sol)) if sol[i] >= sol_threshold and aff[i] <= aff_threshold]
    aff = aff[idx]
    sol = sol[idx]
    return aff,sol

def filter_seqs_sol(sol_threshold,aff,sol,perf):
    idx = [i for i in range(len(sol)) if sol[i] >= sol_threshold]
    aff = aff[idx]
    sol = sol[idx]
    perf = perf[idx]
    return aff,sol,perf

def perc_filter_seqs(sol_threshold,aff_threshold,aff,sol):
    if len(aff) > 0:
        idx = [i for i in range(len(sol)) if sol[i] >= sol_threshold and aff[i] <= aff_threshold]
        return len(idx)
    else:
        return 0

def true_aff_top_box_plots():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    aff_w = [0.85,1.0]
    sol_threshold = [-1.0,4.0,5.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,7,figsize = (56,9))
    budget = 20
    init_seqs,aff_r,sol_r,ppost_r,perf_r = get_init_true_aff_no_dupl()
    top_idx = np.argsort(perf_r)[:budget]
    perf_r = perf_r[top_idx]
    sol_r = sol_r[top_idx]
    
    print(np.min(perf_r))
    print(np.max(perf_r))

    aff_threshold = np.linspace(1.0,-4)
    perc_better = np.zeros((len(sol_threshold),len(aff_threshold),len(beta),len(aff_w)))
    perc_better_init = np.zeros((len(sol_threshold),len(aff_threshold)))

    labels = []
    perfs = []
    sols = []
    for t in range(len(sol_threshold)):
        sol_t = sol_threshold[t]
        for t2 in range(len(aff_threshold)):
            aff_t = aff_threshold[t2]
            perc_better_init[t,t2] = perc_filter_seqs(sol_t,aff_t,perf_r,sol_r)
    labels.append('initial_mutants')
    perfs.append(perf_r)
    sols.append(sol_r)

    count = 1
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            dupl_dict = get_mcmc_true_aff_dict(gw,b,replicate)
            weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for l in range(len(weights)):
                seqs, score = get_mcmc_true_aff_seqs(weights[l],replicate)
                seqs = list(set(seqs))
                seqs = [i for i in seqs if i not in init_seqs]
                all_sol = np.zeros(len(seqs))
                all_aff = np.zeros(len(seqs))
                all_perf = np.zeros(len(seqs))
                for j in range(len(seqs)):
                    r = dupl_dict[seqs[j]]
                    all_sol[j] = r[0]
                    all_aff[j] = r[1]
                    all_perf[j] = r[3]
                perfs.append(all_perf)
                sols.append(all_sol)
                for t in range(len(sol_threshold)):
                    sol_t = sol_threshold[t]
                    aff, sol, perf = filter_seqs_sol(sol_t,all_aff,all_sol,all_perf)
                    top_idx = np.argsort(aff)[-budget:]
                    top_perf = perf[top_idx]
                    top_sol = sol[top_idx]
                    for t2 in range(len(aff_threshold)):
                        aff_t = aff_threshold[t2]
                        perc = perc_filter_seqs(sol_t,aff_t,top_perf,top_sol)
                        perc_better[t,t2,f,l] = perc
                count += 1
                print(count)
                labels.append(r'$\beta$ : ' + '{}'.format(b) + r'/ $method$ : ' + '{}'.format(aff_w[l]))

    perfs = np.array(perfs)
    axs[0].boxplot(-perfs,labels = labels)
    axs[0].set_xticklabels(labels,rotation = 90)
    axs[0].set_ylabel(r'$f^{min}_{aff}$')
    axs[1].boxplot(sols,labels = labels)
    axs[1].set_xticklabels(labels,rotation = 90)
    axs[1].set_ylabel('Solubility')
    axs[2].violinplot(-perfs)
    axs[2].set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
    axs[2].set_ylabel(r'$log_{10}$ (True Affinity (nM))')
    axs[3].violinplot(sols)
    axs[3].set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation = 90)
    axs[3].set_ylabel('Solubility')

    #perc_better = perc_better.reshape((len(sol_threshold),len(aff_threshold),-1))

    beta_color = ['b','y','r','g']
    aff_fmt = ['--','-']
    legend_elements = [Line2D([0], [0], linestyle = '-', color='pink',lw = 1, label=r'initial'),
                        Line2D([0], [0], linestyle = '--', color='k', label='w_aff = 0.85'),
                        Line2D([0], [0], linestyle = '-', color='k', label='w_aff = 1.0'),
                        Line2D([0], [0], linestyle = '-', color='b',lw = 1, label=r'$\beta$ = -1'),
                        Line2D([0], [0], linestyle = '-', color='y',lw = 1, label=r'$\beta$ = 0'),
                        Line2D([0], [0], linestyle = '-', color='r',lw = 1, label=r'$\beta$ = 1'),
                        Line2D([0], [0], linestyle = '-', color='g',lw = 1, label=r'$\beta$ = 2')]
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(aff_w)):
            m = aff_w[j]
            fmt = aff_fmt[j]
            color = beta_color[i]
            area = simpson(-perc_better[0,:,i,j], x = aff_threshold)
            axs[4].plot(-aff_threshold,perc_better[0,:,i,j],fmt,c = color)
    axs[4].plot(-aff_threshold,perc_better_init[0],c = 'pink')
    axs[4].set_ylabel('Number of Sequences above thresholds')
    axs[4].set_xlabel(r'$f^{min}_{aff}$',fontsize = 18)
    axs[4].set_title('No solubility threshold',fontsize = 18)
    axs[4].legend(frameon = False,handles=legend_elements)

    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(aff_w)):
            m = aff_w[j]
            fmt = aff_fmt[j]
            color = beta_color[i]
            area = simpson(-perc_better[1,:,i,j], x = aff_threshold)
            axs[5].plot(-aff_threshold,perc_better[1,:,i,j],fmt, c = color)
    axs[5].plot(-aff_threshold,perc_better_init[1],c = 'pink')
    axs[5].set_ylabel('Number of Sequences above thresholds')
    axs[5].set_xlabel(r'$f^{min}_{aff}$', fontsize = 18)
    axs[5].set_title(r'$f^{min}_{sol}$ = 4.0', fontsize = 18)

    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(aff_w)):
            m = aff_w[j]
            fmt = aff_fmt[j]
            color = beta_color[i]
            area = simpson(-perc_better[2,:,i,j], x = aff_threshold)
            axs[6].plot(-aff_threshold,perc_better[2,:,i,j],fmt, c = color)
    axs[6].plot(-aff_threshold,perc_better_init[1],c = 'grey')
    axs[6].set_ylabel('Number of Sequences above thresholds')
    axs[6].set_xlabel(r'$log_{10}$ ($T_{AFF}$ (nM))')
    axs[6].set_title('Solubility threshold = 5.0')
    axs[6].legend()

    plt.tight_layout()
    plt.savefig('./pareto_final/box_plots_true_aff_top_budget:{}.png'.format(budget))

def true_aff_top_plot_error():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    replicate = 1
    labels = []
    errors = []
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            mu_r,perf_r =  get_mcmc_true_aff_error(gw,b,replicate)
            errors.append(perf_r - mu_r)
            labels.append(r'$\beta$ = {}'.format(b))
    plt.violinplot(errors,showmeans = True)
    plt.xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
    plt.ylabel(r'Error distribution')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_error_distribution.png')

##END OF TRUE AFF MCMC
##START OF TRUE AFF GFLOW

def get_gflow_true_aff_seqs(global_weight,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/gflownet/true_aff/true_aff_gflow_sol:{}_aff:{}_global:{}_beta:{}_rep:{}.txt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/gflownet/true_aff/true_aff_gflow_scores_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate))
    seqs = list(set(seqs))
    init_seqs,_,_,_,_ = get_init_true_aff_no_dupl()
    seqs = [i for i in seqs if i not in init_seqs]
    return seqs,score

def get_all_gflow_true_aff_seqs(gw,beta,replicate):
    #weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.125,0.875,gw,beta,1.0],[1.0,0.1,0.9,gw,beta,1.0],[1.0,0.05,0.95,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    gflow_seqs = []
    gflow_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_gflow_true_aff_seqs(w,replicate)
        gflow_seqs += seqs
        gflow_score = np.concatenate((gflow_score,score))
    return gflow_seqs, gflow_score

def preprocess_pareto_gflow_true_aff(oracle):
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for i in global_weights:
        for b in beta:
            seqs,score = get_all_gflow_true_aff_seqs(i,b,replicate)
            seqs = list(set(seqs))
            n_batches = int(len(seqs)/batch_size) + 1
            sol_r = np.array([])
            aff_r = np.array([])
            ppost_r = np.array([])
            perf_r = np.array([])
            insta_r = np.array([])
            hydro_r = np.array([])
            charge_r = np.array([])
            for j in tqdm(range(n_batches)):
                s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = 12 - a_r
                p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                performance = oracle.true_aff_ora.score_without_noise_simple(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = a_r + b * v_r
                sol_r = np.concatenate((sol_r,s_r))
                aff_r = np.concatenate((aff_r,a_r))
                ppost_r = np.concatenate((ppost_r,p_r))
                insta_r = np.concatenate((insta_r,instability))
                hydro_r = np.concatenate((hydro_r,hydrophobicity))
                charge_r = np.concatenate((charge_r,charge))
                perf_r = np.concatenate((perf_r,performance))
            with open('./lib/dataset/preprocess/gflow_true_aff_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),'w') as f:
                for seq in seqs:
                    f.write(seq + '\n')
            np.save('./lib/dataset/preprocess/gflow_true_aff_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),aff_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),sol_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),ppost_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),insta_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),hydro_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),charge_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),perf_r)


def get_gflow_true_aff_no_dupl(g,beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/gflow_true_aff_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/gflow_true_aff_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    ppost_r = np.load('./lib/dataset/preprocess/gflow_true_aff_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    perf_r = np.load('./lib/dataset/preprocess/gflow_true_aff_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    seqs = []
    with open('./lib/dataset/preprocess/gflow_true_aff_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_gflow_true_aff_dict(g,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_gflow_true_aff_no_dupl(g,beta,replicate)
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i],ppost_r[i],perf_r[i])
    return d

def get_gflow_true_aff_pareto_front(gw,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_gflow_true_aff_no_dupl(gw,beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,perf_r

def plot_gflow_pareto_true_aff_top():
    beta = [1.0,-1.0,0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r = get_gflow_true_aff_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('GFLOW True Aff Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_compare_pareto_temp_rep:{}.png'.format(replicate))


def plot_gflow_true_aff_developability():
    beta = [0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    all_sol = np.array([])
    all_hum = np.array([])
    all_aff = np.array([])
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r = get_gflow_true_aff_no_dupl(g,b,replicate)
            all_aff = np.concatenate((all_aff,aff_r))
            all_sol = np.concatenate((all_sol,sol_r))
            all_hum = np.concatenate((all_hum,ppost_r))
    x,y,z = get_color_plot(all_sol,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('GFLOW True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_developability_hum_sol_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('GFLOW True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_developability_hum_aff_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_sol)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('GFLOW True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_developability_aff_sol_rep:{}.png'.format(replicate))
    plt.clf()

def plot_gflow_pareto_true_aff_density():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_gflow_true_aff_pareto_front(i,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'gflow generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel('affinity score')
            axs[0,m].set_ylabel('solubility score')
            axs[0,m].legend()
            axs[0,m].set_title('gflow/GW:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('affinity score')
            axs[1,m].set_ylabel('true affinity')
            axs[1,m].legend()
            axs[1,m].set_title('gflow/GW:{}/beta:{}/pear:{}'.format(i,b,pearson))

    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_true_aff_density_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))

def plot_gflow_pareto_true_aff_density_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_gflow_true_aff_pareto_front(i,b,replicate)
            idx = np.argsort(dist)[:1000]
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'gflow generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel('affinity score')
            axs[0,m].set_ylabel('solubility score')
            axs[0,m].legend()
            axs[0,m].set_title('gflow/GW:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('affinity score')
            axs[1,m].set_ylabel('true affinity')
            axs[1,m].legend()
            axs[1,m].set_title('gflow/GW:{}/beta:{}/pear:{}'.format(i,b,pearson))
    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_true_aff_density_top_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))


def analyze_gflow_true_aff():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    multiple = 10
    budgets = [20,500]
    fig,axs = plt.subplots(len(budgets),5,figsize = (40,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5,multiple))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_gflow_true_aff_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    for m in range(multiple):
                        seqs, score = get_gflow_true_aff_seqs(weights[l],replicate)
                        seqs = list(set(seqs))
                        seqs = random.choices(seqs,k = budget)
                        all_aff = np.zeros(len(seqs))
                        all_sol = np.zeros(len(seqs))
                        all_ppost = np.zeros(len(seqs))
                        all_perf = np.zeros(len(seqs))
                        for j in range(len(seqs)):
                            r = dupl_dict[seqs[j]]
                            all_sol[j] = r[0]
                            all_aff[j] = r[1]
                            all_ppost[j] = r[2]
                            all_perf[j] = r[3]
                        results[i,f,k,l,0,m] = np.mean(all_sol)
                        results[i,f,k,l,1,m] = np.mean(all_perf)
                        results[i,f,k,l,2,m] = np.mean(all_ppost)
                        results[i,f,k,l,3,m] = diversity(seqs)
                        results[i,f,k,l,4,m] = novelty(init,seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                p_mean = np.mean(results[i,:,k,l,1],axis = 1)
                p_var = np.std(results[i,:,k,l,1],axis = 1)
                s_mean = np.mean(results[i,:,k,l,0],axis = 1)
                s_var = np.std(results[i,:,k,l,0],axis = 1)
                axs[i,0].plot(beta,p_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,0].fill_between(beta, (p_mean-2*p_var), (p_mean+2*p_var), alpha=.1)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta, p_var,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,1].set_ylabel('performance standard deviation')
                axs[i,1].set_xlabel('beta')
                axs[i,2].plot(beta,s_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,2].set_ylabel('solubility')
                axs[i,2].set_xlabel('beta')
                axs[i,3].plot(beta,np.mean(results[i,:,k,l,3],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,3].set_ylabel('diversity')
                axs[i,3].set_xlabel('beta')
                axs[i,4].plot(beta,np.mean(results[i,:,k,l,4],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,4].set_ylabel('novelty')
                axs[i,4].set_xlabel('beta')
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_analyze.png')

def analyze_gflow_true_aff_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    budgets = [20,500]
    fig,axs = plt.subplots(len(budgets),4,figsize = (32,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_gflow_true_aff_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    seqs, score = get_gflow_true_aff_seqs(weights[l],replicate)
                    seqs = list(set(seqs))
                    all_aff = np.zeros(len(seqs))
                    all_sol = np.zeros(len(seqs))
                    all_ppost = np.zeros(len(seqs))
                    all_perf = np.zeros(len(seqs))
                    for j in range(len(seqs)):
                        r = dupl_dict[seqs[j]]
                        all_sol[j] = r[0]
                        all_aff[j] = r[1]
                        all_ppost[j] = r[2]
                        all_perf[j] = r[3]
                    aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist = remove_dominated(all_aff,all_sol)
                    dist = np.array(dist)
                    top_idx = np.argsort(dist)[:budget]

                    top_aff = all_aff[top_idx]
                    top_sol = all_sol[top_idx]
                    top_ppost = all_ppost[top_idx]
                    top_perf = all_perf[top_idx]
                    top_seqs = [seqs[i] for i in top_idx]
                    results[i,f,k,l,0] = np.mean(top_sol)
                    results[i,f,k,l,1] = np.mean(top_perf)
                    results[i,f,k,l,2] = np.mean(top_ppost)
                    results[i,f,k,l,3] = diversity(top_seqs)
                    results[i,f,k,l,4] = novelty(init,top_seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                axs[i,0].plot(beta,results[i,:,k,l,1],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta,results[i,:,k,l,0],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,1].set_ylabel('solubility')
                axs[i,1].set_xlabel('beta')
                axs[i,2].plot(beta,results[i,:,k,l,3],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,2].set_ylabel('diversity')
                axs[i,2].set_xlabel('beta')
                axs[i,3].plot(beta,results[i,:,k,l,4],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,3].set_ylabel('novelty')
                axs[i,3].set_xlabel('beta')
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_analyze_top.png')

##END OF TRUE AFF GFLOW
##START OF ANTBO TRUE AFF

def get_antBO_true_aff_seqs(weights,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_burn_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/antBO/true_aff/antBO_true_aff_exp_lim_burn_prob_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt.npy'.format(weights[0],weights[1],weights[2],replicate))
    seqs = list(set(seqs))
    init_seqs,_,_,_,_ = get_init_true_aff_no_dupl()
    seqs = [i for i in seqs if i not in init_seqs]
    return seqs,score

def get_all_antBO_true_aff_seqs(beta,replicate):
    weights = [[-10.0,-120.0,beta],[4.0,-120.0,beta]]
    mcmc_seqs = []
    mcmc_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_antBO_true_aff_seqs(w,replicate)
        mcmc_seqs += seqs
        mcmc_score = np.concatenate((mcmc_score,score))
    return mcmc_seqs, mcmc_score

def preprocess_pareto_antBO_true_aff(oracle):
    beta = [0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for b in beta:
        seqs,score = get_all_antBO_true_aff_seqs(b,replicate)
        print(len(seqs))
        seqs = list(set(seqs))
        print(len(seqs))
        n_batches = int(len(seqs)/batch_size) + 1
        sol_r = np.array([])
        aff_r = np.array([])
        ppost_r = np.array([])
        insta_r = np.array([])
        hydro_r = np.array([])
        charge_r = np.array([])
        perf_r = np.array([])
        for j in tqdm(range(n_batches)):
            a_r,valid,s_r,p_r = oracle.trust_region_true_aff(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            performance = oracle.true_aff_ora.score_without_noise_simple(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            sol_r = np.concatenate((sol_r,s_r))
            aff_r = np.concatenate((aff_r,a_r))
            ppost_r = np.concatenate((ppost_r,p_r))
            insta_r = np.concatenate((insta_r,instability))
            hydro_r = np.concatenate((hydro_r,hydrophobicity))
            charge_r = np.concatenate((charge_r,charge))
            perf_r = np.concatenate((perf_r,performance))
        with open('./lib/dataset/preprocess/antBO_true_aff_seqs_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),'w') as f:
            for seq in seqs:
                f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/antBO_true_aff_aff_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),aff_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_sol_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),sol_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_ppost_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),ppost_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_insta_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),insta_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hydro_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),hydro_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_charge_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),charge_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_performance_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),perf_r)

def get_antBO_true_aff_no_dupl(beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/antBO_true_aff_aff_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/antBO_true_aff_sol_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    ppost_r = np.load('./lib/dataset/preprocess/antBO_true_aff_ppost_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    perf_r = np.load('./lib/dataset/preprocess/antBO_true_aff_performance_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    seqs = []
    with open('./lib/dataset/preprocess/antBO_true_aff_seqs_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_antBO_true_aff_dict(beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_antBO_true_aff_no_dupl(beta,replicate)
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i],ppost_r[i],perf_r[i])
    return d

def get_antBO_true_aff_pareto_front(beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_antBO_true_aff_no_dupl(beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,perf_r


def plot_antBO_pareto_true_aff_top():
    beta = [0.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        seqs,aff_r,sol_r,ppost_r = get_antBO_true_aff_no_dupl(b,replicate)
        pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
        plt.scatter(pareto_aff,pareto_sol,label = 'beta:{}'.format(b))
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('antBO True Aff Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/antBO_true_aff_compare_pareto_temp_rep:{}.png'.format(replicate))

def plot_antBO_pareto_true_aff_density():
    global_weights = [10.0]
    beta = [0.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_antBO_true_aff_pareto_front(b,replicate)
        idx = random.choices(range(len(seqs)),k = 20)
        seqs = [seqs[i] for i in idx]
        all_aff = all_aff[idx]
        all_sol = all_sol[idx]
        all_perf = all_perf[idx]
        axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
        x,y,z = get_color_plot(all_aff,all_sol)
        z = axs[0,m].scatter(x,y,c = z,label = 'antBO generated seqs')
        plt.colorbar(z,ax = axs[0,m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        axs[0,m].set_xlabel('affinity score')
        axs[0,m].set_ylabel('solubility score')
        axs[0,m].legend()
        axs[0,m].set_title('antBO/beta:{}'.format(b))
        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
        axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
        axs[1,m].set_xlabel('affinity score')
        axs[1,m].set_ylabel('true affinity')
        axs[1,m].legend()
        axs[1,m].set_title('antBO/beta:{}/pear:{}'.format(b,pearson))

    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_antBO_true_aff_density_initial:{}_rep:{}.png'.format(add_initial,replicate))

def plot_antBO_pareto_true_aff_density_top():
    global_weights = [10.0]
    beta = [0.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_antBO_true_aff_pareto_front(b,replicate)
        idx = np.argsort(dist)[:1000]
        seqs = [seqs[i] for i in idx]
        all_aff = all_aff[idx]
        all_sol = all_sol[idx]
        all_perf = all_perf[idx]
        axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
        x,y,z = get_color_plot(all_aff,all_sol)
        z = axs[0,m].scatter(x,y,c = z,label = 'antBO generated seqs')
        plt.colorbar(z,ax = axs[0,m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        axs[0,m].set_xlabel('affinity score')
        axs[0,m].set_ylabel('solubility score')
        axs[0,m].legend()
        axs[0,m].set_title('antBO/beta:{}'.format(b))
        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
        axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
        axs[1,m].set_xlabel('affinity score')
        axs[1,m].set_ylabel('true affinity')
        axs[1,m].legend()
        axs[1,m].set_title('antBO/beta:{}/pear:{}'.format(b,pearson))
    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_antBO_true_aff_density_top_initial:{}_rep:{}.png'.format(add_initial,replicate))


def analyze_antBO_true_aff():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    sol_min = [3.0,4.0]
    add_initial = False
    replicate = 1
    multiple = 10
    budgets = [10,20]
    fig,axs = plt.subplots(len(budgets),5,figsize = (40,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5,multiple))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_antBO_true_aff_dict(b,replicate)
                weights = [[3.0,-105.0,b],[4.0,-105.0,b]]
                for l in range(len(weights)):
                    for m in range(multiple):
                        seqs, score = get_antBO_true_aff_seqs(weights[l],replicate)
                        seqs = list(set(seqs))
                        seqs = random.choices(seqs,k = budget)
                        all_aff = np.zeros(len(seqs))
                        all_sol = np.zeros(len(seqs))
                        all_ppost = np.zeros(len(seqs))
                        all_perf = np.zeros(len(seqs))
                        for j in range(len(seqs)):
                            r = dupl_dict[seqs[j]]
                            all_sol[j] = r[0]
                            all_aff[j] = r[1]
                            all_ppost[j] = r[2]
                            all_perf[j] = r[3]
                        results[i,f,k,l,0,m] = np.mean(all_sol)
                        results[i,f,k,l,1,m] = np.mean(all_perf)
                        results[i,f,k,l,2,m] = np.mean(all_ppost)
                        results[i,f,k,l,3,m] = diversity(seqs)
                        results[i,f,k,l,4,m] = novelty(init,seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(sol_min)):
                p_mean = np.mean(results[i,:,k,l,1],axis = 1)
                p_var = np.std(results[i,:,k,l,1],axis = 1)
                s_mean = np.mean(results[i,:,k,l,0],axis = 1)
                s_var = np.std(results[i,:,k,l,0],axis = 1)
                axs[i,0].plot(beta,p_mean,label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,0].fill_between(beta, (p_mean-2*p_var), (p_mean+2*p_var), alpha=.1)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta, p_var,label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,1].set_ylabel('performance standard deviation')
                axs[i,1].set_xlabel('beta')
                axs[i,1].legend()
                axs[i,2].plot(beta,s_mean,label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,2].set_ylabel('solubility')
                axs[i,2].set_xlabel('beta')
                axs[i,2].legend()
                axs[i,3].plot(beta,np.mean(results[i,:,k,l,3],axis = 1),label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,3].set_ylabel('diversity')
                axs[i,3].set_xlabel('beta')
                axs[i,3].legend()
                axs[i,4].plot(beta,np.mean(results[i,:,k,l,4],axis = 1),label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,4].set_ylabel('novelty')
                axs[i,4].set_xlabel('beta')
                axs[i,4].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/antBO_true_aff_analyze.png')

def analyze_antBO_true_aff_top():
    global_weights = [10.0]
    beta = [0.0]
    sol_min = [-10.0,4.0]
    add_initial = False
    replicate = 1
    budgets = [5,500]
    fig,axs = plt.subplots(len(budgets),4,figsize = (32,8*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,10))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_antBO_true_aff_dict(b,replicate)
                weights = [[-10.0,-120.0,b],[4.0,-120.0,b]]
                for l in range(len(weights)):
                    seqs, score = get_antBO_true_aff_seqs(weights[l],replicate)
                    seqs = list(set(seqs))
                    all_aff = np.zeros(len(seqs))
                    all_sol = np.zeros(len(seqs))
                    all_ppost = np.zeros(len(seqs))
                    all_perf = np.zeros(len(seqs))
                    for j in range(len(seqs)):
                        r = dupl_dict[seqs[j]]
                        all_sol[j] = r[0]
                        all_aff[j] = r[1]
                        all_ppost[j] = r[2]
                        all_perf[j] = r[3]
                    aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist = remove_dominated(all_aff,all_sol)
                    dist = np.array(dist)
                    top_idx = np.argsort(dist)[:budget]

                    top_aff = all_aff[top_idx]
                    top_sol = all_sol[top_idx]
                    top_ppost = all_ppost[top_idx]
                    top_perf = all_perf[top_idx]
                    top_seqs = [seqs[i] for i in top_idx]
                    div_mean, div_std = diversity(top_seqs,return_std = True)
                    nov_mean, nov_std = novelty(init,top_seqs,return_std = True)
                    results[i,f,k,l,0] = np.mean(top_sol)
                    results[i,f,k,l,1] = np.mean(top_perf)
                    results[i,f,k,l,2] = np.mean(top_ppost)
                    results[i,f,k,l,3] = div_mean
                    results[i,f,k,l,4] = nov_mean
                    results[i,f,k,l,5] = np.std(top_sol)
                    results[i,f,k,l,6] = np.std(top_perf)
                    results[i,f,k,l,7] = np.std(top_ppost)
                    results[i,f,k,l,8] = div_std
                    results[i,f,k,l,9] = nov_std

    beta = np.array(beta)
    methods_shift = [-0.125,0.125]
    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(sol_min)):
                axs[i,0].errorbar(beta + methods_shift[l],results[i,:,k,l,1],yerr = results[i,:,k,l,6],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,0].bar(beta + methods_shift[l],results[i,:,k,l,1],label = r'budget:{} / $min sol$:{}'.format(budgets[i],sol_min[l]),width = 0.25)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel(r'$\beta$')
                axs[i,0].set_xticks(beta)
                axs[i,0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
                axs[i,1].errorbar(beta + methods_shift[l],results[i,:,k,l,0],yerr = results[i,:,k,l,5],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,1].bar(beta + methods_shift[l],results[i,:,k,l,0],label = r'budget:{} / $min sol$:{}'.format(budgets[i],sol_min[l]),width = 0.25)
                axs[i,1].set_ylabel('solubility')
                axs[i,1].set_xlabel(r'$\beta$')
                axs[i,1].set_xticks(beta)
                axs[i,2].errorbar(beta + methods_shift[l],results[i,:,k,l,3],yerr = results[i,:,k,l,8],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,2].bar(beta + methods_shift[l],results[i,:,k,l,3],label = r'$min sol$' + ': {}'.format(sol_min[l]),width = 0.25)
                axs[i,2].set_ylabel('Diversity')
                axs[i,2].set_xlabel(r'$\beta$')
                axs[i,2].set_xticks(beta)
                axs[i,2].set_ylim(0,10)
                axs[i,2].legend(loc='lower center', bbox_to_anchor=(0.15, 0.85), frameon = False)
                axs[i,3].errorbar(beta + methods_shift[l],results[i,:,k,l,4],yerr = results[i,:,k,l,9],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,3].bar(beta + methods_shift[l],results[i,:,k,l,4],label = r'$min sol$:' + ' {}'.format(sol_min[l]),width = 0.25)
                axs[i,3].set_ylabel('Novelty')
                axs[i,3].set_xlabel(r'$\beta$')
                axs[i,3].set_xticks(beta)
                #axs[i,3].legend(loc='lower center', bbox_to_anchor=(0.25, 0.8), frameon = False)
    plt.tight_layout()
    plt.savefig('./pareto_final/antBO_true_aff_analyze_top.png')

## END OF ANTBO TRUE AFF
## START OF COMPARE TRUE AFF

def compare_true_aff_methods():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    budgets = [20,500]
    replicate = 1
    multiple = 1
    results = np.zeros((len(global_weights),len(beta),len(budgets),3,4,multiple))
    init = get_true_aff_initial()
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(global_weights)):
            gw = global_weights[j]
            mcmc_seqs,mcmc_pareto_aff,mcmc_pareto_sol,mcmc_all_aff,mcmc_all_sol,mcmc_dist,mcmc_all_perf = get_mcmc_true_aff_pareto_front(gw,b,replicate)
            gflow_seqs,gflow_pareto_aff,gflow_pareto_sol,gflow_all_aff,gflow_all_sol,gflow_dist,gflow_all_perf = get_gflow_true_aff_pareto_front(gw,b,replicate)
            antBO_seqs,antBO_pareto_aff,antBO_pareto_sol,antBO_all_aff,antBO_all_sol,antBO_dist,antBO_all_perf = get_antBO_true_aff_pareto_front(b,replicate)
            for k in range(len(budgets)):
                budget = budgets[k]
                for n in range(multiple):
                    idx = random.choices(range(len(mcmc_seqs)),k = budget)
                    seqs = [mcmc_seqs[i] for i in idx]
                    sol = mcmc_all_sol[idx]
                    perf = mcmc_all_perf[idx]
                    results[j,i,k,0,0,n] = np.mean(perf)
                    results[j,i,k,0,1,n] = np.mean(sol)
                    results[j,i,k,0,2,n] = diversity(mcmc_seqs)
                    results[j,i,k,0,3,n] = novelty(init,mcmc_seqs)

                    idx = random.choices(range(len(gflow_seqs)),k = budget)
                    seqs = [gflow_seqs[i] for i in idx]
                    sol = gflow_all_sol[idx]
                    perf = gflow_all_perf[idx]
                    results[j,i,k,1,0,n] = np.mean(perf)
                    results[j,i,k,1,1,n] = np.mean(sol)
                    results[j,i,k,1,2,n] = diversity(gflow_seqs)
                    results[j,i,k,1,3,n] = novelty(init,gflow_seqs)

                    idx = random.choices(range(len(antBO_seqs)),k = budget)
                    seqs = [antBO_seqs[i] for i in idx]
                    sol = antBO_all_sol[idx]
                    perf = antBO_all_perf[idx]
                    results[j,i,k,2,0,n] = np.mean(perf)
                    results[j,i,k,2,1,n] = np.mean(sol)
                    results[j,i,k,2,2,n] = diversity(antBO_seqs)
                    results[j,i,k,2,3,n] = novelty(init,antBO_seqs)
    
    methods = ['mcmc','gflownet','antBO']
    task = ['performance','solubility','diversity','novelty']
    fig,axs = plt.subplots(len(task),len(budgets),figsize = (16,24))
    for i in range(len(global_weights)):
        for j in range(len(budgets)):
            for t in range(len(task)):
                axs[t,j].set_xlabel('beta')
                axs[t,j].set_ylabel(task[t])
                axs[t,j].set_title('budget:{}'.format(budgets[j]))
                for k in range(len(methods)):
                    axs[t,j].plot(beta,np.mean(results[i,:,j,k,t],axis = 1),label = methods[k])
    plt.legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_methods_true_aff.png')


def compare_true_aff_methods_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    budgets = [5,500]
    replicate = 1
    multiple = 10
    results = np.zeros((len(global_weights),len(beta),len(budgets),3,8))
    init = get_true_aff_initial()
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(global_weights)):
            gw = global_weights[j]
            mcmc_seqs,mcmc_pareto_aff,mcmc_pareto_sol,mcmc_all_aff,mcmc_all_sol,mcmc_dist,mcmc_all_perf = get_mcmc_true_aff_pareto_front(gw,b,replicate)
            gflow_seqs,gflow_pareto_aff,gflow_pareto_sol,gflow_all_aff,gflow_all_sol,gflow_dist,gflow_all_perf = get_gflow_true_aff_pareto_front(gw,b,replicate)
            antBO_seqs,antBO_pareto_aff,antBO_pareto_sol,antBO_all_aff,antBO_all_sol,antBO_dist,antBO_all_perf = get_antBO_true_aff_pareto_front(b,replicate)
            for k in range(len(budgets)):
                budget = budgets[k]
                idx = np.argsort(mcmc_dist)[:budget]
                seqs = [mcmc_seqs[i] for i in idx]
                sol = mcmc_all_sol[idx]
                perf = mcmc_all_perf[idx]
                results[j,i,k,0,0] = np.mean(perf)
                results[j,i,k,0,1] = np.mean(sol)
                results[j,i,k,0,2] = diversity(seqs)
                results[j,i,k,0,3] = novelty(init,seqs)
                results[j,i,k,0,4] = np.std(perf)
                results[j,i,k,0,5] = np.std(sol)
                results[j,i,k,0,6] = 0
                results[j,i,k,0,7] = 0

                idx = np.argsort(gflow_dist)[:budget]
                seqs = [gflow_seqs[i] for i in idx]
                sol = gflow_all_sol[idx]
                perf = gflow_all_perf[idx]
                results[j,i,k,1,0] = np.mean(perf)
                results[j,i,k,1,1] = np.mean(sol)
                results[j,i,k,1,2] = diversity(seqs)
                results[j,i,k,1,3] = novelty(init,seqs)
                results[j,i,k,1,4] = np.std(perf)
                results[j,i,k,1,5] = np.std(sol)
                results[j,i,k,1,6] = 0
                results[j,i,k,1,7] = 0

                idx = np.argsort(antBO_dist)[:budget]
                seqs = [antBO_seqs[i] for i in idx]
                sol = antBO_all_sol[idx]
                perf = antBO_all_perf[idx]
                results[j,i,k,2,0] = np.mean(perf)
                results[j,i,k,2,1] = np.mean(sol)
                results[j,i,k,2,2] = diversity(seqs)
                results[j,i,k,2,3] = novelty(init,seqs)
                results[j,i,k,2,4] = np.std(perf)
                results[j,i,k,2,5] = np.std(sol)
                results[j,i,k,2,6] = 0
                results[j,i,k,2,7] = 0
    
    methods_shift = [-0.07,0.0,0.07]
    beta = np.array(beta)
    methods = ['mcmc','gflownet','antBO']
    task = ['performance','solubility','diversity','novelty']
    fig,axs = plt.subplots(len(task),len(budgets),figsize = (16,24))
    for i in range(len(global_weights)):
        for j in range(len(budgets)):
            for t in range(len(task)):
                axs[t,j].set_xlabel(r'$\beta_{epi}$')
                axs[t,j].set_ylabel(task[t])
                axs[t,j].set_title('budget : {}'.format(budgets[j]))
                axs[t,j].set_xticks(beta)
                for k in range(len(methods)):
                    axs[t,j].errorbar(beta + methods_shift[k],results[i,:,j,k,t],yerr = results[i,:,j,k,t + 4], label = methods[k],capsize = 10,fmt = 'o-',elinewidth = 5)
                if t == 0 and j == 0:
                    axs[t,j].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_methods_true_aff_top.png')


def compare_true_aff_top_box_plots():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    methods = ['MCMC','GFlowNet','AntBO']
    sol_threshold = [-1.0,4.0,5.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,7,figsize = (56,9))
    budget = 20
    seqs,aff_r,sol_r,ppost_r,perf_r = get_init_true_aff_no_dupl()
    top_idx = np.argsort(perf_r)[:budget]
    perf_r = perf_r[top_idx]
    sol_r = sol_r[top_idx]
    random_seqs,random_aff_r,random_sol_r,random_ppost_r,random_perf_r = get_random_true_aff_no_dupl()
    top_idx = np.argsort(random_aff_r)[-budget:]
    random_perf_r = random_perf_r[top_idx]
    random_sol_r = random_sol_r[top_idx]
    print(np.min(perf_r))

    aff_threshold = np.linspace(1,-4)
    perc_better = np.zeros((len(sol_threshold),len(aff_threshold),len(beta),len(methods)))
    perc_better_init = np.zeros((len(sol_threshold),len(aff_threshold)))
    perc_better_random = np.zeros((len(sol_threshold),len(aff_threshold)))

    labels = []
    perfs = []
    sols = []
    for t in range(len(sol_threshold)):
        sol_t = sol_threshold[t]
        for t2 in range(len(aff_threshold)):
            aff_t = aff_threshold[t2]
            perc_better_init[t,t2] = perc_filter_seqs(sol_t,aff_t,perf_r,sol_r)
    labels.append('initial_mutants')
    perfs.append(perf_r)
    sols.append(sol_r)

    for t in range(len(sol_threshold)):
        sol_t = sol_threshold[t]
        for t2 in range(len(aff_threshold)):
            aff_t = aff_threshold[t2]
            perc_better_random[t,t2] = perc_filter_seqs(sol_t,aff_t,random_perf_r,random_sol_r)
    labels.append('random mutants')
    perfs.append(random_perf_r)
    sols.append(random_sol_r)

    count = 1
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            for m in range(len(methods)):
                if methods[m] == 'MCMC':
                    seqs,aff,sol,ppost,perf = get_mcmc_true_aff_no_dupl(gw,b,replicate)
                elif methods[m] == 'GFlowNet':
                    seqs,aff,sol,ppost,perf = get_gflow_true_aff_no_dupl(gw,b,replicate)
                else:
                    b = 0.0
                    seqs,aff,sol,ppost,perf = get_antBO_true_aff_no_dupl(b,replicate)
                perfs.append(perf)
                sols.append(sol)
                for t in range(len(sol_threshold)):
                    sol_t = sol_threshold[t]
                    aff2,sol2,perf2 = filter_seqs_sol(sol_t,aff,sol,perf)
                    top_idx = np.argsort(aff2)[-budget:]
                    top_perf = perf2[top_idx]
                    top_sol = sol2[top_idx]
                    for t2 in range(len(aff_threshold)):
                        aff_t = aff_threshold[t2]
                        perc = perc_filter_seqs(sol_t,aff_t,top_perf,top_sol)
                        perc_better[t,t2,f,m] = perc
                count += 1
                print(count)
                labels.append(r'$\beta$ : ' + '{}'.format(b) + r'/ $method$ : ' + '{}'.format(methods[m]))

    axs[0].boxplot(perfs,labels = labels)
    axs[0].set_xticklabels(labels,rotation = 90)
    axs[0].set_ylabel('True Affinity')
    axs[1].boxplot(sols,labels = labels)
    axs[1].set_xticklabels(labels,rotation = 90)
    axs[1].set_ylabel('Solubility')
    axs[2].violinplot(perfs)
    axs[2].set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
    axs[2].set_ylabel('True Affinity')
    axs[3].violinplot(sols)
    axs[3].set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation = 90)
    axs[3].set_ylabel('Solubility')

    #perc_better = perc_better.reshape((len(sol_threshold),len(aff_threshold),-1))


    legend_elements = [Line2D([0], [0], linestyle = '-', color='pink',lw = 1, label=r'initial'),
                        Line2D([0], [0], linestyle = '-', color='orange',lw = 1, label=r'random'),
                        Line2D([0], [0], linestyle = '-', color='k', label='MCMC'),
                        Line2D([0], [0], linestyle = '--', color='k', label='GFlowNet'),
                        Line2D([0], [0], linestyle = ':', color='m', label='AntBO'),
                        Line2D([0], [0], linestyle = '-', color='b',lw = 1, label=r'$\beta$ = -1'),
                        Line2D([0], [0], linestyle = '-', color='y',lw = 1, label=r'$\beta$ = 0'),
                        Line2D([0], [0], linestyle = '-', color='r',lw = 1, label=r'$\beta$ = 1'),
                        Line2D([0], [0], linestyle = '-', color='g',lw = 1, label=r'$\beta$ = 2')]

    methods_fmt = ['-','--',':']
    beta_color = ['b','y','r','g']
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(methods)):
            m = methods[j]
            #area = simpson(-perc_better[0,:,i,j], x = aff_threshold)
            fmt = methods_fmt[j]
            color = beta_color[i]
            if m == 'AntBO':
                color = 'm'
            axs[4].plot(-aff_threshold,perc_better[0,:,i,j],fmt,c = color)
    axs[4].plot(-aff_threshold,perc_better_init[0],c = 'pink')
    axs[4].plot(-aff_threshold,perc_better_random[0],c = 'orange')
    axs[4].set_ylabel('Number of Sequences above thresholds')
    axs[4].set_xlabel(r'$f^{min}_{aff}$')
    axs[4].set_title('No solubility threshold')
    axs[4].legend(frameon = False,handles=legend_elements)


    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(methods)):
            m = methods[j]
            #area = simpson(-perc_better[1,:,i,j], x = aff_threshold)
            fmt = methods_fmt[j]
            color = beta_color[i]
            if m == 'AntBO':
                color = 'm'
            axs[5].plot(-aff_threshold,perc_better[1,:,i,j],fmt,c = color)
    axs[5].plot(-aff_threshold,perc_better_init[1],c = 'pink')
    axs[5].plot(-aff_threshold,perc_better_random[1],c = 'orange')
    axs[5].set_ylabel('Number of Sequences above thresholds')
    axs[5].set_xlabel(r'$f^{min}_{aff}$')
    axs[5].set_title(r'$f^{min}_{sol}$ = 4.0')

    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(methods)):
            m = methods[j]
            #area = simpson(-perc_better[1,:,i,j], x = aff_threshold)
            fmt = methods_fmt[j]
            color = beta_color[i]
            if m == 'AntBO':
                color = 'm'
            axs[6].plot(-aff_threshold,perc_better[2,:,i,j],fmt,c = color)
    axs[6].plot(-aff_threshold,perc_better_init[2],c = 'pink')
    axs[6].plot(-aff_threshold,perc_better_random[2],c = 'orange')
    axs[6].set_ylabel('Number of Sequences above thresholds')
    axs[6].set_xlabel(r'$log_{10}$ ($T_{AFF}$ (nM))')
    axs[6].set_title(r'$T_{sol}$ = 5.0')
    axs[6].legend()

    plt.tight_layout()
    plt.savefig('./pareto_final/box_plots_compare_true_aff_top_budget:{}.png'.format(budget))

## END OF TRUE AFF

def preprocess_pareto_init_true_aff_hard(oracle):
    oracle.update_true_aff_gp(1.0,0.8,'hard')
    seqs = get_true_aff_initial()
    seqs = list(set(seqs))
    batch_size = 128
    n_batches = int(len(seqs)/batch_size) + 1
    sol_r = np.array([])
    aff_r = np.array([])
    mu_r = np.array([])
    ppost_r = np.array([])
    perf_r = np.array([])
    insta_r = np.array([])
    hydro_r = np.array([])
    charge_r = np.array([])
    for j in tqdm(range(n_batches)):
        s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r = - a_r
        p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        print(oracle.true_aff_ora.method)
        performance = oracle.true_aff_ora.score_without_noise_hard(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        mu_r = np.concatenate((mu_r,a_r))
        sol_r = np.concatenate((sol_r,s_r))
        aff_r = np.concatenate((aff_r,a_r))
        ppost_r = np.concatenate((ppost_r,p_r))
        insta_r = np.concatenate((insta_r,instability))
        hydro_r = np.concatenate((hydro_r,hydrophobicity))
        charge_r = np.concatenate((charge_r,charge))
        perf_r = np.concatenate((perf_r,performance))
    print(np.min(perf_r))
    with open('./lib/dataset/preprocess/init_true_aff_hard_seqs_preprocess.npy','w') as f:
        for seq in seqs:
            f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/init_true_aff_hard_mu_preprocess.npy',mu_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_aff_preprocess.npy',aff_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_sol_preprocess.npy',sol_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_ppost_preprocess.npy',ppost_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_insta_preprocess.npy',insta_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_hydro_preprocess.npy',hydro_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_charge_preprocess.npy',charge_r)
        np.save('./lib/dataset/preprocess/init_true_aff_hard_performance_preprocess.npy',perf_r)

def get_init_true_aff_hard_no_dupl():
    aff_r = np.load('./lib/dataset/preprocess/init_true_aff_hard_aff_preprocess.npy')
    sol_r = np.load('./lib/dataset/preprocess/init_true_aff_hard_sol_preprocess.npy')
    ppost_r = np.load('./lib/dataset/preprocess/init_true_aff_hard_ppost_preprocess.npy')
    perf_r = np.load('./lib/dataset/preprocess/init_true_aff_hard_performance_preprocess.npy')
    seqs = []
    with open('./lib/dataset/preprocess/init_true_aff_hard_seqs_preprocess.npy','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def preprocess_pareto_random_true_aff_hard(oracle):
    oracle.update_true_aff_gp(1.0,0.8,'hard')
    seqs = get_true_aff_random()
    seqs = list(set(seqs))
    batch_size = 128
    n_batches = int(len(seqs)/batch_size) + 1
    sol_r = np.array([])
    aff_r = np.array([])
    mu_r = np.array([])
    ppost_r = np.array([])
    perf_r = np.array([])
    insta_r = np.array([])
    hydro_r = np.array([])
    charge_r = np.array([])
    for j in tqdm(range(n_batches)):
        s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        a_r = - a_r
        p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        print(oracle.true_aff_ora.method)
        performance = oracle.true_aff_ora.score_without_noise_hard(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
        mu_r = np.concatenate((mu_r,a_r))
        sol_r = np.concatenate((sol_r,s_r))
        aff_r = np.concatenate((aff_r,a_r))
        ppost_r = np.concatenate((ppost_r,p_r))
        insta_r = np.concatenate((insta_r,instability))
        hydro_r = np.concatenate((hydro_r,hydrophobicity))
        charge_r = np.concatenate((charge_r,charge))
        perf_r = np.concatenate((perf_r,performance))
    print(np.min(perf_r))
    with open('./lib/dataset/preprocess/random_true_aff_hard_seqs_preprocess.npy','w') as f:
        for seq in seqs:
            f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/random_true_aff_hard_mu_preprocess.npy',mu_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_aff_preprocess.npy',aff_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_sol_preprocess.npy',sol_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_ppost_preprocess.npy',ppost_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_insta_preprocess.npy',insta_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_hydro_preprocess.npy',hydro_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_charge_preprocess.npy',charge_r)
        np.save('./lib/dataset/preprocess/random_true_aff_hard_performance_preprocess.npy',perf_r)

def get_random_true_aff_hard_no_dupl():
    aff_r = np.load('./lib/dataset/preprocess/random_true_aff_hard_aff_preprocess.npy')
    sol_r = np.load('./lib/dataset/preprocess/random_true_aff_hard_sol_preprocess.npy')
    ppost_r = np.load('./lib/dataset/preprocess/random_true_aff_hard_ppost_preprocess.npy')
    perf_r = np.load('./lib/dataset/preprocess/random_true_aff_hard_performance_preprocess.npy')
    seqs = []
    with open('./lib/dataset/preprocess/random_true_aff_hard_seqs_preprocess.npy','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r



## START OF HARD TRUE AFF

##START OF TRUE_AFF_MCMC

def get_mcmc_true_aff_hard_seqs(weights,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/mcmc/true_aff_hard/mcmc_true_aff_hard_exp_lim_burn_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/mcmc/true_aff_hard/mcmc_true_aff_hard_exp_lim_burn_prob_sol:{}_aff:{}_global:{}_beta:{}_immuno:{}_rep:{}.txt.npy'.format(weights[1],weights[2],weights[3],weights[4],weights[5],replicate))
    seqs = list(set(seqs))
    init_seqs,_,_,_,_ = get_init_true_aff_hard_no_dupl()
    seqs = [i for i in seqs if i not in init_seqs]
    return seqs,score

def get_all_mcmc_true_aff_hard_seqs(gw,beta,replicate):
    #weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.125,0.875,gw,beta,1.0],[1.0,0.1,0.9,gw,beta,1.0],[1.0,0.05,0.95,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    mcmc_seqs = []
    mcmc_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_mcmc_true_aff_hard_seqs(w,replicate)
        mcmc_seqs += seqs
        mcmc_score = np.concatenate((mcmc_score,score))
    return mcmc_seqs, mcmc_score

def preprocess_pareto_mcmc_true_aff_hard(oracle):
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for i in global_weights:
        for b in beta:
            seqs,score = get_all_mcmc_true_aff_hard_seqs(i,b,replicate)
            seqs = list(set(seqs))
            n_batches = int(len(seqs)/batch_size) + 1
            sol_r = np.array([])
            aff_r = np.array([])
            mu_r = np.array([])
            ppost_r = np.array([])
            perf_r = np.array([])
            insta_r = np.array([])
            hydro_r = np.array([])
            charge_r = np.array([])
            for j in tqdm(range(n_batches)):
                s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = 12 - a_r
                p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                performance = oracle.true_aff_ora.score_without_noise_hard(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                mu_r = np.concatenate((mu_r,a_r))
                a_r = a_r + b * v_r
                sol_r = np.concatenate((sol_r,s_r))
                aff_r = np.concatenate((aff_r,a_r))
                ppost_r = np.concatenate((ppost_r,p_r))
                insta_r = np.concatenate((insta_r,instability))
                hydro_r = np.concatenate((hydro_r,hydrophobicity))
                charge_r = np.concatenate((charge_r,charge))
                perf_r = np.concatenate((perf_r,performance))
            with open('./lib/dataset/preprocess/mcmc_true_aff_hard_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),'w') as f:
                for seq in seqs:
                    f.write(seq + '\n')
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_mu_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),mu_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),aff_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),sol_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),ppost_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),insta_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_hard_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),hydro_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),charge_r)
            np.save('./lib/dataset/preprocess/mcmc_true_aff_hard_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),perf_r)

def get_mcmc_true_aff_hard_no_dupl(g,beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_hard_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    aff_r = aff_r - 12
    sol_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_hard_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    ppost_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_hard_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    perf_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_hard_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    seqs = []
    with open('./lib/dataset/preprocess/mcmc_true_aff_hard_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_mcmc_true_aff_hard_dict(g,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_hard_no_dupl(g,beta,replicate)
    d = {}
    for i in range(len(seqs)):
        if seqs[i] not in d:
            d[seqs[i]] = (sol_r[i],aff_r[i],ppost_r[i],perf_r[i])
        else:
            print('error!!!!')
    return d

def get_mcmc_true_aff_hard_error(g,beta,replicate):
    perf_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_hard_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    mu_r = np.load('./lib/dataset/preprocess/mcmc_true_aff_hard_mu_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    mu_r = mu_r - 12
    return mu_r,perf_r

def get_mcmc_true_aff_hard_pareto_front(gw,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_hard_no_dupl(gw,beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,perf_r

def plot_mcmc_pareto_true_aff_hard_top():
    beta = [1.0,-1.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_hard_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('MCMC True Aff Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_compare_pareto_temp_rep:{}.png'.format(replicate))

def plot_mcmc_true_aff_hard_developability():
    beta = [0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    all_sol = np.array([])
    all_hum = np.array([])
    all_aff = np.array([])
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r,perf_r = get_mcmc_true_aff_hard_no_dupl(g,b,replicate)
            all_aff = np.concatenate((all_aff,aff_r))
            all_sol = np.concatenate((all_sol,sol_r))
            all_hum = np.concatenate((all_hum,ppost_r))
    x,y,z = get_color_plot(all_sol,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('MCMC True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_developability_hum_sol_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('MCMC True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_developability_hum_aff_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_sol)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('MCMC True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_developability_aff_sol_rep:{}.png'.format(replicate))
    plt.clf()

def find_seqs_in_trust():
    beta = [1.0,0.0,-1.0]
    global_weight = [10.0]
    replicate = 1
    sol_min = 4
    hum_min = -105
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r = get_mcmc_true_aff_hard_no_dupl(g,b,replicate)
            sol_mask = sol_r > sol_min
            hum_mask = ppost_r > hum_min
            aff_r = (aff_r * sol_mask) * hum_mask
            print(np.sum(aff_r > 0))
            print(np.quantile(aff_r,0.9))

def plot_mcmc_pareto_true_aff_hard_density():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_mcmc_true_aff_hard_pareto_front(i,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'MCMC generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel('affinity score')
            axs[0,m].set_ylabel('solubility score')
            axs[0,m].legend()
            axs[0,m].set_title('MCMC/GW:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('affinity score')
            axs[1,m].set_ylabel('true affinity')
            axs[1,m].legend()
            axs[1,m].set_title('MCMC/GW:{}/beta:{}/pear:{}'.format(i,b,pearson))

    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_mcmc_true_aff_hard_density_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))

def plot_mcmc_pareto_true_aff_hard_density_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,5,figsize = (40,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_mcmc_true_aff_hard_pareto_front(i,b,replicate)
            idx = np.argsort(dist)[:1000]
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,4].scatter(all_perf,all_sol,label = r'$\beta$' + ' : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'MCMC generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel(r'$\hat f_{\rm sol}$')
            axs[0,m].set_ylabel(r'$\hat f_{\rm sol}$')
            axs[0,m].set_ylim(1,7)
            if m == 0: 
                axs[0,m].legend(frameon = False)
            axs[0,m].set_title(r'$\beta$ : ' + ' {}'.format(b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('Affinity Score')
            axs[1,m].set_ylabel('True affinity')
            axs[1,m].legend()
            axs[1,m].set_title(r'$\beta$ : ' + r' {} / $\rho$:{}'.format(b,pearson))
    axs[0,4].set_xlabel('True affinity')
    axs[0,4].set_ylabel('Solubility')
    axs[0,4].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_mcmc_true_aff_hard_density_top_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))

def analyze_mcmc_true_aff_hard():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    multiple = 10
    budgets = [5,500]
    fig,axs = plt.subplots(len(budgets),5,figsize = (40,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5,multiple))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_mcmc_true_aff_hard_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    for m in range(multiple):
                        seqs, score = get_mcmc_true_aff_hard_seqs(weights[l],replicate)
                        seqs = list(set(seqs))
                        seqs = random.choices(seqs,k = budget)
                        all_aff = np.zeros(len(seqs))
                        all_sol = np.zeros(len(seqs))
                        all_ppost = np.zeros(len(seqs))
                        all_perf = np.zeros(len(seqs))
                        for j in range(len(seqs)):
                            r = dupl_dict[seqs[j]]
                            all_sol[j] = r[0]
                            all_aff[j] = r[1]
                            all_ppost[j] = r[2]
                            all_perf[j] = r[3]
                        results[i,f,k,l,0,m] = np.mean(all_sol)
                        results[i,f,k,l,1,m] = np.mean(all_perf)
                        results[i,f,k,l,2,m] = np.mean(all_ppost)
                        results[i,f,k,l,3,m] = diversity(seqs)
                        results[i,f,k,l,4,m] = novelty(init,seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                p_mean = np.mean(results[i,:,k,l,1],axis = 1)
                p_var = np.std(results[i,:,k,l,1],axis = 1)
                s_mean = np.mean(results[i,:,k,l,0],axis = 1)
                s_var = np.std(results[i,:,k,l,0],axis = 1)
                axs[i,0].plot(beta,p_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,0].fill_between(beta, (p_mean-2*p_var), (p_mean+2*p_var), alpha=.1)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta, p_var,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,1].set_ylabel('performance standard deviation')
                axs[i,1].set_xlabel('beta')
                axs[i,2].plot(beta,s_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,2].set_ylabel('solubility')
                axs[i,2].set_xlabel('beta')
                axs[i,3].plot(beta,np.mean(results[i,:,k,l,3],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,3].set_ylabel('diversity')
                axs[i,3].set_xlabel('beta')
                axs[i,4].plot(beta,np.mean(results[i,:,k,l,4],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,4].set_ylabel('novelty')
                axs[i,4].set_xlabel('beta')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_analyze.png')

def analyze_mcmc_true_aff_hard_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    budgets = [5,500]
    fig,axs = plt.subplots(len(budgets),4,figsize = (32,8*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,10))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_mcmc_true_aff_hard_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    seqs, score = get_mcmc_true_aff_hard_seqs(weights[l],replicate)
                    seqs = list(set(seqs))
                    all_aff = np.zeros(len(seqs))
                    all_sol = np.zeros(len(seqs))
                    all_ppost = np.zeros(len(seqs))
                    all_perf = np.zeros(len(seqs))
                    for j in range(len(seqs)):
                        r = dupl_dict[seqs[j]]
                        all_sol[j] = r[0]
                        all_aff[j] = r[1]
                        all_ppost[j] = r[2]
                        all_perf[j] = r[3]
                    aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist = remove_dominated(all_aff,all_sol)
                    dist = np.array(dist)
                    top_idx = np.argsort(dist)[:budget]

                    top_aff = all_aff[top_idx]
                    top_sol = all_sol[top_idx]
                    top_ppost = all_ppost[top_idx]
                    top_perf = all_perf[top_idx]
                    top_seqs = [seqs[i] for i in top_idx]
                    div_mean, div_std = diversity(top_seqs,return_std = True)
                    nov_mean, nov_std = novelty(init,top_seqs,return_std = True)
                    results[i,f,k,l,0] = np.mean(top_sol)
                    results[i,f,k,l,1] = np.mean(top_perf)
                    results[i,f,k,l,2] = np.mean(top_ppost)
                    results[i,f,k,l,3] = div_mean
                    results[i,f,k,l,4] = nov_mean
                    results[i,f,k,l,5] = np.std(top_sol)
                    results[i,f,k,l,6] = np.std(top_perf)
                    results[i,f,k,l,7] = np.std(top_ppost)
                    results[i,f,k,l,8] = div_std
                    results[i,f,k,l,9] = nov_std

    beta = np.array(beta)
    methods_shift = [-0.125,0.125]
    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                axs[i,0].errorbar(beta + methods_shift[l],results[i,:,k,l,1],yerr = results[i,:,k,l,6],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,0].bar(beta + methods_shift[l],results[i,:,k,l,1],label = r'budget:{} / $aff_w$:{}'.format(budgets[i],aff_w[l]),width = 0.25)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel(r'$\beta$')
                axs[i,0].set_xticks(beta)
                axs[i,0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
                axs[i,1].errorbar(beta + methods_shift[l],results[i,:,k,l,0],yerr = results[i,:,k,l,5],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,1].bar(beta + methods_shift[l],results[i,:,k,l,0],label = r'budget:{} / $aff_w$:{}'.format(budgets[i],aff_w[l]),width = 0.25)
                axs[i,1].set_ylabel('solubility')
                axs[i,1].set_xlabel(r'$\beta$')
                axs[i,1].set_xticks(beta)
                axs[i,2].errorbar(beta + methods_shift[l],results[i,:,k,l,3],yerr = results[i,:,k,l,8],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,2].bar(beta + methods_shift[l],results[i,:,k,l,3],label = r'$w$' + ': {}'.format(aff_w[l]),width = 0.25)
                axs[i,2].set_ylabel('Diversity')
                axs[i,2].set_xlabel(r'$\beta$')
                axs[i,2].set_xticks(beta)
                axs[i,2].set_ylim(0,8)
                axs[i,2].legend(loc='lower center', bbox_to_anchor=(0.15, 0.85), frameon = False)
                axs[i,3].errorbar(beta + methods_shift[l],results[i,:,k,l,4],yerr = results[i,:,k,l,9],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,3].bar(beta + methods_shift[l],results[i,:,k,l,4],label = r'$w$:' + ' {}'.format(aff_w[l]),width = 0.25)
                axs[i,3].set_ylabel('Novelty')
                axs[i,3].set_xlabel(r'$\beta$')
                axs[i,3].set_xticks(beta)
                #axs[i,3].legend(loc='lower center', bbox_to_anchor=(0.25, 0.8), frameon = False)
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_analyze_top.png')

def true_aff_hard_top_box_plots():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    aff_w = [0.85,1.0]
    sol_threshold = [-1.0,4.0,5.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,7,figsize = (56,9))
    budget = 20
    init_seqs,aff_r,sol_r,ppost_r,perf_r = get_init_true_aff_hard_no_dupl()
    top_idx = np.argsort(perf_r)[:budget]
    perf_r = perf_r[top_idx]
    sol_r = sol_r[top_idx]

    aff_threshold = np.linspace(1.0,-4)
    perc_better = np.zeros((len(sol_threshold),len(aff_threshold),len(beta),len(aff_w)))
    perc_better_init = np.zeros((len(sol_threshold),len(aff_threshold)))

    labels = []
    perfs = []
    sols = []
    for t in range(len(sol_threshold)):
        sol_t = sol_threshold[t]
        for t2 in range(len(aff_threshold)):
            aff_t = aff_threshold[t2]
            perc_better_init[t,t2] = perc_filter_seqs(sol_t,aff_t,perf_r,sol_r)
    labels.append('initial_mutants')
    perfs.append(perf_r)
    sols.append(sol_r)

    count = 1
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            dupl_dict = get_mcmc_true_aff_hard_dict(gw,b,replicate)
            weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for l in range(len(weights)):
                seqs, score = get_mcmc_true_aff_hard_seqs(weights[l],replicate)
                seqs = list(set(seqs))
                seqs = [i for i in seqs if i not in init_seqs]
                all_sol = np.zeros(len(seqs))
                all_aff = np.zeros(len(seqs))
                all_perf = np.zeros(len(seqs))
                for j in range(len(seqs)):
                    r = dupl_dict[seqs[j]]
                    all_sol[j] = r[0]
                    all_aff[j] = r[1]
                    all_perf[j] = r[3]
                perfs.append(all_perf)
                sols.append(all_sol)
                for t in range(len(sol_threshold)):
                    sol_t = sol_threshold[t]
                    aff, sol, perf = filter_seqs_sol(sol_t,all_aff,all_sol,all_perf)
                    top_idx = np.argsort(aff)[-budget:]
                    top_perf = perf[top_idx]
                    top_sol = sol[top_idx]
                    for t2 in range(len(aff_threshold)):
                        aff_t = aff_threshold[t2]
                        perc = perc_filter_seqs(sol_t,aff_t,top_perf,top_sol)
                        perc_better[t,t2,f,l] = perc
                count += 1
                print(count)
                labels.append(r'$\beta$ : ' + '{}'.format(b) + r'/ $method$ : ' + '{}'.format(aff_w[l]))

    perfs = np.array(perfs)
    axs[0].boxplot(-perfs,labels = labels)
    axs[0].set_xticklabels(labels,rotation = 90)
    axs[0].set_ylabel(r'$f^{min}_aff$')
    axs[1].boxplot(sols,labels = labels)
    axs[1].set_xticklabels(labels,rotation = 90)
    axs[1].set_ylabel('Solubility')
    axs[2].violinplot(-perfs)
    axs[2].set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
    axs[2].set_ylabel(r'$f^{min}_aff$')
    axs[3].violinplot(sols)
    axs[3].set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation = 90)
    axs[3].set_ylabel('Solubility')

    #perc_better = perc_better.reshape((len(sol_threshold),len(aff_threshold),-1))

    beta_color = ['b','y','r','g']
    aff_fmt = ['--','-']
    legend_elements = [Line2D([0], [0], linestyle = '-', color='pink',lw = 1, label=r'initial'),
                        Line2D([0], [0], linestyle = '--', color='k', label='w_aff = 0.85'),
                        Line2D([0], [0], linestyle = '-', color='k', label='w_aff = 1.0'),
                        Line2D([0], [0], linestyle = '-', color='b',lw = 1, label=r'$\beta$ = -1'),
                        Line2D([0], [0], linestyle = '-', color='y',lw = 1, label=r'$\beta$ = 0'),
                        Line2D([0], [0], linestyle = '-', color='r',lw = 1, label=r'$\beta$ = 1'),
                        Line2D([0], [0], linestyle = '-', color='g',lw = 1, label=r'$\beta$ = 2')]
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(aff_w)):
            m = aff_w[j]
            fmt = aff_fmt[j]
            color = beta_color[i]
            area = simpson(-perc_better[0,:,i,j], x = aff_threshold)
            axs[4].plot(-aff_threshold,perc_better[0,:,i,j],fmt,c = color)
    axs[4].plot(-aff_threshold,perc_better_init[0],c = 'pink')
    axs[4].set_ylabel('Number of Sequences above thresholds')
    axs[4].set_xlabel(r'$f^{min}_{aff}$',fontsize = 18)
    axs[4].set_title('No solubility threshold')
    #axs[4].legend(handles=legend_elements)

    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(aff_w)):
            m = aff_w[j]
            fmt = aff_fmt[j]
            color = beta_color[i]
            area = simpson(-perc_better[1,:,i,j], x = aff_threshold)
            axs[5].plot(-aff_threshold,perc_better[1,:,i,j],fmt, c = color)
    axs[5].plot(-aff_threshold,perc_better_init[1],c = 'pink')
    axs[5].set_ylabel('Number of Sequences above thresholds')
    axs[5].set_xlabel(r'$f^{min}_{aff}$',fontsize = 18)
    axs[5].set_title(r'$f^{min}_{sol} = 4.0$',fontsize = 18)

    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(aff_w)):
            m = aff_w[j]
            fmt = aff_fmt[j]
            color = beta_color[i]
            area = simpson(-perc_better[2,:,i,j], x = aff_threshold)
            axs[6].plot(-aff_threshold,perc_better[2,:,i,j],fmt, c = color)
    axs[6].plot(-aff_threshold,perc_better_init[1],c = 'grey')
    axs[6].set_ylabel('Number of Sequences above thresholds')
    axs[6].set_xlabel(r'$f^{min}_aff$')
    axs[6].set_title('Solubility threshold = 5.0')
    axs[6].legend()

    plt.tight_layout()
    plt.savefig('./pareto_final/box_plots_true_aff_hard_top_budget:{}.png'.format(budget))

def true_aff_hard_top_plot_error():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    replicate = 1
    labels = []
    errors = []
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            mu_r,perf_r =  get_mcmc_true_aff_error(gw,b,replicate)
            errors.append(perf_r - mu_r)
            labels.append(r'Simple \ $\beta$ = {}'.format(b))
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            mu_r,perf_r =  get_mcmc_true_aff_hard_error(gw,b,replicate)
            errors.append(perf_r - mu_r)
            labels.append(r'Hard \ $\beta$ = {}'.format(b))
    plt.violinplot(errors,showmeans = True)
    plt.xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
    plt.ylabel(r'Error distribution')
    plt.tight_layout()
    plt.savefig('./pareto_final/mcmc_true_aff_hard_error_distribution.png')



## GFLOW HARD 

def get_gflow_true_aff_hard_seqs(global_weight,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/gflownet/true_aff_hard/true_aff_hard_gflow_sol:{}_aff:{}_global:{}_beta:{}_rep:{}.txt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/gflownet/true_aff_hard/true_aff_hard_gflow_scores_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],replicate))
    seqs = list(set(seqs))
    init_seqs,_,_,_,_ = get_init_true_aff_hard_no_dupl()
    seqs = [i for i in seqs if i not in init_seqs]
    return seqs,score

def get_all_gflow_true_aff_hard_seqs(gw,beta,replicate):
    #weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.125,0.875,gw,beta,1.0],[1.0,0.1,0.9,gw,beta,1.0],[1.0,0.05,0.95,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    weights = [[1.0,0.15,0.85,gw,beta,1.0],[1.0,0.0,1.0,gw,beta,1.0]]
    gflow_seqs = []
    gflow_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_gflow_true_aff_hard_seqs(w,replicate)
        gflow_seqs += seqs
        gflow_score = np.concatenate((gflow_score,score))
    return gflow_seqs, gflow_score

def preprocess_pareto_gflow_true_aff_hard(oracle):
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for i in global_weights:
        for b in beta:
            seqs,score = get_all_gflow_true_aff_hard_seqs(i,b,replicate)
            seqs = list(set(seqs))
            n_batches = int(len(seqs)/batch_size) + 1
            sol_r = np.array([])
            aff_r = np.array([])
            ppost_r = np.array([])
            perf_r = np.array([])
            insta_r = np.array([])
            hydro_r = np.array([])
            charge_r = np.array([])
            for j in tqdm(range(n_batches)):
                s_r = oracle.get_sasa_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r,v_r = oracle.get_trueaff_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = 12 - a_r
                p_r = oracle.get_ppost_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                performance = oracle.true_aff_ora.score_without_noise_hard(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
                a_r = a_r + b * v_r
                sol_r = np.concatenate((sol_r,s_r))
                aff_r = np.concatenate((aff_r,a_r))
                ppost_r = np.concatenate((ppost_r,p_r))
                insta_r = np.concatenate((insta_r,instability))
                hydro_r = np.concatenate((hydro_r,hydrophobicity))
                charge_r = np.concatenate((charge_r,charge))
                perf_r = np.concatenate((perf_r,performance))
            with open('./lib/dataset/preprocess/gflow_true_aff_hard_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),'w') as f:
                for seq in seqs:
                    f.write(seq + '\n')
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),aff_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),sol_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),ppost_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_insta_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),insta_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_hydro_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),hydro_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_charge_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),charge_r)
            np.save('./lib/dataset/preprocess/gflow_true_aff_hard_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(i,b,replicate),perf_r)


def get_gflow_true_aff_hard_no_dupl(g,beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/gflow_true_aff_hard_aff_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/gflow_true_aff_hard_sol_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    ppost_r = np.load('./lib/dataset/preprocess/gflow_true_aff_hard_ppost_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    perf_r = np.load('./lib/dataset/preprocess/gflow_true_aff_hard_performance_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate))
    seqs = []
    with open('./lib/dataset/preprocess/gflow_true_aff_hard_seqs_preprocess_global:{}_beta:{}_rep_{}.npy'.format(g,beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_gflow_true_aff_hard_dict(g,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_gflow_true_aff_hard_no_dupl(g,beta,replicate)
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i],ppost_r[i],perf_r[i])
    return d

def get_gflow_true_aff_hard_pareto_front(gw,beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_gflow_true_aff_hard_no_dupl(gw,beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,perf_r

def plot_gflow_pareto_true_aff_hard_top():
    beta = [1.0,-1.0,0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r = get_gflow_true_aff_hard_no_dupl(g,b,replicate)
            pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
            plt.scatter(pareto_aff,pareto_sol,label = 'inverse temp = {}/beta:{}'.format(g,b))
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('GFLOW True Aff Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_hard_compare_pareto_temp_rep:{}.png'.format(replicate))


def plot_gflow_true_aff_hard_developability():
    beta = [0.0]
    global_weight = [10.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    all_sol = np.array([])
    all_hum = np.array([])
    all_aff = np.array([])
    for b in beta:
        print(b)
        for g in global_weight:
            seqs,aff_r,sol_r,ppost_r = get_gflow_true_aff_hard_no_dupl(g,b,replicate)
            all_aff = np.concatenate((all_aff,aff_r))
            all_sol = np.concatenate((all_sol,sol_r))
            all_hum = np.concatenate((all_hum,ppost_r))
    x,y,z = get_color_plot(all_sol,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('GFLOW True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_hard_developability_hum_sol_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_hum)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('GFLOW True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_hard_developability_hum_aff_rep:{}.png'.format(replicate))
    plt.clf()

    x,y,z = get_color_plot(all_aff,all_sol)
    plt.scatter(x,y,c = z,label = 'inverse temp = {}/beta:{}'.format(g,b))
    plt.ylabel('solubility score')
    plt.xlabel('humaness score')
    plt.title('GFLOW True Aff Developability')
    plt.colorbar()
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_hard_developability_aff_sol_rep:{}.png'.format(replicate))
    plt.clf()

def plot_gflow_pareto_true_aff_hard_density():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_gflow_true_aff_hard_pareto_front(i,b,replicate)
            idx = random.choices(range(len(seqs)),k = 1000)
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'gflow generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel('affinity score')
            axs[0,m].set_ylabel('solubility score')
            axs[0,m].legend()
            axs[0,m].set_title('gflow/GW:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('affinity score')
            axs[1,m].set_ylabel('true affinity')
            axs[1,m].legend()
            axs[1,m].set_title('gflow/GW:{}/beta:{}/pear:{}'.format(i,b,pearson))

    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_true_aff_hard_density_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))

def plot_gflow_pareto_true_aff_hard_density_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        for i in global_weights:
            seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_gflow_true_aff_hard_pareto_front(i,b,replicate)
            idx = np.argsort(dist)[:1000]
            seqs = [seqs[i] for i in idx]
            all_aff = all_aff[idx]
            all_sol = all_sol[idx]
            all_perf = all_perf[idx]
            axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
            x,y,z = get_color_plot(all_aff,all_sol)
            z = axs[0,m].scatter(x,y,c = z,label = 'gflow generated seqs')
            plt.colorbar(z,ax = axs[0,m])
            #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
            idx = np.argsort(pareto_aff).tolist()
            pareto_aff = [pareto_aff[i] for i in idx]
            pareto_sol = [pareto_sol[i] for i in idx]
            axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
            axs[0,m].set_xlabel('affinity score')
            axs[0,m].set_ylabel('solubility score')
            axs[0,m].legend()
            axs[0,m].set_title('gflow/GW:{}/beta:{}'.format(i,b))
            if add_initial:
                cdrlist = []
                with open('./lib/Covid/data/cdrlist.txt','r') as f:
                        for line in f:
                            cdrlist.append(line.split('\n')[0])
                sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
                aff_r = aff_r + b * var_r
                plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

            pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
            axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
            axs[1,m].set_xlabel('affinity score')
            axs[1,m].set_ylabel('true affinity')
            axs[1,m].legend()
            axs[1,m].set_title('gflow/GW:{}/beta:{}/pear:{}'.format(i,b,pearson))
    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_gflow_true_aff_hard_density_top_global:{}_initial:{}_rep:{}.png'.format(i,add_initial,replicate))


def analyze_gflow_true_aff_hard():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    multiple = 10
    budgets = [20,500]
    fig,axs = plt.subplots(len(budgets),5,figsize = (40,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5,multiple))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_gflow_true_aff_hard_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    for m in range(multiple):
                        seqs, score = get_gflow_true_aff_hard_seqs(weights[l],replicate)
                        seqs = list(set(seqs))
                        seqs = random.choices(seqs,k = budget)
                        all_aff = np.zeros(len(seqs))
                        all_sol = np.zeros(len(seqs))
                        all_ppost = np.zeros(len(seqs))
                        all_perf = np.zeros(len(seqs))
                        for j in range(len(seqs)):
                            r = dupl_dict[seqs[j]]
                            all_sol[j] = r[0]
                            all_aff[j] = r[1]
                            all_ppost[j] = r[2]
                            all_perf[j] = r[3]
                        results[i,f,k,l,0,m] = np.mean(all_sol)
                        results[i,f,k,l,1,m] = np.mean(all_perf)
                        results[i,f,k,l,2,m] = np.mean(all_ppost)
                        results[i,f,k,l,3,m] = diversity(seqs)
                        results[i,f,k,l,4,m] = novelty(init,seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                p_mean = np.mean(results[i,:,k,l,1],axis = 1)
                p_var = np.std(results[i,:,k,l,1],axis = 1)
                s_mean = np.mean(results[i,:,k,l,0],axis = 1)
                s_var = np.std(results[i,:,k,l,0],axis = 1)
                axs[i,0].plot(beta,p_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,0].fill_between(beta, (p_mean-2*p_var), (p_mean+2*p_var), alpha=.1)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta, p_var,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,1].set_ylabel('performance standard deviation')
                axs[i,1].set_xlabel('beta')
                axs[i,2].plot(beta,s_mean,label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,2].set_ylabel('solubility')
                axs[i,2].set_xlabel('beta')
                axs[i,3].plot(beta,np.mean(results[i,:,k,l,3],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,3].set_ylabel('diversity')
                axs[i,3].set_xlabel('beta')
                axs[i,4].plot(beta,np.mean(results[i,:,k,l,4],axis = 1),label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,4].set_ylabel('novelty')
                axs[i,4].set_xlabel('beta')
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_hard_analyze.png')

def analyze_gflow_true_aff_hard_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    aff_w = [0.85,1.0]
    add_initial = False
    replicate = 1
    budgets = [20,500]
    fig,axs = plt.subplots(len(budgets),4,figsize = (32,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_gflow_true_aff_hard_dict(gw,b,replicate)
                weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
                for l in range(len(weights)):
                    seqs, score = get_gflow_true_aff_hard_seqs(weights[l],replicate)
                    seqs = list(set(seqs))
                    all_aff = np.zeros(len(seqs))
                    all_sol = np.zeros(len(seqs))
                    all_ppost = np.zeros(len(seqs))
                    all_perf = np.zeros(len(seqs))
                    for j in range(len(seqs)):
                        r = dupl_dict[seqs[j]]
                        all_sol[j] = r[0]
                        all_aff[j] = r[1]
                        all_ppost[j] = r[2]
                        all_perf[j] = r[3]
                    aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist = remove_dominated(all_aff,all_sol)
                    dist = np.array(dist)
                    top_idx = np.argsort(dist)[:budget]

                    top_aff = all_aff[top_idx]
                    top_sol = all_sol[top_idx]
                    top_ppost = all_ppost[top_idx]
                    top_perf = all_perf[top_idx]
                    top_seqs = [seqs[i] for i in top_idx]
                    results[i,f,k,l,0] = np.mean(top_sol)
                    results[i,f,k,l,1] = np.mean(top_perf)
                    results[i,f,k,l,2] = np.mean(top_ppost)
                    results[i,f,k,l,3] = diversity(top_seqs)
                    results[i,f,k,l,4] = novelty(init,top_seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(aff_w)):
                axs[i,0].plot(beta,results[i,:,k,l,1],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta,results[i,:,k,l,0],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,1].set_ylabel('solubility')
                axs[i,1].set_xlabel('beta')
                axs[i,2].plot(beta,results[i,:,k,l,3],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,2].set_ylabel('diversity')
                axs[i,2].set_xlabel('beta')
                axs[i,3].plot(beta,results[i,:,k,l,4],label = 'budget:{}/aff_w:{}'.format(budgets[i],aff_w[l]))
                axs[i,3].set_ylabel('novelty')
                axs[i,3].set_xlabel('beta')
    plt.tight_layout()
    plt.savefig('./pareto_final/gflow_true_aff_hard_analyze_top.png')

## antBO HARD

def get_antBO_true_aff_hard_seqs(weights,replicate):
    seqs = []
    with open('./lib/dataset/gen_seqs/antBO/true_aff_hard/antBO_true_aff_hard_exp_lim_burn_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt'.format(weights[0],weights[1],weights[2],replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    score = np.load('./lib/dataset/gen_seqs/antBO/true_aff_hard/antBO_true_aff_hard_exp_lim_burn_prob_sol_min:{}_hum_min:{}_beta:{}_rep:{}.txt.npy'.format(weights[0],weights[1],weights[2],replicate))
    seqs = list(set(seqs))
    init_seqs,_,_,_,_ = get_init_true_aff_hard_no_dupl()
    seqs = [i for i in seqs if i not in init_seqs]
    return seqs,score

def get_all_antBO_true_aff_hard_seqs(beta,replicate):
    weights = [[-10.0,-120.0,beta],[4.0,-120.0,beta]]
    mcmc_seqs = []
    mcmc_score = np.array([])
    for w in weights:
        print(w)
        seqs,score = get_antBO_true_aff_hard_seqs(w,replicate)
        mcmc_seqs += seqs
        mcmc_score = np.concatenate((mcmc_score,score))
    return mcmc_seqs, mcmc_score

def preprocess_pareto_antBO_true_aff_hard(oracle):
    beta = [0.0]
    oracle.gen_all_cdr = True
    add_initial = False
    replicate = 1
    batch_size = 128
    for b in beta:
        seqs,score = get_all_antBO_true_aff_hard_seqs(b,replicate)
        seqs = list(set(seqs))
        n_batches = int(len(seqs)/batch_size) + 1
        sol_r = np.array([])
        aff_r = np.array([])
        ppost_r = np.array([])
        insta_r = np.array([])
        hydro_r = np.array([])
        charge_r = np.array([])
        perf_r = np.array([])
        for j in tqdm(range(n_batches)):
            a_r,valid,s_r,p_r = oracle.trust_region_true_aff(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            performance = oracle.true_aff_ora.score_without_noise_simple(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            instability,hydrophobicity,charge = oracle.get_dev_score(seqs[j*batch_size:min(len(seqs),(j+1)*batch_size)])
            sol_r = np.concatenate((sol_r,s_r))
            aff_r = np.concatenate((aff_r,a_r))
            ppost_r = np.concatenate((ppost_r,p_r))
            insta_r = np.concatenate((insta_r,instability))
            hydro_r = np.concatenate((hydro_r,hydrophobicity))
            charge_r = np.concatenate((charge_r,charge))
            perf_r = np.concatenate((perf_r,performance))
        with open('./lib/dataset/preprocess/antBO_true_aff_hard_seqs_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),'w') as f:
            for seq in seqs:
                f.write(seq + '\n')
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_aff_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),aff_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_sol_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),sol_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_ppost_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),ppost_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_insta_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),insta_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_hydro_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),hydro_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_charge_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),charge_r)
        np.save('./lib/dataset/preprocess/antBO_true_aff_hard_performance_preprocess_beta:{}_rep_{}.npy'.format(b,replicate),perf_r)

def get_antBO_true_aff_hard_no_dupl(beta,replicate):
    aff_r = np.load('./lib/dataset/preprocess/antBO_true_aff_hard_aff_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    sol_r = np.load('./lib/dataset/preprocess/antBO_true_aff_hard_sol_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    ppost_r = np.load('./lib/dataset/preprocess/antBO_true_aff_hard_ppost_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    perf_r = np.load('./lib/dataset/preprocess/antBO_true_aff_hard_performance_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate))
    seqs = []
    aff_r = aff_r - 12
    with open('./lib/dataset/preprocess/antBO_true_aff_hard_seqs_preprocess_beta:{}_rep_{}.npy'.format(beta,replicate),'r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs,aff_r,sol_r,ppost_r,perf_r

def get_antBO_true_aff_hard_dict(beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_antBO_true_aff_hard_no_dupl(beta,replicate)
    d = {}
    for i in range(len(seqs)):
        d[seqs[i]] = (sol_r[i],aff_r[i],ppost_r[i],perf_r[i])
    return d

def get_antBO_true_aff_hard_pareto_front(beta,replicate):
    seqs,aff_r,sol_r,ppost_r,perf_r = get_antBO_true_aff_hard_no_dupl(beta,replicate)
    pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
    return seqs,pareto_aff,pareto_sol,aff_r,sol_r,dist_from_pareto,perf_r


def plot_antBO_pareto_true_aff_hard_top():
    beta = [0.0]
    replicate = 1
    plt.figure(figsize = (10,6))
    for b in beta:
        print(b)
        seqs,aff_r,sol_r,ppost_r = get_antBO_true_aff_hard_no_dupl(b,replicate)
        pareto_aff,pareto_sol,aff_not_pareto,sol_not_pareto,dist_from_pareto = remove_dominated(aff_r,sol_r)
        plt.scatter(pareto_aff,pareto_sol,label = 'beta:{}'.format(b))
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        plt.plot(pareto_aff,pareto_sol)
    plt.ylabel('solubility score')
    plt.xlabel('affinity score')
    plt.title('antBO True Aff Hard Pareto Fronts')
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./pareto_final/antBO_true_aff_hard_compare_pareto_temp_rep:{}.png'.format(replicate))

def plot_antBO_pareto_true_aff_hard_density():
    global_weights = [10.0]
    beta = [0.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_antBO_true_aff_hard_pareto_front(b,replicate)
        idx = random.choices(range(len(seqs)),k = 20)
        seqs = [seqs[i] for i in idx]
        all_aff = all_aff[idx]
        all_sol = all_sol[idx]
        all_perf = all_perf[idx]
        axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
        print(all_aff)
        print(all_sol)
        x,y,z = get_color_plot(all_aff,all_sol)
        z = axs[0,m].scatter(x,y,c = z,label = 'antBO Hard generated seqs')
        plt.colorbar(z,ax = axs[0,m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        axs[0,m].set_xlabel('affinity score')
        axs[0,m].set_ylabel('solubility score')
        axs[0,m].legend()
        axs[0,m].set_title('antBO/beta:{}'.format(b))
        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
        axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
        axs[1,m].set_xlabel('affinity score')
        axs[1,m].set_ylabel('true affinity')
        axs[1,m].legend()
        axs[1,m].set_title('antBO/beta:{}/pear:{}'.format(b,pearson))

    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_antBO_true_aff_hard_density_initial:{}_rep:{}.png'.format(add_initial,replicate))

def plot_antBO_pareto_true_aff_hard_density_top():
    global_weights = [10.0]
    beta = [0.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(2,4,figsize = (32,12))
    for m in range(len(beta)):
        b = beta[m]
        seqs,pareto_aff,pareto_sol,all_aff,all_sol,dist,all_perf = get_antBO_true_aff_hard_pareto_front(b,replicate)
        idx = np.argsort(dist)[:1000]
        seqs = [seqs[i] for i in idx]
        all_aff = all_aff[idx]
        all_sol = all_sol[idx]
        all_perf = all_perf[idx]
        axs[0,3].scatter(all_perf,all_sol,label = 'beta : {}'.format(b))
        x,y,z = get_color_plot(all_aff,all_sol)
        z = axs[0,m].scatter(x,y,c = z,label = 'antBO generated seqs')
        plt.colorbar(z,ax = axs[0,m])
        #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
        idx = np.argsort(pareto_aff).tolist()
        pareto_aff = [pareto_aff[i] for i in idx]
        pareto_sol = [pareto_sol[i] for i in idx]
        axs[0,m].plot(pareto_aff,pareto_sol,label = 'empirical pareto front',color = 'r')
        axs[0,m].set_xlabel('affinity score')
        axs[0,m].set_ylabel('solubility score')
        axs[0,m].legend()
        axs[0,m].set_title('antBO/beta:{}'.format(b))
        if add_initial:
            cdrlist = []
            with open('./lib/Covid/data/cdrlist.txt','r') as f:
                    for line in f:
                        cdrlist.append(line.split('\n')[0])
            sasa_r,aff_r,var_r = oracle.return_indiv_scores(random.choices(cdrlist,k = 1000))
            aff_r = aff_r + b * var_r
            plt.scatter(aff_r,sasa_r, c = 'y', marker = '+',label = 'initial dataset')

        pearson = np.round(pearsonr(all_aff,all_perf)[0],2)
        axs[1,m].scatter(all_aff,all_perf,label = 'prediction vs affinity')
        axs[1,m].set_xlabel('affinity score')
        axs[1,m].set_ylabel('true affinity')
        axs[1,m].legend()
        axs[1,m].set_title('antBO/beta:{}/pear:{}'.format(b,pearson))
    axs[0,3].set_xlabel('true affinity')
    axs[0,3].set_ylabel('solubility')
    axs[0,3].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/pareto_antBO_true_aff_hard_density_top_initial:{}_rep:{}.png'.format(add_initial,replicate))


def analyze_antBO_true_aff_hard():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    sol_min = [-10.0,4.0]
    add_initial = False
    replicate = 1
    multiple = 10
    budgets = [10,20]
    fig,axs = plt.subplots(len(budgets),5,figsize = (40,6*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,5,multiple))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_antBO_true_aff_hard_dict(b,replicate)
                weights = [[3.0,-105.0,b],[4.0,-105.0,b]]
                for l in range(len(weights)):
                    for m in range(multiple):
                        seqs, score = get_antBO_true_aff_hard_seqs(weights[l],replicate)
                        seqs = list(set(seqs))
                        seqs = random.choices(seqs,k = budget)
                        all_aff = np.zeros(len(seqs))
                        all_sol = np.zeros(len(seqs))
                        all_ppost = np.zeros(len(seqs))
                        all_perf = np.zeros(len(seqs))
                        for j in range(len(seqs)):
                            r = dupl_dict[seqs[j]]
                            all_sol[j] = r[0]
                            all_aff[j] = r[1]
                            all_ppost[j] = r[2]
                            all_perf[j] = r[3]
                        results[i,f,k,l,0,m] = np.mean(all_sol)
                        results[i,f,k,l,1,m] = np.mean(all_perf)
                        results[i,f,k,l,2,m] = np.mean(all_ppost)
                        results[i,f,k,l,3,m] = diversity(seqs)
                        results[i,f,k,l,4,m] = novelty(init,seqs)

    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(sol_min)):
                p_mean = np.mean(results[i,:,k,l,1],axis = 1)
                p_var = np.std(results[i,:,k,l,1],axis = 1)
                s_mean = np.mean(results[i,:,k,l,0],axis = 1)
                s_var = np.std(results[i,:,k,l,0],axis = 1)
                axs[i,0].plot(beta,p_mean,label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,0].fill_between(beta, (p_mean-2*p_var), (p_mean+2*p_var), alpha=.1)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel('beta')
                axs[i,0].legend()
                axs[i,1].plot(beta, p_var,label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,1].set_ylabel('performance standard deviation')
                axs[i,1].set_xlabel('beta')
                axs[i,1].legend()
                axs[i,2].plot(beta,s_mean,label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,2].set_ylabel('solubility')
                axs[i,2].set_xlabel('beta')
                axs[i,2].legend()
                axs[i,3].plot(beta,np.mean(results[i,:,k,l,3],axis = 1),label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,3].set_ylabel('diversity')
                axs[i,3].set_xlabel('beta')
                axs[i,3].legend()
                axs[i,4].plot(beta,np.mean(results[i,:,k,l,4],axis = 1),label = 'budget:{}/sol_min:{}'.format(budgets[i],sol_min[l]))
                axs[i,4].set_ylabel('novelty')
                axs[i,4].set_xlabel('beta')
                axs[i,4].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/antBO_true_aff_hard_analyze.png')

def analyze_antBO_true_aff_hard_top():
    global_weights = [10.0]
    beta = [0.0]
    sol_min = [-10.0,4.0]
    add_initial = False
    replicate = 1
    budgets = [5,500]
    fig,axs = plt.subplots(len(budgets),4,figsize = (32,8*len(budgets)))
    results = np.zeros((len(budgets),len(beta),len(global_weights),2,10))
    init = get_true_aff_initial()
    for i in range(len(budgets)):
        budget = budgets[i]
        for f in range(len(beta)):
            b = beta[f]
            for k in range(len(global_weights)):
                gw = global_weights[k]
                dupl_dict = get_antBO_true_aff_hard_dict(b,replicate)
                weights = [[-10.0,-120.0,b],[4.0,-120.0,b]]
                for l in range(len(weights)):
                    seqs, score = get_antBO_true_aff_hard_seqs(weights[l],replicate)
                    seqs = list(set(seqs))
                    all_aff = np.zeros(len(seqs))
                    all_sol = np.zeros(len(seqs))
                    all_ppost = np.zeros(len(seqs))
                    all_perf = np.zeros(len(seqs))
                    for j in range(len(seqs)):
                        r = dupl_dict[seqs[j]]
                        all_sol[j] = r[0]
                        all_aff[j] = r[1]
                        all_ppost[j] = r[2]
                        all_perf[j] = r[3]
                    aff_pareto,sol_pareto,aff_not_pareto,sol_not_pareto,dist = remove_dominated(all_aff,all_sol)
                    dist = np.array(dist)
                    top_idx = np.argsort(dist)[:budget]

                    top_aff = all_aff[top_idx]
                    top_sol = all_sol[top_idx]
                    top_ppost = all_ppost[top_idx]
                    top_perf = all_perf[top_idx]
                    top_seqs = [seqs[i] for i in top_idx]
                    div_mean, div_std = diversity(top_seqs,return_std = True)
                    nov_mean, nov_std = novelty(init,top_seqs,return_std = True)
                    results[i,f,k,l,0] = np.mean(top_sol)
                    results[i,f,k,l,1] = np.mean(top_perf)
                    results[i,f,k,l,2] = np.mean(top_ppost)
                    results[i,f,k,l,3] = div_mean
                    results[i,f,k,l,4] = nov_mean
                    results[i,f,k,l,5] = np.std(top_sol)
                    results[i,f,k,l,6] = np.std(top_perf)
                    results[i,f,k,l,7] = np.std(top_ppost)
                    results[i,f,k,l,8] = div_std
                    results[i,f,k,l,9] = nov_std

    beta = np.array(beta)
    methods_shift = [-0.125,0.125]
    for i in range(len(budgets)):
        for k in range(len(global_weights)):
            for l in range(len(sol_min)):
                axs[i,0].errorbar(beta + methods_shift[l],results[i,:,k,l,1],yerr = results[i,:,k,l,6],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,0].bar(beta + methods_shift[l],results[i,:,k,l,1],label = r'budget:{} / $min sol$:{}'.format(budgets[i],sol_min[l]),width = 0.25)
                axs[i,0].set_ylabel('performance')
                axs[i,0].set_xlabel(r'$\beta$')
                axs[i,0].set_xticks(beta)
                axs[i,0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
                axs[i,1].errorbar(beta + methods_shift[l],results[i,:,k,l,0],yerr = results[i,:,k,l,5],capsize = 22,fmt = 'none',elinewidth = 5,ecolor = 'k')
                axs[i,1].bar(beta + methods_shift[l],results[i,:,k,l,0],label = r'budget:{} / $min sol$:{}'.format(budgets[i],sol_min[l]),width = 0.25)
                axs[i,1].set_ylabel('solubility')
                axs[i,1].set_xlabel(r'$\beta$')
                axs[i,1].set_xticks(beta)
                axs[i,2].errorbar(beta + methods_shift[l],results[i,:,k,l,3],yerr = results[i,:,k,l,8],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,2].bar(beta + methods_shift[l],results[i,:,k,l,3],label = r'$min sol$' + ': {}'.format(sol_min[l]),width = 0.25)
                axs[i,2].set_ylabel('Diversity')
                axs[i,2].set_xlabel(r'$\beta$')
                axs[i,2].set_xticks(beta)
                axs[i,2].set_ylim(0,10)
                axs[i,2].legend(loc='lower center', bbox_to_anchor=(0.15, 0.85), frameon = False)
                axs[i,3].errorbar(beta + methods_shift[l],results[i,:,k,l,4],yerr = results[i,:,k,l,9],capsize = 10,fmt = 'none',elinewidth = 5, ecolor = 'k')
                axs[i,3].bar(beta + methods_shift[l],results[i,:,k,l,4],label = r'$min sol$:' + ' {}'.format(sol_min[l]),width = 0.25)
                axs[i,3].set_ylabel('Novelty')
                axs[i,3].set_xlabel(r'$\beta$')
                axs[i,3].set_xticks(beta)
                #axs[i,3].legend(loc='lower center', bbox_to_anchor=(0.25, 0.8), frameon = False)
    plt.tight_layout()
    plt.savefig('./pareto_final/antBO_true_aff_hard_analyze_top.png')

## COMPARE TRUE AFF HARD

def compare_true_aff_hard_methods():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    budgets = [5,500]
    replicate = 1
    multiple = 1
    results = np.zeros((len(global_weights),len(beta),len(budgets),3,4,multiple))
    init = get_true_aff_initial()
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(global_weights)):
            gw = global_weights[j]
            mcmc_seqs,mcmc_pareto_aff,mcmc_pareto_sol,mcmc_all_aff,mcmc_all_sol,mcmc_dist,mcmc_all_perf = get_mcmc_true_aff_hard_pareto_front(gw,b,replicate)
            gflow_seqs,gflow_pareto_aff,gflow_pareto_sol,gflow_all_aff,gflow_all_sol,gflow_dist,gflow_all_perf = get_gflow_true_aff_hard_pareto_front(gw,b,replicate)
            antBO_seqs,antBO_pareto_aff,antBO_pareto_sol,antBO_all_aff,antBO_all_sol,antBO_dist,antBO_all_perf = get_antBO_true_aff_hard_pareto_front(b,replicate)
            for k in range(len(budgets)):
                budget = budgets[k]
                for n in range(multiple):
                    idx = random.choices(range(len(mcmc_seqs)),k = budget)
                    seqs = [mcmc_seqs[i] for i in idx]
                    sol = mcmc_all_sol[idx]
                    perf = mcmc_all_perf[idx]
                    results[j,i,k,0,0,n] = np.mean(perf)
                    results[j,i,k,0,1,n] = np.mean(sol)
                    results[j,i,k,0,2,n] = diversity(mcmc_seqs)
                    results[j,i,k,0,3,n] = novelty(init,mcmc_seqs)

                    idx = random.choices(range(len(gflow_seqs)),k = budget)
                    seqs = [gflow_seqs[i] for i in idx]
                    sol = gflow_all_sol[idx]
                    perf = gflow_all_perf[idx]
                    results[j,i,k,1,0,n] = np.mean(perf)
                    results[j,i,k,1,1,n] = np.mean(sol)
                    results[j,i,k,1,2,n] = diversity(gflow_seqs)
                    results[j,i,k,1,3,n] = novelty(init,gflow_seqs)

                    idx = random.choices(range(len(antBO_seqs)),k = budget)
                    seqs = [antBO_seqs[i] for i in idx]
                    sol = antBO_all_sol[idx]
                    perf = antBO_all_perf[idx]
                    results[j,i,k,2,0,n] = np.mean(perf)
                    results[j,i,k,2,1,n] = np.mean(sol)
                    results[j,i,k,2,2,n] = diversity(antBO_seqs)
                    results[j,i,k,2,3,n] = novelty(init,antBO_seqs)
    
    methods = ['mcmc','gflownet','antBO']
    task = ['performance','solubility','diversity','novelty']
    fig,axs = plt.subplots(len(task),len(budgets),figsize = (16,24))
    for i in range(len(global_weights)):
        for j in range(len(budgets)):
            for t in range(len(task)):
                axs[t,j].set_xlabel('beta')
                axs[t,j].set_ylabel(task[t])
                axs[t,j].set_title('budget:{}'.format(budgets[j]))
                for k in range(len(methods)):
                    axs[t,j].plot(beta,np.mean(results[i,:,j,k,t],axis = 1),label = methods[k])
                if t == 0 and j == 0:
                    axs[t,j].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_methods_true_aff_hard.png')


def compare_true_aff_hard_methods_top():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0]
    budgets = [5,500]
    replicate = 1
    multiple = 10
    results = np.zeros((len(global_weights),len(beta),len(budgets),3,8))
    init = get_true_aff_initial()
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(global_weights)):
            gw = global_weights[j]
            mcmc_seqs,mcmc_pareto_aff,mcmc_pareto_sol,mcmc_all_aff,mcmc_all_sol,mcmc_dist,mcmc_all_perf = get_mcmc_true_aff_hard_pareto_front(gw,b,replicate)
            gflow_seqs,gflow_pareto_aff,gflow_pareto_sol,gflow_all_aff,gflow_all_sol,gflow_dist,gflow_all_perf = get_gflow_true_aff_hard_pareto_front(gw,b,replicate)
            antBO_seqs,antBO_pareto_aff,antBO_pareto_sol,antBO_all_aff,antBO_all_sol,antBO_dist,antBO_all_perf = get_antBO_true_aff_hard_pareto_front(b,replicate)
            for k in range(len(budgets)):
                budget = budgets[k]
                idx = np.argsort(mcmc_dist)[:budget]
                seqs = [mcmc_seqs[i] for i in idx]
                sol = mcmc_all_sol[idx]
                perf = mcmc_all_perf[idx]
                results[j,i,k,0,0] = np.mean(perf)
                results[j,i,k,0,1] = np.mean(sol)
                results[j,i,k,0,2] = diversity(seqs)
                results[j,i,k,0,3] = novelty(init,seqs)
                results[j,i,k,0,4] = np.std(perf)
                results[j,i,k,0,5] = np.std(sol)
                results[j,i,k,0,6] = 0
                results[j,i,k,0,7] = 0

                idx = np.argsort(gflow_dist)[:budget]
                seqs = [gflow_seqs[i] for i in idx]
                sol = gflow_all_sol[idx]
                perf = gflow_all_perf[idx]
                results[j,i,k,1,0] = np.mean(perf)
                results[j,i,k,1,1] = np.mean(sol)
                results[j,i,k,1,2] = diversity(seqs)
                results[j,i,k,1,3] = novelty(init,seqs)
                results[j,i,k,1,4] = np.std(perf)
                results[j,i,k,1,5] = np.std(sol)
                results[j,i,k,1,6] = 0
                results[j,i,k,1,7] = 0

                idx = np.argsort(antBO_dist)[:budget]
                seqs = [antBO_seqs[i] for i in idx]
                sol = antBO_all_sol[idx]
                perf = antBO_all_perf[idx]
                results[j,i,k,2,0] = np.mean(perf)
                results[j,i,k,2,1] = np.mean(sol)
                results[j,i,k,2,2] = diversity(seqs)
                results[j,i,k,2,3] = novelty(init,seqs)
                results[j,i,k,2,4] = np.std(perf)
                results[j,i,k,2,5] = np.std(sol)
                results[j,i,k,2,6] = 0
                results[j,i,k,2,7] = 0
    
    methods_shift = [-0.07,0.0,0.07]
    beta = np.array(beta)
    methods = ['mcmc','gflownet','antBO']
    task = ['performance','solubility','diversity','novelty']
    fig,axs = plt.subplots(len(task),len(budgets),figsize = (16,24))
    for i in range(len(global_weights)):
        for j in range(len(budgets)):
            for t in range(len(task)):
                axs[t,j].set_xlabel(r'$\beta_{epi}$')
                axs[t,j].set_ylabel(task[t])
                axs[t,j].set_title('budget : {}'.format(budgets[j]))
                axs[t,j].set_xticks(beta)
                for k in range(len(methods)):
                    axs[t,j].errorbar(beta + methods_shift[k],results[i,:,j,k,t],yerr = results[i,:,j,k,t + 4], label = methods[k],capsize = 10,fmt = 'o-',elinewidth = 5)
                if t == 0 and j == 0:
                    axs[t,j].legend()
    plt.tight_layout()
    plt.savefig('./pareto_final/compare_methods_true_aff_hard_top.png')

def compare_true_aff_hard_top_box_plots():
    global_weights = [10.0]
    beta = [-1.0,0.0,1.0,2.0]
    methods = ['MCMC','GFlowNet','AntBO']
    sol_threshold = [-1.0,4.0,5.0]
    add_initial = False
    replicate = 1
    fig,axs = plt.subplots(1,7,figsize = (56,9))
    budget = 20
    seqs,aff_r,sol_r,ppost_r,perf_r = get_init_true_aff_hard_no_dupl()
    top_idx = np.argsort(perf_r)[:budget]
    perf_r = perf_r[top_idx]
    sol_r = sol_r[top_idx]
    random_seqs,random_aff_r,random_sol_r,random_ppost_r,random_perf_r = get_random_true_aff_hard_no_dupl()
    top_idx = np.argsort(random_aff_r)[-budget:]
    random_perf_r = random_perf_r[top_idx]
    random_sol_r = random_sol_r[top_idx]

    aff_threshold = np.linspace(1.0,-4)
    perc_better = np.zeros((len(sol_threshold),len(aff_threshold),len(beta),len(methods)))
    perc_better_init = np.zeros((len(sol_threshold),len(aff_threshold)))
    perc_better_random = np.zeros((len(sol_threshold),len(aff_threshold)))

    labels = []
    perfs = []
    sols = []
    for t in range(len(sol_threshold)):
        sol_t = sol_threshold[t]
        for t2 in range(len(aff_threshold)):
            aff_t = aff_threshold[t2]
            perc_better_init[t,t2] = perc_filter_seqs(sol_t,aff_t,perf_r,sol_r)
    labels.append('initial_mutants')
    perfs.append(perf_r)
    sols.append(sol_r)

    for t in range(len(sol_threshold)):
        sol_t = sol_threshold[t]
        for t2 in range(len(aff_threshold)):
            aff_t = aff_threshold[t2]
            perc_better_random[t,t2] = perc_filter_seqs(sol_t,aff_t,random_perf_r,random_sol_r)
    labels.append('random mutants')
    perfs.append(random_perf_r)
    sols.append(random_sol_r)

    count = 1
    for f in range(len(beta)):
        b = beta[f]
        for k in range(len(global_weights)):
            gw = global_weights[k]
            dupl_dict = get_mcmc_true_aff_hard_dict(gw,b,replicate)
            for m in range(len(methods)):
                if methods[m] == 'MCMC':
                    seqs,aff,sol,ppost,perf = get_mcmc_true_aff_hard_no_dupl(gw,b,replicate)
                elif methods[m] == 'GFlowNet':
                    seqs,aff,sol,ppost,perf = get_gflow_true_aff_hard_no_dupl(gw,b,replicate)
                else:
                    b = 0.0
                    seqs,aff,sol,ppost,perf = get_antBO_true_aff_hard_no_dupl(b,replicate)
                perfs.append(perf)
                sols.append(sol)
                for t in range(len(sol_threshold)):
                    sol_t = sol_threshold[t]
                    aff2,sol2,perf2 =  filter_seqs_sol(sol_t,aff,sol,perf)
                    top_idx = np.argsort(aff2)[-budget:]
                    top_perf = perf2[top_idx]
                    top_sol = sol2[top_idx]
                    for t2 in range(len(aff_threshold)):
                        aff_t = aff_threshold[t2]
                        perc = perc_filter_seqs(sol_t,aff_t,top_perf,top_sol)
                        perc_better[t,t2,f,m] = perc
                count += 1
                print(count)
                labels.append(r'$\beta$ : ' + '{}'.format(b) + r'/ $method$ : ' + '{}'.format(methods[m]))

    axs[0].boxplot(perfs,labels = labels)
    axs[0].set_xticklabels(labels,rotation = 90)
    axs[0].set_ylabel('True Affinity')
    axs[1].boxplot(sols,labels = labels)
    axs[1].set_xticklabels(labels,rotation = 90)
    axs[1].set_ylabel('Solubility')
    axs[2].violinplot(perfs)
    axs[2].set_xticks(np.arange(1, len(labels) + 1), labels=labels,rotation = 90)
    axs[2].set_ylabel('True Affinity')
    axs[3].violinplot(sols)
    axs[3].set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation = 90)
    axs[3].set_ylabel('Solubility')

    #perc_better = perc_better.reshape((len(sol_threshold),len(aff_threshold),-1))


    legend_elements = [Line2D([0], [0], linestyle = '-', color='pink',lw = 1, label=r'initial'),
                        Line2D([0], [0], linestyle = '-', color='orange',lw = 1, label=r'random'),
                        Line2D([0], [0], linestyle = '-', color='k', label='MCMC'),
                        Line2D([0], [0], linestyle = '--', color='k', label='GFlowNet'),
                        Line2D([0], [0], linestyle = ':', color='m', label='AntBO'),
                        Line2D([0], [0], linestyle = '-', color='b',lw = 1, label=r'$\beta$ = -1'),
                        Line2D([0], [0], linestyle = '-', color='y',lw = 1, label=r'$\beta$ = 0'),
                        Line2D([0], [0], linestyle = '-', color='r',lw = 1, label=r'$\beta$ = 1'),
                        Line2D([0], [0], linestyle = '-', color='g',lw = 1, label=r'$\beta$ = 2')]

    methods_fmt = ['-','--',':']
    beta_color = ['b','y','r','g']
    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(methods)):
            m = methods[j]
            #area = simpson(-perc_better[0,:,i,j], x = aff_threshold)
            fmt = methods_fmt[j]
            color = beta_color[i]
            if m == 'AntBO':
                color = 'm'
            axs[4].plot(-aff_threshold,perc_better[0,:,i,j],fmt,c = color)
    axs[4].plot(-aff_threshold,perc_better_init[0],c = 'pink')
    axs[4].plot(-aff_threshold,perc_better_random[0],c = 'orange')
    axs[4].set_ylabel('Number of Sequences above thresholds')
    axs[4].set_xlabel(r'$f^{min}_{aff}$')
    axs[4].set_title('No solubility threshold')
    #axs[4].legend(handles=legend_elements)


    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(methods)):
            m = methods[j]
            #area = simpson(-perc_better[1,:,i,j], x = aff_threshold)
            fmt = methods_fmt[j]
            color = beta_color[i]
            if m == 'AntBO':
                color = 'm'
            axs[5].plot(-aff_threshold,perc_better[1,:,i,j],fmt,c = color)
    axs[5].plot(-aff_threshold,perc_better_init[1],c = 'pink')
    axs[5].plot(-aff_threshold,perc_better_random[1],c = 'orange')
    axs[5].set_ylabel('Number of Sequences above thresholds')
    axs[5].set_xlabel(r'$f^{min}_{aff}$')
    axs[5].set_title(r'$f^{min}_{sol}$ = 4.0')

    for i in range(len(beta)):
        b = beta[i]
        for j in range(len(methods)):
            m = methods[j]
            #area = simpson(-perc_better[1,:,i,j], x = aff_threshold)
            fmt = methods_fmt[j]
            color = beta_color[i]
            if m == 'AntBO':
                color = 'm'
            axs[6].plot(-aff_threshold,perc_better[2,:,i,j],fmt,c = color)
    axs[6].plot(-aff_threshold,perc_better_init[2],c = 'pink')
    axs[6].plot(-aff_threshold,perc_better_random[2],c = 'orange')
    axs[6].set_ylabel('Number of Sequences above thresholds')
    axs[6].set_xlabel(r'$log_{10}$ ($T_{AFF}$ (nM))')
    axs[6].set_title(r'$T_{sol}$ = 5.0')
    axs[6].legend()

    plt.tight_layout()
    plt.savefig('./pareto_final/box_plots_compare_true_aff_hard_top_budget:{}.png'.format(budget))

def make_small_pareto():
    prop1 = np.random.normal(0.2, 0.2, 100000)
    prop2 = np.random.normal(0.2, 0.2, 100000)
    pareto1,pareto2,not_pareto1,not_pareto2,dist_from_pareto = remove_dominated(prop1,prop2)
    print(len(not_pareto1))
    idx = np.argsort(dist_from_pareto)[:500]
    not_pareto1 = prop1[idx]
    not_pareto2 = prop2[idx]
    x,y,z = get_color_plot(not_pareto1,not_pareto2)
    plt.scatter(x,y,c = z)
    #plt.scatter(pareto_aff,pareto_sol,color = 'r',label = 'pareto front seqs')
    idx = np.argsort(pareto1).tolist()
    pareto1 = [pareto1[i] for i in idx]
    pareto2 = [pareto2[i] for i in idx]
    plt.plot(pareto1,pareto2,'o-',c = 'r')
    plt.xlabel('Property 1')
    plt.ylabel('Property 2')
    plt.tick_params(
    axis = 'both',
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left = False,        # ticks along the top edge are off
    labelbottom=False,
    labelleft = False) # labels along the bottom edge are off
    plt.tight_layout()
    plt.savefig('./pareto_final/cartoon_pareto.png')


def main_mcmc_covid():
    #oracle = get_oracle(args)
    #preprocess_pareto_init_covid(oracle)
    #preprocess_pareto_mcmc(oracle)
    #preprocess_pareto_mcmc_combine(oracle)
    #d = get_mcmc_covid_dict(20.0,1.0,1)
    plot_mcmc_pareto_covid_top()
    #plot_mcmc_pareto_covid_density()
    #plot_mcmc_pareto_covid_density_top()
    #plot_mcmc_pareto_covid_density_combine()
    #plot_mcmc_pareto_covid_density_combine_top()
    #make_logos_mcmc_covid_combine()
    #get_chain_length_mcmc_covid()

def main_gflow_covid():
    #plot_spearmanr_losses()
    #seqs,score = get_all_gflow_covid_seqs(20.0,0.0,1)
    oracle = get_oracle(args)
    preprocess_pareto_gflow(oracle)
    preprocess_pareto_gflow_combine(oracle)
    #plot_gflow_pareto_covid_top()
    #plot_gflow_pareto_covid_density()
    #plot_gflow_pareto_covid_density_combine() #not necessary
    #plot_gflow_pareto_covid_density_combine_top() 
    #make_logos_gflow_covid_combine()

def main_compare_covid():
    #oracle = get_oracle(args)
    #plot_mcmc_gflow_compare_pareto_covid_top()
    #plot_compare_pareto_covid_density()
    #plot_compare_pareto_covid_density_top(oracle)
    #compare_covid_methods()
    compare_covid_methods_top()

def main_mcmc_true_aff():
    #oracle = get_oracle(args)
    #oracle.update_true_aff_gp(1.0,0.8,'simple')
    #preprocess_pareto_init_true_aff(oracle)
    #preprocess_pareto_random_true_aff(oracle)
    #preprocess_pareto_mcmc_true_aff(oracle)
    #plot_mcmc_pareto_true_aff_density()
    #plot_mcmc_pareto_true_aff_density_top()
    #plot_mcmc_pareto_true_aff_top()
    #plot_mcmc_true_aff_developability()
    #analyze_mcmc_true_aff()
    #analyze_mcmc_true_aff()
    #analyze_mcmc_true_aff_top()
    #find_seqs_in_trust()
    true_aff_top_box_plots()
    #true_aff_top_plot_error()

def main_gflow_true_aff():
    oracle = get_oracle(args)
    oracle.update_true_aff_gp(1.0,0.8,'simple')
    preprocess_pareto_gflow_true_aff(oracle)
    #plot_gflow_pareto_true_aff_density()
    #plot_gflow_pareto_true_aff_density_top()
    #analyze_gflow_true_aff()
    #analyze_gflow_true_aff_top()

def main_antBO_true_aff():
    #oracle = get_oracle(args)
    #oracle.update_true_aff_gp(1.0,0.8,'simple')
    #preprocess_pareto_antBO_true_aff(oracle)
    #plot_antBO_pareto_true_aff_density()
    #plot_antBO_pareto_true_aff_density_top()
    #plot_antBO_pareto_true_aff_top()
    #analyze_antBO_true_aff()
    analyze_antBO_true_aff_top()


def main_compare_true_aff():
    #compare_true_aff_methods()
    #compare_true_aff_methods_top()
    compare_true_aff_top_box_plots()
    
def main_mcmc_true_aff_hard():
    #oracle = get_oracle(args)
    #oracle.update_true_aff_gp(1.0,0.8,'hard')
    #preprocess_pareto_init_true_aff_hard(oracle)
    #preprocess_pareto_random_true_aff_hard(oracle)
    #preprocess_pareto_mcmc_true_aff_hard(oracle)
    #plot_mcmc_pareto_true_aff_hard_density()
    #plot_mcmc_pareto_true_aff_hard_density_top()
    #plot_mcmc_pareto_true_aff_top()
    #plot_mcmc_true_aff_developability()
    #analyze_mcmc_true_aff_hard()
    #analyze_mcmc_true_aff_hard()
    #analyze_mcmc_true_aff_hard_top()
    #find_seqs_in_trust()
    true_aff_hard_top_box_plots()
    #true_aff_hard_top_plot_error()

def main_gflow_true_aff_hard():
    oracle = get_oracle(args)
    oracle.update_true_aff_gp(1.0,0.8,'hard')
    preprocess_pareto_gflow_true_aff_hard(oracle)
    #plot_gflow_pareto_true_aff_hard_density()
    plot_gflow_pareto_true_aff_hard_density_top()
    #analyze_gflow_true_aff_hard()
    #analyze_gflow_true_aff_hard_top()   

def main_antBO_true_aff_hard():
    #oracle = get_oracle(args)
    #oracle.update_true_aff_gp(1.0,0.8,'hard')
    #preprocess_pareto_antBO_true_aff_hard(oracle)
    #plot_antBO_pareto_true_aff_hard_density()
    #plot_antBO_pareto_true_aff_hard_density_top()
    #plot_antBO_pareto_true_aff_top()
    #analyze_antBO_true_aff_hard()
    analyze_antBO_true_aff_hard_top()

def main_compare_true_aff_hard():
    #compare_true_aff_hard_methods()
    #compare_true_aff_hard_methods_top()
    compare_true_aff_hard_top_box_plots()

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.logger = get_logger(args)
    args.device = torch.device('cuda')
    plt.rcParams.update({'font.size': 16})
    #main_mcmc_covid()
    #main_gflow_covid()
    #main_compare_covid()
    #main_mcmc_true_aff()
    #main_gflow_true_aff()
    #main_antBO_true_aff()
    #main_compare_true_aff()
    #plot_spearmanr_losses_true_aff()
    main_mcmc_true_aff_hard()
    #main_gflow_true_aff_hard()
    #main_antBO_true_aff_hard()
    #main_compare_true_aff_hard()
    #make_small_pareto()
    #plot_spearmanr_losses()
    #plot_spearmanr_losses_true_aff()
    #plot_spearmanr_losses_true_aff_hard()
    #get_chain_length_mcmc_true_aff()
    #get_chain_length_mcmc_true_aff_hard()