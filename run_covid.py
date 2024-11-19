import argparse
import gzip
import pickle
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from numpy.random import dirichlet
import random

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.oracle_wrapper import get_oracle

from lib.utils.env import get_tokenizer
from lib.Covid.covid_dataset import get_covid_seqs_gflow
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--saving", default = True, type = bool)
parser.add_argument("--loading", default = False, type = bool)
parser.add_argument("--saving_num", default = 11, type = int)
parser.add_argument("--loading_num", default = 11, type = int)
parser.add_argument("--gen_learning_rate", default=1e-3, type=float)
parser.add_argument("--gen_Z_learning_rate", default=5e-2, type=float)
parser.add_argument("--gen_num_iterations", default=50, type=int) # Maybe this is too low?
parser.add_argument("--update_learning_rate", default=4000, type=int)
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
parser.add_argument("--gen_model_type", default="cnn")
parser.add_argument("--use_replay_buffer", default=True, type=bool)
parser.add_argument("--log_results", default=True, type=bool)
parser.add_argument("--gen_random_action_prob", default=0.00, type=float)
parser.add_argument("--gen_all_cdr", default=True, type=bool)
parser.add_argument("--use_mcmc_seqs", default=False, type=bool)
parser.add_argument("--save_base", default=False, type=bool)
parser.add_argument("--load_base", default=False, type=bool)
parser.add_argument("--dataset_type", default = 'covid', type=str)

#global_weight = [1.0,0.2,1.8,10.0,1.0,1.0]

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
parser.add_argument("--gen_max_len", default=35)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=3, type=int)
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
parser.add_argument("--gen_partition_init", default=-100,type=float)
parser.add_argument("--pad_token_id", default=21, type=int)

#genbytenet args
parser.add_argument("--gen_small_embedding", default=16, type=int)
parser.add_argument("--cnn_hidden_size", default=32, type=int)
parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
parser.add_argument("--cnn_max_r", default=1, type=int)
parser.add_argument("--cnn_n_layers", default=4, type=int)


# Soft-QLearning/GFlownet gen
parser.add_argument("--gen_reward_exp_ramping", default=1, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=1, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
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

'''
Seqs passed to the generator to update loss have an end of sequence token
Seqs passed to the oracle do not
'''

class MbStack:

    """An implementation of a stack to store generated sequences and their reward.

    Attributes:
        stack: A list used to store tuples if sequences and indices
        f: an oracle function that computes the score of an amino acid sequence
    """

    def __init__(self, f):
        """Initializes the stack based on the oracle function.

        Args:
          f: oracle function
        """
        self.stack = []
        self.f = f

    def push(self, x, i):
        """Add an amino acid sequence and its index to the stack.

        Args:
          x: amino acid sequence
          i: index
        """
        self.stack.append((x, i))

    def pop_all(self):
        """Computes the reward of every sequence in the stack and returns it with the associated index.

        Returns:
          ys: list of rewards of amino acid sequences
          idxs: index of the sequence
        """
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack])
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)


def filter_len(x, y, max_len):
    """Removes from a list the amino acid sequences that are longer than a certain length.

    Args:
        x: list amino acid seqs
        y: list of rewards
        max_len: maximum length allowed for an amino acid sequence

    Returns:
        res: tuple of two lists containing allowed amino acid seqs and their associated score
    """

    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) <= max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res


class RolloutWorker:
    """An implementation of a stack to store generated sequences and their reward.

    Attributes:
        oracle: A function that takes as input a batch of amino acid seqs and outputs a reward signal
        max_len: The maximum length allowed for an amino acid seq, including eos token
        episodes_per_step: The number of sequences generated in a batch
        random_action_prob: The probability of taking a random uniform action
        reward_exp: Exponent used to amplify the reward signal
        sampling_temperature: TO-DO
        eos_tok: The token found at the end of every amino acid sequence
        tokenizer: A class used to map amino acid space to integer token space
        device: used to indicate whether the model should be trained using CPU or a GPU
        workers: A stack used to compute the score of the amino acid sequences generated
    """
    def __init__(self, args, oracle, tokenizer):
        self.oracle = oracle
        self.max_len = args.gen_max_len
        self.episodes_per_step = args.gen_episodes_per_step
        self.random_action_prob = args.gen_random_action_prob
        self.reward_exp = args.gen_reward_exp
        self.sampling_temperature = args.gen_sampling_temperature
        self.eos_tok = 21
        self.out_coef = args.gen_output_coef
        self.eos_char = tokenizer.eos_token

        #self.balanced_loss = args.gen_balanced_loss == 1
        #self.reward_norm = args.gen_reward_norm
        #self.reward_min = torch.tensor(float(args.gen_reward_min))
        #self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        #self.leaf_coef = args.gen_leaf_coef
        self.exp_ramping_factor = args.gen_reward_exp_ramping
        
        self.tokenizer = tokenizer
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = args.device
        self.args = args
        self.workers = MbStack(oracle)

    def rollout(self, model, episodes, use_rand_policy=True):
        """A method that uses an autoregressive model to generate amino acid sequences.

        Args:
            model: an autoregressive model that takes as input an amino acid seq and outputs logits to generate the next amino
                    acid in the sequence
            episodes: number of sequences to generate
            use_rand_policy: if True, may use random policy to generate amino acid sequence with probability determined
                                by random_aciton_prob

        Returns:
            visited: an empty list
            states: list of strings that contain amino acid sequences
            traj_states: list of lists containing the trajectories generated by the autoregressive model
            traj_actions: list of lists. Each inner list contains the actions chosen over a trajectory
            traj_rewards: list of lists containing no information
            traj_dones: list of lists. Each inner lists contains only 0s except for the position where the last amino acid
                        is added. This position is marked with a one
        """

        visited = []
        lists = lambda n: [list() for i in range(n)]
        states = [''] * episodes
        traj_states = [[''] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)
        for t in (range(self.max_len - 2) if episodes > 0 else []):
            active_indices = np.int32([i for i in range(episodes)
                                       if not states[i].endswith(self.eos_char)])
            x = [states[i] for i in active_indices]
            lens = torch.tensor([len(i) for i in states
                                if not i.endswith(self.eos_char)]).long().to(self.device)
            with torch.no_grad():
                logits = model(x, lens, index = t)
                #print(logits)
            try:
                cat = Categorical(logits=logits)
            except Exception as e:
                import pdb; pdb.set_trace()
            actions = cat.sample()
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0,1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            chars = [self.tokenizer.vocab.itos[i.item()] for i in actions]
            
            # Append predicted characters for active trajectories
            for i, c, a in zip(active_indices, chars, actions):
                if c == self.eos_char or t == self.max_len - 3:
                    self.workers.push(states[i] + (c if c != self.eos_char else ''), i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + c)
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += c
            if all(i.endswith(self.eos_char) for i in states):
                break
        return visited, states, traj_states, traj_actions, traj_rewards, traj_dones

    def prob_of_generation(self,generator,states,episodes):
        """A method that computes the log probability of a sequence being generated by an autoregressive model.

        Args:
            generator: an autoregressive model that takes as input an amino acid seq and outputs logits to generate the next amino
                    acid in the sequence
            states: list of strings representing amino acid sequence
            episodes: size of batch

        Returns:
            probs: numpy array containing log probability of generating a sequence
        """
        probs = np.zeros(episodes)
        active_indices = np.int32([i for i in range(episodes)
                                       if not states[0] == (self.eos_char)])
        for t in range(args.gen_max_len-1):
            active_indices = np.int32([i for i in active_indices
                                       if (len(states[i]) > t)])
            active_indices = np.int32([i for i in active_indices
                                       if (not states[i][t] == (self.eos_char))])
            if len(active_indices) > 0:
                x = [states[i][0:t] for i in active_indices]
                y = [states[i] for i in active_indices]
                y = [i[t] for i in y]
                y = self.tokenizer.process(y)
                lens = torch.tensor([t for i in active_indices]).long().to(self.device)
                with torch.no_grad():
                    logits = generator(x, lens, index = t)
            else:
                break
            try:
                cat = Categorical(logits=logits)
                y = y[:,0]
                n = len(active_indices)
                p = cat.probs[torch.arange(n, device=self.device),(y)]
                probs[active_indices] = probs[active_indices]+np.log(p.cpu().numpy())

                #print(model.Z)

            except Exception as e:
                print(states)
                print(x)
                print(logits)
                print(list(model.model.parameters()))
                print(e)
                import pdb; pdb.set_trace()
        return probs

    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True,sampling = False):
        """A method that uses rollout to generate a batch of new sequences, computes their score using the oracle and 
            combines them with previously evaluated seqs in the dataset.

        Args:
            model: an autoregressive model that takes as input an amino acid seq and outputs logits to generate the next amino
                    acid in the sequence
            dataset: a container of sequences and their score. Has a sample function to return randomply selected seqs
            use_rand_policy: used to indicate whether to use a random policy as part of the generation process

        Returns:
            visited: numpy array containing log probability of generating a sequence
            states: list of strings that contain amino acid sequences
            traj_states: list of lists containing the trajectories generated by the autoregressive model
            traj_actions: list of lists. Each inner list contains the actions chosen over a trajectory
            traj_rewards: list of lists containing no information
            traj_dones: list of lists. Each inner lists contains only 0s except for the position where the last amino acid
                        is added. This position is marked with a one
            bulk_trajs: list of tuples containing an amino acid sequence with eos token and the associated reward
                        of the form [('AAAA%',0.0),('ABAA%',1.0)]
        """
        # run an episode
        lists = lambda n: [list() for i in range(n)]
        visited, states, traj_states, \
            traj_actions, traj_rewards, traj_dones = self.rollout(model, self.episodes_per_step, use_rand_policy=use_rand_policy) 
        lens = np.mean([len(i) for i in traj_rewards])
        bulk_trajs = []
        rq = []
        
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = self.l2r(r, it)
            rq.append(r.item())
            s = states[mbidx]
            s = s + (self.eos_char if not s.endswith(self.eos_char) else '')
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))

            if args.use_replay_buffer and not sampling:
                dataset.add_seq(states[mbidx],r.item())
        
        #add previously sampled seqs to the output
        if args.gen_data_sample_per_step > 0 and dataset is not None:
            n = args.gen_data_sample_per_step
            m = len(traj_states)
            if self.args.proxy_type == "classification":
                x, y = dataset.sample(n, 0.5)
            elif self.args.proxy_type == "regression":
                if self.args.use_mcmc_seqs:
                    x, y = dataset.sample_mcmc(n)
                else:
                    x, y = dataset.sample(n)
            x, y = filter_len(x, y, self.max_len)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            bulk_trajs += list(zip([i+self.eos_char for i in x],
                                   [self.l2r(torch.tensor(i), it) for i in y]))
            for i in range(len(x)):
                traj_states[i+m].append('')
                for c, a in zip(x[i] + self.eos_char, self.tokenizer.process([x[i] + self.eos_char])[0]-2):
                    traj_states[i+m].append(traj_states[i+m][-1] + c)
                    traj_actions[i+m].append(a)
                    traj_rewards[i+m].append(0 if c != self.eos_char else self.l2r(y[i], it))
                    traj_dones[i+m].append(float(c == self.eos_char))
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }

def kl_div(p,q,z1,z2):
    return np.mean(q-p)

def train_generator(args, generator, oracle, tokenizer, dataset, global_weight):
    """A method that updates the parameters of the generator.

        Args:
            args: 
            generator: an autoregressive model that takes as input an amino acid seq and outputs logits to generate the next amino
                    acid in the sequence
            oracle: function that maps an amino acid seq to its reward
            tokenizer: function that maps amino acid seqs to integer space and back
            dataset: a container of sequences and their score. Has a sample function to return randomply selected seqs

        Returns:
            rollout_worker: 
        """
    print("Training generator")
    visited = []
    spearmanr_mcmc = []
    kl_debug = []
    rollout_worker = RolloutWorker(args, oracle, tokenizer)
    d_weights = np.array(global_weight)
    if args.gen_all_cdr:
        oracle.gen_all_cdr = True
        oracle.use_hum = True
    oracle.set_weights(d_weights)
    oracle.lim_dist = True
    for it in tqdm(range(args.gen_num_iterations)):
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset)
        visited.extend(rollout_artifacts["visited"])
        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])        
        if it % 100 == 1:
            print(rollout_artifacts["trajectories"]["bulk_trajs"])
            print('partition function estimate:{}'.format(generator.Z))
            print('new weights:{}'.format(d_weights))
            generated_seqs = [i[0][:-1] if i[0][-1] == '%' else i[0] for i in rollout_artifacts["trajectories"]["bulk_trajs"]]
            sol_r,aff_r,var_r = oracle.return_indiv_scores(generated_seqs)
            print(np.mean(aff_r + d_weights[4] * var_r))
            print(np.mean(sol_r))
            train_seqs, train_scores = dataset.sample(256)
            valid_seqs, valid_scores = dataset.sample_valid(256)
            mcmc_seqs, mcmc_scores = dataset.sample_mcmc(256)
            gen_prob_train = rollout_worker.prob_of_generation(generator,train_seqs,len(train_seqs))
            gen_prob_valid = rollout_worker.prob_of_generation(generator,valid_seqs,len(valid_seqs))
            gen_prob_mcmc = rollout_worker.prob_of_generation(generator,mcmc_seqs,len(mcmc_seqs))
            print('spearman score train {}'.format(spearmanr(gen_prob_train,train_scores)))
            print('spearman score valid {}'.format(spearmanr(gen_prob_valid,valid_scores)))
            print('spearman score mcmc train {}'.format(spearmanr(gen_prob_mcmc,mcmc_scores)))
            spearmanr_mcmc.append(spearmanr(gen_prob_mcmc,mcmc_scores)[0])
            if args.saving:
                if args.save_base:
                    torch.save(generator.state_dict(), './model_parameters/base_rep:{}.pt'.format(args.seed))
                else:
                    torch.save(generator.state_dict(), './model_parameters/generator_sol:{},aff:{},gw:{},beta:{},rep:{}.pt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed))
                    
        if it % args.update_learning_rate == 0 and it >= args.update_learning_rate:
            print('changing learning rate to {}'.format(args.gen_learning_rate/(2.0**(int(it/args.update_learning_rate)))))
            generator.change_learning_rate(args.gen_learning_rate/(2.0**(int(it/args.update_learning_rate))))
    
    train_seqs, train_scores = dataset.sample(256)
    valid_seqs, valid_scores = dataset.sample_valid(256)
    mcmc_seqs, mcmc_scores = dataset.sample_mcmc(256)
    gen_prob_train = rollout_worker.prob_of_generation(generator,train_seqs,len(train_seqs))
    gen_prob_valid = rollout_worker.prob_of_generation(generator,valid_seqs,len(valid_seqs))
    gen_prob_mcmc = rollout_worker.prob_of_generation(generator,mcmc_seqs,len(mcmc_seqs))
    print('spearman score train {}'.format(spearmanr(gen_prob_train,train_scores)))
    print('spearman score valid {}'.format(spearmanr(gen_prob_valid,valid_scores)))
    print('spearman score mcmc train {}'.format(spearmanr(gen_prob_mcmc,mcmc_scores)))

    return rollout_worker, spearmanr_mcmc


def filter_samples(samples, scores):
    idx = [i for i in range(len(samples)) if len(samples[i]) == 33]
    samples = [samples[i] for i in idx]
    scores = [scores[i] for i in idx]
    return samples,scores


def sample_batch(args, rollout_worker, generator, current_dataset, oracle, global_weight):
    """A method that samples amino acid sequences from the generator, computes their score using the oracle and
        plots the different reward signals.

        Args:
            args: 
            rollout_worker
            generator
            current_dataset
            oracle 
        Returns:
            rollout_worker: 
    """
    print("Generating samples")
    samples = ([], [])
    scores = []
    rollout_worker.episodes_per_step = 64
    sample_n = 2560
    while len(samples[0]) < sample_n:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, use_rand_policy = False,sampling = True)
        states = rollout_artifacts["trajectories"]["states"]
        samples[0].extend(states)
        scores.extend([rews[-1] for rews in rollout_artifacts["trajectories"]["traj_rewards"]])
    seqs = [i[:-1] if i[-1] == '%' else i for i in samples[0]]
    seqs,scores = filter_samples(seqs,scores)
    if args.log_results:
        with open('./lib/dataset/gen_seqs/gflownet/covid/gflow_sol:{}_aff:{}_global:{}_beta:{}_rep:{}.txt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed),'w') as f:
            for s in seqs:
                f.write('{}\n'.format(s))
        np.save('./lib/dataset/gen_seqs/gflownet/covid/gflow_scores_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed),scores)

    rollout_worker.episodes_per_step = args.gen_episodes_per_step

def log_results(aff_r,sol_r):
    with open('./pareto_final.txt','a') as f:
        for i in range(len(aff_r)):
            f.write('{},{},{}\n'.format(aff_r[i],sol_r[i],args.saving_num))

def _top_k(data, scores, k):
        topk_scores, topk_prots = [], []
        indices = np.argsort(scores)[::-1][:k]
        print(indices)
        topk_scores = np.concatenate((topk_scores, scores[indices]))
        topk_prots = np.concatenate((topk_prots, np.array(data)[indices]))
        return topk_prots.tolist(), topk_scores

def top_k(data, scores, k):
        print(data)
        print(scores)
        return _top_k(data, scores, k)

def fix_random(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def train(args, oracle, dataset, global_weight):
    """A method that trains an autoregressive generator of amino acid sequences
        .

        Args:
            args: 
            oracle
            dataset 
        Returns:
            rollout_worker: 
    """
    tokenizer = get_tokenizer(args)
    fix_random(args)
    for round in range(args.num_rounds):
        generator = get_generator(args, tokenizer)
        d_weights = np.array(global_weight)
        dataset.set_weights(d_weights)
        if args.loading:
            if args.load_base:
                generator.load_state_dict(torch.load('./model_parameters/base_rep:{}.pt'.format(args.seed),map_location=args.device))
            else:
                generator.load_state_dict(torch.load('./model_parameters/generator_sol:{},aff:{},gw:{},beta:{},rep:{}.pt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed),map_location=args.device))
                spearman_mcmc_prior = np.load('./losses/spearmanr_mcmc_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed))
                dataset.load_dataset()
        
        rollout_worker, spearmanr_mcmc = train_generator(args, generator, oracle, tokenizer, dataset, global_weight)
        spearmanr_mcmc = np.array(spearmanr_mcmc)
        if args.loading:
            spearmanr_mcmc = np.concatenate((spearman_mcmc_prior,spearmanr_mcmc))
        batch = sample_batch(args, rollout_worker, generator, dataset, oracle, global_weight)
        if args.saving:
            if args.save_base:
                torch.save(generator.state_dict(), './model_parameters/base_rep:{}.pt'.format(args.seed))
            else:
                torch.save(generator.state_dict(), './model_parameters/generator_sol:{},aff:{},gw:{},beta:{},rep:{}.pt'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed))
                np.save('./losses/spearmanr_mcmc_sol:{}_aff:{}_gw:{}_beta:{}_rep:{}.npy'.format(global_weight[1],global_weight[2],global_weight[3],global_weight[4],args.seed),spearmanr_mcmc)
                dataset.save_dataset()

def main(args):
    fix_random(args)
    args.device = torch.device('cuda')
    GW = [20.0,25.0,30.0]
    beta = [-1.0,0.0,1.0,2.0]
    for gw in GW:
        for b in beta:
            weights = [[1.0,0.15,0.85,gw,b,1.0],[1.0,0.125,0.875,gw,b,1.0],[1.0,0.1,0.9,gw,b,1.0],[1.0,0.05,0.95,gw,b,1.0],[1.0,0.0,1.0,gw,b,1.0]]
            for w in weights:
                print('Training GFlownet for beta = {} and inverse temperature = {}'.format(b,gw))
                oracle = get_oracle(args)
                dataset = get_dataset(args, oracle)
                train(args, oracle, dataset, w)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)