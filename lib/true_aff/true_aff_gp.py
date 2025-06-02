import tqdm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

import urllib.request
import os
from scipy.io import loadmat
from math import floor
#from true_aff_data import TrueAffDataset
from lib.true_aff.true_aff_data import TrueAffDataset
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from devices import get_device


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def update_gp_model(gp,new_train_x,new_train_y):
    device = get_device()
    train_x = gp.train_x
    train_y = gp.train_y
    train_x = torch.cat((train_x,new_train_x))
    train_y = torch.cat((train_y,new_train_y))
    print('size of dataset')
    print(len(train_x))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.load_state_dict(torch.load('./model_params/exactgp_param_{}_{}_{}.pt'.format(var,dataset.method,perc),map_location = device))
    likelihood.load_state_dict(torch.load('./model_params/likelihood_exactgp_{}_{}_{}.pt'.format(var,dataset.method,perc),map_location = device))
    model.eval()
    likelihood.eval()
    return model.to(device),likelihood.to(device)

def get_trueaff_gp_model(var,perc,method):
    device = get_device()
    dataset = TrueAffDataset()
    dataset.method = method
    dataset.rescore(var)
    train_x, train_y, valid_x, valid_y = dataset.random_split(perc)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir,'model_params', 'exactgp_param_{}_{}_{}.pt'.format(var,dataset.method,perc))
    model.load_state_dict(torch.load(file_path,map_location = device))
    file_path = os.path.join(script_dir,'model_params', 'likelihood_exactgp_{}_{}_{}.pt'.format(var,dataset.method,perc))
    likelihood.load_state_dict(torch.load(file_path,map_location = device))
    model.eval()
    likelihood.eval()
    return model.to(device),likelihood.to(device), torch.min(train_y).item()

def eval_model(model,valid_x,valid_y,mll):
    model.eval()
    device = get_device()
    X = valid_x.to(device)
    valid_y = valid_y.to(device)
    with torch.no_grad():
        pred = model(X)
        loss = -mll(pred, valid_y)
        print('validation loss:{} :'.format(loss.item()))
    model.train()

def train_gp_model(train_x,train_y,var,method,perc):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    device = get_device()
    model = model.to(device)
    likelihood = likelihood.to(device)

    training_iter = 150

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training = False
    if training:
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f var: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
                model.covar_module.outputscale
            ))
            optimizer.step()

        torch.save(model.state_dict(), './model_params/exactgp_param_{}_{}_{}.pt'.format(var,method,perc))
        torch.save(likelihood.state_dict(), './model_params/likelihood_exactgp_{}_{}_{}.pt'.format(var,method,perc))
    else:
        model.load_state_dict(torch.load('./model_params/exactgp_param_{}_{}_{}.pt'.format(var,method,perc),map_location = device))
        likelihood.load_state_dict(torch.load('./model_params/likelihood_exactgp_{}_{}_{}.pt'.format(var,method,perc),map_location = device))
    
    return model,likelihood

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    var_l = [0.0,1.0]
    #var_l = [0.0]
    perc = 0.8
    dataset = TrueAffDataset()
    dataset.method = 'hard'
    device = get_device()
    print(dataset.method)
    #fig, ax = plt.subplots(1,4,figsize = (34,8))
    for i in range(len(var_l)):
        var = var_l[i]
        dataset.rescore(var)
        train_x, train_y, valid_x, valid_y = dataset.random_split(perc)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        valid_x = valid_x.to(device)
        valid_y = valid_y.cpu().numpy()
        mut_count = dataset.valid_mut_count
        colormap = {0:'b',1:'r',2:'b',3:'g'}
        color = [colormap[key] for key in mut_count]
        model, likelihood = train_gp_model(train_x,train_y,var,dataset.method,perc)

        print(model.state_dict())
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            pred = model(valid_x)
            aff = pred.mean.cpu().numpy()
            true_score = dataset.true_valid_score
            plt.xlim(0,6)
            plt.ylim(-4,8)
            plt.scatter(aff,valid_y,c = color,marker = '+')
            plt.xlabel('prediction')
            plt.ylabel('true affinity with noise')
            plt.title('var_aff = {}'.format(var))
            print(spearmanr(aff,true_score))
            print(np.max(valid_y - true_score))
    plt.tight_layout()
    plt.savefig('./figures/exactgppred_{}_final_{}.png'.format(dataset.method,perc))
    plt.clf()
    '''

        with torch.no_grad():
            valid_x = dataset.seqs_enc_muts_2.to(device)
            pred = model(valid_x)
            aff = pred.mean.cpu().numpy()
            std = np.sqrt(pred.variance.cpu().numpy())
            valid_y = dataset.score_muts_2
            plt.xlim(0,6)
            plt.ylim(-4,8)
            print(aff.shape)
            print(valid_y.shape)
            plt.scatter(aff,valid_y,marker = '+')
            plt.xlabel('prediction')
            plt.ylabel('true affinity')
            plt.savefig('./figures/exactgppred_{}_final_n_muts:{}_var:{}.png'.format(dataset.method,2,var))
            print(spearmanr(aff,valid_y))
            plt.clf()
            counts,bins = np.histogram(std)
            plt.stairs(counts,bins)
            plt.savefig('./figures/histo_var_method:{}_n_muts:{}_var:{}.png'.format(dataset.method,2,var))
            plt.clf()

            valid_x = dataset.seqs_enc_muts_3.to(device)
            pred = model(valid_x)
            aff = pred.mean.cpu().numpy()
            std = np.sqrt(pred.variance.cpu().numpy())
            valid_y = dataset.score_muts_3
            plt.xlim(0,6)
            plt.ylim(-4,8)
            print(aff.shape)
            print(valid_y.shape)
            plt.scatter(aff,valid_y,marker = '+')
            plt.xlabel('prediction')
            plt.ylabel('true affinity')
            plt.savefig('./figures/exactgppred_{}_final_n_muts:{}_var:{}.png'.format(dataset.method,3,var))
            print(spearmanr(aff,valid_y))
            plt.clf()
            counts,bins = np.histogram(std)
            plt.stairs(counts,bins)
            plt.savefig('./figures/histo_var_method:{}_n_muts:{}_var:{}.png'.format(dataset.method,3,var))
            plt.clf()

            valid_x = dataset.seqs_enc_muts_4.to(device)
            pred = model(valid_x)
            aff = pred.mean.cpu().numpy()
            std = np.sqrt(pred.variance.cpu().numpy())
            valid_y = dataset.score_muts_4
            plt.xlim(0,6)
            plt.ylim(-4,8)
            print(aff.shape)
            print(valid_y.shape)
            plt.scatter(aff,valid_y,marker = '+')
            plt.xlabel('prediction')
            plt.ylabel('true affinity')
            plt.savefig('./figures/exactgppred_{}_final_n_muts:{}_var:{}.png'.format(dataset.method,4,var))
            print(spearmanr(aff,valid_y))
            plt.clf()
            counts,bins = np.histogram(std)
            plt.stairs(counts,bins)
            plt.savefig('./figures/histo_var_method:{}_n_muts:{}_var:{}.png'.format(dataset.method,4,var))
            plt.clf()
        
            valid_x = dataset.seqs_enc_muts_5.to(device)
            pred = model(valid_x)
            aff = pred.mean.cpu().numpy()
            std = np.sqrt(pred.variance.cpu().numpy())
            valid_y = dataset.score_muts_5
            plt.xlim(0,6)
            plt.ylim(-4,8)
            plt.scatter(aff,valid_y,marker = '+')
            plt.xlabel('prediction')
            plt.ylabel('true affinity')
            plt.savefig('./figures/exactgppred_{}_final_n_muts:{}_var:{}.png'.format(dataset.method,5,var))
            print(spearmanr(aff,valid_y))
            plt.clf()
            counts,bins = np.histogram(std)
            plt.stairs(counts,bins)
            plt.savefig('./figures/histo_var_method:{}_n_muts:{}_var:{}.png'.format(dataset.method,5,var))
            plt.clf()
        
            valid_x = dataset.seqs_enc_muts_6.to(device)
            pred = model(valid_x)
            aff = pred.mean.cpu().numpy()
            std = np.sqrt(pred.variance.cpu().numpy())
            valid_y = dataset.score_muts_6
            plt.xlim(0,6)
            plt.ylim(-4,8)
            plt.scatter(aff,valid_y,marker = '+')
            plt.xlabel('prediction')
            plt.ylabel('true affinity')
            plt.savefig('./figures/exactgppred_{}_final_n_muts:{}_var:{}.png'.format(dataset.method,6,var))
            print(spearmanr(aff,valid_y))
            plt.clf()
            counts,bins = np.histogram(std)
            plt.stairs(counts,bins)
            plt.savefig('./figures/histo_var_method:{}_n_muts:{}_var:{}.png'.format(dataset.method,6,var))
            plt.clf()
        '''
