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
from lib.Covid.covid_dataset import ESMTrainBestCovidDataset
from lib.Covid.covid_dataset import args
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

script_dir = os.path.dirname(__file__)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_covid_gp_model_12():
    a = args()
    dataset = ESMTrainBestCovidDataset(a) 
    train_x = dataset.seqs_mut12.cuda()
    train_y = dataset.score_mut12.cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    script_dir = os.path.dirname(__file__)
    model.load_state_dict(torch.load(os.path.join(script_dir, 'model_params','exactgp_param12.pt')))
    likelihood.load_state_dict(torch.load(os.path.join(script_dir, 'model_params','likelihood_exactgp12.pt')))
    model.eval()
    likelihood.eval()
    return model.cuda(),likelihood.cuda()

def eval_model(model,valid_x,valid_y):
    model.eval()
    X = valid_x.cuda()
    valid_y = valid_y.cuda()
    with torch.no_grad():
        pred = model(X)
        loss = -mll(pred, valid_y)
        print('validation loss:{} :'.format(loss.item()))
    model.train()

if __name__ == '__main__':
    a = args()
    dataset = ESMTrainBestCovidDataset(a) 
    train_x = dataset.seqs_mut12.cuda()
    train_y = dataset.score_mut12.cuda()
    test_x = dataset.seqs_mut3.cuda()
    test_y = dataset.score_mut3.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iter = 100

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
            if i % 10 == 0:
                eval_model(model,test_x,test_y)

        torch.save(model.state_dict(), './model_params/exactgp_param12.pt')
        torch.save(likelihood.state_dict(), './model_params/likelihood_exactgp12.pt')
    else:
        model.load_state_dict(torch.load('./model_params/exactgp_param12.pt'))
        likelihood.load_state_dict(torch.load('./model_params/likelihood_exactgp12.pt'))

    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        pred = model(test_x)
        aff = pred.mean.cpu()
        plt.plot(aff,test_y.cpu(),'r+')
        plt.savefig('./figures/exactgppred12.png')
    print(pearsonr(aff,test_y.cpu()))