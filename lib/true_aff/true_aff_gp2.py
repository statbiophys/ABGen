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
#from true_aff_data import TrueAffDataset2
from lib.true_aff.true_aff_data import TrueAffDataset2
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


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
    train_x = gp.train_x
    train_y = gp.train_y
    train_x = torch.cat((train_x,new_train_x))
    train_y = torch.cat((train_y,new_train_y))
    print('size of dataset')
    print(len(train_x))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(self.script_dir,'model_params', 'exactgp_param_2.pt')
    model.load_state_dict(torch.load(file_path))
    file_path = os.path.join(self.script_dir,'model_params', 'likelihood_exactgp_param_2.pt')
    likelihood.load_state_dict(torch.load(file_path))
    model.eval()
    likelihood.eval()
    return model.cuda(),likelihood.cuda()

def get_trueaff_gp_model_2():
    dataset = TrueAffDataset2()
    perc = 0.2
    train_x, train_y, valid_x, valid_y = dataset.random_split(perc)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(self.script_dir,'model_params', 'exactgp_param_2.pt')
    model.load_state_dict(torch.load(file_path))
    file_path = os.path.join(self.script_dir,'model_params', 'likelihood_exactgp_param_2.pt')
    likelihood.load_state_dict(torch.load(file_path))
    model.eval()
    likelihood.eval()
    return model.cuda(),likelihood.cuda()

def eval_model(model,valid_x,valid_y,mll):
    model.eval()
    X = valid_x.cuda()
    valid_y = valid_y.cuda()
    with torch.no_grad():
        pred = model(X)
        loss = -mll(pred, valid_y)
        print('validation loss:{} :'.format(loss.item()))
    model.train()

def train_gp_model(train_x,train_y):
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

    training = True
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

        torch.save(model.state_dict(), './model_params/exactgp_param2.pt')
        torch.save(likelihood.state_dict(), './model_params/likelihood_exactgp_2.pt')
    else:
        model.load_state_dict(torch.load('./model_params/exactgp_param_2.pt'))
        likelihood.load_state_dict(torch.load('./model_params/likelihood_exactgp_2.pt'))
    
    return model,likelihood

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    perc = 0.2

    fig, ax = plt.subplots(1,2,figsize = (8,8))
    dataset = TrueAffDataset2()
    train_x, train_y, valid_x, valid_y = dataset.random_split(perc)
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    valid_x = valid_x.cuda()
    valid_y = valid_y.cuda()
    
    model, likelihood = train_gp_model(train_x,train_y)

    print(model.state_dict())
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        X = valid_x.cuda()
        pred = model(X)
        aff = pred.mean.cpu()
        ax[1].plot(aff,valid_y.cpu(),'r+')
        ax[1].set_xlabel('prediction')
        ax[1].set_ylabel('true affinity with noise')
        print(spearmanr(aff,valid_y.cpu()))
    plt.tight_layout()
    plt.savefig('./figures/exactgppred_final2.png')
    plt.clf()

    '''
    with torch.no_grad():
        X = valid_x.cuda()
        pred = model(X)
        mean = pred.mean.cpu()
        var = pred.variance.cpu()
        observed_pred = likelihood(pred)
        lower, upper = observed_pred.confidence_region()
        noise = model.likelihood.noise.item()
        print(var[0:10])
        print(noise)
    '''
