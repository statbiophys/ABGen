import tqdm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import random
import urllib.request
import os
from scipy.io import loadmat
from math import floor
from lib.Covid.covid_dataset import ESMTrainBestCovidDataset
from lib.Covid.covid_dataset import ESMTrainBestCovidDataset2
from lib.Covid.covid_dataset import args
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from Levenshtein import distance
from scipy.stats import gaussian_kde

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

def get_covid_gp_model(type_enc):
    a = args()
    dataset = ESMTrainBestCovidDataset(a)
    perc = 0.3
    task = 'AAYL49'
    train_x, train_y, valid_x, valid_y = dataset.random_split(type_enc,perc)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.load_state_dict(torch.load(os.path.join(script_dir, 'model_params','exactgp_param_{}_{}.pt'.format(type_enc,perc))))
    likelihood.load_state_dict(torch.load(os.path.join(script_dir, 'model_params','likelihood_exactgp_{}_{}.pt'.format(type_enc,perc))))
    model.eval()
    likelihood.eval()
    return model.cuda(),likelihood.cuda()

def get_color_plot(aff,sol):
    aff = np.array(aff)
    sol = np.array(sol)
    xy = np.vstack([aff,sol])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = aff[idx], sol[idx], z[idx]
    return x,y,z

def train_gp(type_enc,task,perc):
    a = args()
    if task == 'AAYL49':
        dataset = ESMTrainBestCovidDataset(a)
    else:
        dataset = ESMTrainBestCovidDataset2(a)
    train_x, train_y, valid_x, valid_y = dataset.random_split(type_enc,perc)
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iter = 600
    
    model.train()
    likelihood.train()

    model.covar_module.base_kernel.lengthscale = 6.0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training = False
    if training:
        for i in range(training_iter):
            # Zero gradients from previous iteration
            #idx = random.choices(range(train_x.shape[0]),k = mini_batch_size)
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
            if i % 50 == 0:
                pass
                #eval_model(model,valid_x,valid_y)

        torch.save(model.state_dict(), './model_params/exactgp_param_{}_{}_{}.pt'.format(type_enc,perc,task))
        torch.save(likelihood.state_dict(), './model_params/likelihood_exactgp_{}_{}_{}.pt'.format(type_enc,perc,task))
    else:
        model.load_state_dict(torch.load('./model_params/exactgp_param_{}_{}_{}.pt'.format(type_enc,perc,task)))
        likelihood.load_state_dict(torch.load('./model_params/likelihood_exactgp_{}_{}_{}.pt'.format(type_enc,perc,task)))

    valid_x = valid_x.cuda()
    
    model.eval()
    likelihood.eval()
    
    batch_size = 512
    num_it = int(valid_x.shape[0]/batch_size)
    with torch.no_grad():
        aff = torch.zeros((valid_x.shape[0]))
        epi_uncert = np.zeros((valid_x.shape[0]))
        uncert = np.zeros((valid_x.shape[0]))
        conf_upper = np.zeros((valid_x.shape[0]))
        conf_lower = np.zeros((valid_x.shape[0]))
        for i in range(num_it + 1):
            print(i)
            pred = model(valid_x[(i*batch_size):min(((i+1)*batch_size),valid_x.shape[0])])
            observed_pred = likelihood(pred)
            aff[i*batch_size:min(((i+1)*batch_size),valid_x.shape[0])] = pred.mean.cpu()
            epi_uncert[i*batch_size:min(((i+1)*batch_size),valid_x.shape[0])] = np.sqrt(pred.variance.cpu().numpy())
            uncert[i*batch_size:min(((i+1)*batch_size),valid_x.shape[0])] = 2 * np.sqrt(pred.variance.cpu().numpy() + likelihood.noise.item())
            lower, upper = observed_pred.confidence_region()
            conf_lower[i*batch_size:min(((i+1)*batch_size),valid_x.shape[0])] = lower.cpu() 
            conf_upper[i*batch_size:min(((i+1)*batch_size),valid_x.shape[0])] = upper.cpu()

    plt.rcParams.update({'font.size': 18})
    #plot prediction vs observed affinity
    plt.figure(figsize = (8,6))        
    pear = pearsonr(-valid_y,-aff)
    print(pear)
    r2 = r2_score(valid_y,aff)
    print(r2)

    l = np.linspace(-6,0)
    textstr = r'$\rho$: {}'.format(np.round(pear[0],2)) +  '\nP = {}'.format(np.format_float_scientific(pear[1],precision = 2))
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(-6.0, 0.75, textstr, fontsize=18,
        verticalalignment='top')
    x,y,z = get_color_plot(-aff,-valid_y)
    plt.plot(l,l)
    plt.scatter(x,y,c = z)
    plt.ylabel(r'$log_{10}$ (1 (nM) / $K_d$)')
    plt.xlabel(r'$\mu_{aff}$')
    #plt.title('{} Encoding/{} split Sequence'.format(type_enc,perc))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./figures/exactgppred_{}_{}_{}.png'.format(type_enc,perc,task))
    plt.clf()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    e = ['esm_t30']
    perc = [0.8]
    project = ['AAYL49']
    plt.rcParams.update({'font.size': 16})
    
    for type_enc in e:
        for p in perc:
            for i in project:
                train_gp(type_enc,i,p)