import numpy as np
import matplotlib.pyplot as plt
from SASA_oracle import SASA_oracle_2
from hic_data import HICDataset
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import random
from sklearn.metrics import precision_recall_curve

def rmse(true,guess):
    return np.sqrt((true - guess)**2)

def subsample(y):
    n_pos = np.sum(y)
    n_neg = len(y) - n_pos
    idx_pos = [i for i in range(len(y)) if y[i] == 1]
    idx_neg = [i for i in range(len(y)) if y[i] == 0]
    idx_pos = random.choices(idx_pos,k = n_neg)
    return idx_neg + idx_pos

plt.rcParams.update({'font.size': 18})
dataset = HICDataset()
print(len(dataset))
oracle = SASA_oracle_2()
sasa_r = oracle(dataset.seqs)
print(sasa_r.shape)
print(np.mean(sasa_r))
sol_r = oracle.compute_score_without_SASA(dataset.seqs)
#sasa_camsol = oracle.use_custom_weights(dataset.seqs,dataset.camsol_per_aa_score)

analysis = [ProteinAnalysis(i) for i in dataset.seqs]
hydrophobicity = np.array([i.gravy() for i in analysis])

fig,ax = plt.subplots(5,figsize = (8,25))

pearsons = []
methods = ['HW \n + SASA', 'HW','CAMSOL', 'GRAVY']
spear = [0.40,0.15,0.33,0.05]
sp = spearmanr(sasa_r,dataset.hic)[0]
p_value = spearmanr(sasa_r,dataset.hic)[1]
print(p_value)
textstr = r'$r_s$: 0.40' + '\nP = {}'.format(np.format_float_scientific(p_value,precision = 2))
ax[0].text(-4.3, 13.8, textstr, fontsize=18,
verticalalignment='top')

#pearsons.append(pearsonr(sasa_r,dataset.hic)[0])
ax[0].scatter(sasa_r,dataset.hic)
print(spearmanr(sasa_r,dataset.hic)[0])
ax[0].set_xlabel(r'-$\hat f_{\rm sol}$')
ax[0].set_ylabel('HIC Retention Time (min)')

plt.tight_layout()
plt.savefig('./figures/hic_final.png')
plt.clf()

plt.figure(figsize = (8,5))
plt.xticks(range(4),labels = methods)
x = np.arange(4)
plt.bar(x,spear,width = 0.5)
plt.ylabel('Spearman\'s correlation')
plt.tight_layout()
plt.savefig('./figures/compare_solubility_methods.png')
