import numpy as np
from Levenshtein import distance

best_seq = 'GYHLNSYGISIYSDGRRTFYGDSVGRAAGTFDF'
wt = 'GFTLNSYGISIYSDGRRTFYGDSVGRAAGTFDS'
aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
mut_pos = [i for i in range(len(best_seq)) if best_seq[i] == wt[i]]
print(len(mut_pos))

mutants = []
n_muts = 500
for i in range(n_muts):
    mut = list('GYHLNSYGISIYSDGRRTFYGDSVGRAAGTFDF')
    pos = np.random.choice(mut_pos,size = 3,replace = False)
    for p in pos:
        valid_aa = [j for j in aa_list if j != wt[p]]
        mut[p] = np.random.choice(valid_aa)
    mutants.append(''.join(mut))

with open('./mutants_of_best.txt','w') as f:
    for m in mutants:
        f.write('{}\n'.format(m))