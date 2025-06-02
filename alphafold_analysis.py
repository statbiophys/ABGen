import numpy as np
from lib.Covid.covid_dataset import args, ESMTrainBestCovidDataset
import matplotlib.pyplot as plt

def get_mean_indices(scores):
    mean_val = np.mean(scores)
    # Compute absolute differences from the mean
    diffs = np.abs(scores - mean_val)
    # Get indices of the 10 smallest differences (i.e., closest to the mean)
    centered_indices = np.argsort(diffs)[:20]
    return centered_indices

def list_sequences():
    dataset_args = args()
    dataset = ESMTrainBestCovidDataset(dataset_args)
    worse_idx = np.argsort(dataset.score)[-20:]
    top_idx = np.argsort(dataset.score)[:20]
    centered_idx = get_mean_indices(dataset.score)
    random_idx = np.random.randint(low = 0, high = len(dataset), size = 20)
    top_seqs = [dataset.seqlist[i] for i in top_idx]
    worse_seqs = [dataset.seqlist[i] for i in worse_idx]
    centered_seqs = [dataset.seqlist[i] for i in random_idx]
    top_scores = dataset.score[top_idx]
    worse_scores = dataset.score[worse_idx]
    centered_scores = dataset.score[random_idx]
    with open('./lib/dataset/alphafold_seqs.csv','w') as f:
        f.write(f'index,sequence,affinity\n')
        for i in range(len(top_seqs)):
            f.write(f'{i},{top_seqs[i]},{top_scores[i]}\n')
        for i in range(len(centered_seqs)):
            f.write(f'{i+20},{centered_seqs[i]},{centered_scores[i]}\n')
        for i in range(len(centered_seqs)):
            f.write(f'{i+40},{worse_seqs[i]},{worse_scores[i]}\n') 

import pandas as pd
import json
import os

def write_json_file():
    # --- User Input Section ---
    light_chain_sequence = "DVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK"
    antigen_peptide_sequence = "PDVDLGDISGINAS"
    beta = 2.0
    input_csv_path = f"./lib/dataset/generated_seqs_alphafold_beta:{beta}.csv"  # Path to your CSV
    output_json_path = f"./lib/dataset/af3_batch_jobs_generated_beta:{beta}.json"  # Output file with all jobs
    # --------------------------

    # Read the CSV
    df = pd.read_csv(input_csv_path)

    # Prepare all jobs
    all_jobs = []

    for idx, row in df.iterrows():
        heavy_chain_sequence = row["sequence"]
        job_name = f"Generated_{idx}_3"

        job = {
            "name": job_name,
            "modelSeeds": [],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": heavy_chain_sequence,
                        "count": 1
                    }
                },
                {
                    "proteinChain": {
                        "sequence": light_chain_sequence,
                        "count": 1
                    }
                },
                {
                    "proteinChain": {
                        "sequence": antigen_peptide_sequence,
                        "count": 1
                    }
                }
            ],
            "dialect": "alphafoldserver",
            "version": 1
        }

        all_jobs.append(job)

    # Write the final batch JSON file
    with open(output_json_path, 'w') as f:
        json.dump(all_jobs, f, indent=2)

    print(f"Saved {len(all_jobs)} AlphaFold 3 job entries to '{output_json_path}'")

def write_json_file_antBO():
    # --- User Input Section ---
    light_chain_sequence = "DVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK"
    antigen_peptide_sequence = "PDVDLGDISGINAS"
    beta = 0.0
    input_csv_path = f"./lib/dataset/generated_seqs_antBO_alphafold_beta:{beta}.csv"  # Path to your CSV
    output_json_path = f"./lib/dataset/af3_batch_jobs_generated_antBO_beta:{beta}.json"  # Output file with all jobs
    # --------------------------

    # Read the CSV
    df = pd.read_csv(input_csv_path)

    # Prepare all jobs
    all_jobs = []

    for idx, row in df.iterrows():
        heavy_chain_sequence = row["sequence"]
        job_name = f"Generated_{idx}_4"

        job = {
            "name": job_name,
            "modelSeeds": [],
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": heavy_chain_sequence,
                        "count": 1
                    }
                },
                {
                    "proteinChain": {
                        "sequence": light_chain_sequence,
                        "count": 1
                    }
                },
                {
                    "proteinChain": {
                        "sequence": antigen_peptide_sequence,
                        "count": 1
                    }
                }
            ],
            "dialect": "alphafoldserver",
            "version": 1
        }

        all_jobs.append(job)

    # Write the final batch JSON file
    with open(output_json_path, 'w') as f:
        json.dump(all_jobs, f, indent=2)

    print(f"Saved {len(all_jobs)} AlphaFold 3 job entries to '{output_json_path}'")

import os
import re
import json
import sys

def extract_iptm_initial():
    # Get root directory from argument or use current directory
    root_path = './lib/dataset/alphafold_pred'
    pattern = re.compile(r"initial_(\d+)")
    results = {}

    for folder in os.listdir(root_path):
        match = pattern.fullmatch(folder)
        folder_path = os.path.join(root_path, folder)

        if match and os.path.isdir(folder_path):
            idx = int(match.group(1))
            json_filename = f"fold_initial_{idx}_summary_confidences_0.json"
            json_path = os.path.join(folder_path, json_filename)

            if os.path.isfile(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        chain_iptm = data.get("chain_iptm", [])
                        if len(chain_iptm) >= 3:
                            results[idx] = chain_iptm[2]
                        else:
                            print(f"Warning: 'chain_iptm' has fewer than 3 entries in {json_filename}")
                except json.JSONDecodeError:
                    print(f"Error: Failed to parse JSON in {json_filename}")
            else:
                print(f"Missing: {json_filename}")

    return results

def extract_iptm_generated(beta = 0.0):
    # Get root directory from argument or use current directory
    root_path = './lib/dataset/alphafold_pred'
    if beta == 0.0:
        pattern = re.compile(r"generated_(\d+)")
    elif beta == -1.0:
        pattern = re.compile(r"generated_(\d+)_1")
    elif beta == 1.0:
        pattern = re.compile(r"generated_(\d+)_2")
    elif beta == 2.0:
        pattern = re.compile(r"generated_(\d+)_3")
    elif beta == 'antBO':
        pattern = re.compile(r"generated_(\d+)_4")
    results = {}

    for folder in os.listdir(root_path):
        match = pattern.fullmatch(folder)
        folder_path = os.path.join(root_path, folder)

        if match and os.path.isdir(folder_path):
            idx = int(match.group(1))
            if beta == 0.0:
                json_filename = f"fold_generated_{idx}_summary_confidences_0.json"
            elif beta == -1.0:
                json_filename = f"fold_generated_{idx}_1_summary_confidences_0.json"
            elif beta == 1.0:
                json_filename = f"fold_generated_{idx}_2_summary_confidences_0.json"
            elif beta == 2.0:
                json_filename = f"fold_generated_{idx}_3_summary_confidences_0.json"
            elif beta == 'antBO':
                json_filename = f"fold_generated_{idx}_4_summary_confidences_0.json"
            json_path = os.path.join(folder_path, json_filename)
            if os.path.isfile(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        chain_iptm = data.get("chain_iptm", [])
                        if len(chain_iptm) >= 3:
                            results[idx] = chain_iptm[2]
                        else:
                            print(f"Warning: 'chain_iptm' has fewer than 3 entries in {json_filename}")
                except json.JSONDecodeError:
                    print(f"Error: Failed to parse JSON in {json_filename}")
            else:
                print(f"Missing: {json_filename}")

    return results

def get_initial_affinity():
    dict_initial = {}
    with open('./lib/dataset/alphafold_seqs.csv','r') as f:
        f.readline()
        for line in f:
            values = line.split(',')
            dict_initial[values[0]] = float(values[2].strip())
    return dict_initial

def get_generated_affinity(beta = 0.0):
    gen_dict = {}
    with open(f'./lib/dataset/generated_seqs_alphafold_beta:{beta}.csv','r') as f:
        f.readline()
        for line in f:
            values = line.split(',')
            gen_dict[int(values[0])] = [float(values[2]),float(values[3])]
    return gen_dict

def get_generated_antBO(beta = 0.0):
    gen_dict = {}
    with open(f'./lib/dataset/generated_seqs_antBO_alphafold_beta:{beta}.csv','r') as f:
        f.readline()
        for line in f:
            values = line.split(',')
            gen_dict[int(values[0])] = [float(values[2]),float(values[3])]
    return gen_dict

from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_initial_alphafold(affinity_dict,alphafold_dict):
    affs = []
    iptm = []
    labels = []
    plt.figure(figsize = (8,6))
    for key in affinity_dict:
        if int(key) < 20:
            plt.scatter(alphafold_dict[int(key)],affinity_dict[key],c = 'r')
            affs.append(affinity_dict[key])
            iptm.append(alphafold_dict[int(key)])
            labels.append(1)
        elif int(key) < 40 and int(key) >= 20:
            plt.scatter(alphafold_dict[int(key)],affinity_dict[key],c = 'g')
            affs.append(affinity_dict[key])
            iptm.append(alphafold_dict[int(key)])
            if affinity_dict[key] < 1:
                labels.append(1)
            else:
                labels.append(0)
        else:
            plt.scatter(alphafold_dict[int(key)],affinity_dict[key],c = 'b')
            affs.append(affinity_dict[key])
            iptm.append(alphafold_dict[int(key)])
            labels.append(0)

    custom_handles = [
    Line2D([0], [0], marker='o', color='w', label=r'best $K_d$',
           markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label=r'random',
           markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label=r'worse $K_d$',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], color='grey', linestyle='--', label = 'Thresholds')
    ]
    correlation, p_value = spearmanr(affs, iptm)
    print(correlation)
    fpr, tpr, thresholds = roc_curve(labels,iptm)
    precision,recall, thresholds = precision_recall_curve(labels,iptm)
    print(precision)
    print(recall)
    print(thresholds)
    plt.text(0.47,4.5,rf'$r_s$ : {correlation.round(2)}' +  f'\nP-value = {p_value.round(2)}',fontsize = 12)
    plt.axvline(x=0.7, color='gray', linestyle='--', linewidth=2,label = 'ipTM threshold')
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,label = 'ipTM threshold')
    plt.xlabel('ipTM',fontsize = 18)
    plt.ylabel(r'$K_d$ ($log_{10}$ (1 (nM)))',fontsize = 18)
    plt.legend(handles=custom_handles,fontsize = 12,frameon = False)
    plt.tight_layout()
    plt.savefig('./pareto_final/alphafold_analysis_initial.png')

def plot_generated_alphafold(affinity_dict,alphafold_dict,beta = 0.0):
    affs = []
    iptm = []
    labels = []
    count = 0
    total = 0
    plt.figure(figsize = (8,6))
    for key in alphafold_dict:
        plt.scatter(alphafold_dict[int(key)],affinity_dict[key][0],c = 'r')
        affs.append(affinity_dict[key][0])
        iptm.append(alphafold_dict[int(key)])
        if alphafold_dict[int(key)] > 0.7:
            count += 1
        total += 1
    correlation, p_value = spearmanr(affs, iptm)
    correlation = correlation.round(2)
    p_value = p_value.round(2)
    print(correlation)
    print(count/total)
    plt.text(0.53,max(affs) - 0.05,rf'$r_s$: {correlation}' + '\n' + f'p-value: {p_value}',fontsize = 12)
    if beta == 0.0:
        plt.axvline(x=0.7, color='gray', linestyle='--', linewidth=2,label = 'ipTM threshold')
        plt.axhline(y=-1.0, color='gray', linestyle='--', linewidth=2,label = 'ipTM threshold')

    custom_handles = [
    Line2D([0], [0], marker='o', color='w', label=r'Generated sequences',
           markerfacecolor='red', markersize=10),
    Line2D([0], [0], color='grey', linestyle='--', label = 'Thresholds')
    ]

    plt.xlabel('ipTM',fontsize = 18)
    plt.ylabel(r'$\hat{f}_{\rm aff}$',fontsize = 18)
    plt.title(rf'$\beta$ = {beta}',fontsize = 18)
    plt.legend(handles = custom_handles, frameon = False,fontsize = 12)
    plt.tight_layout()
    plt.savefig(f'./pareto_final/alphafold_analysis_generated_beta:{beta}.png')
    plt.clf()


    above_aff = [affinity_dict[key][0] for key in alphafold_dict if alphafold_dict[int(key)] > 0.7]
    below_aff = [affinity_dict[key][0] for key in alphafold_dict if alphafold_dict[int(key)] <= 0.7]
    above_sol = [affinity_dict[key][1] for key in alphafold_dict if alphafold_dict[int(key)] > 0.7]
    below_sol = [affinity_dict[key][1] for key in alphafold_dict if alphafold_dict[int(key)] <= 0.7]
    plt.scatter(above_aff,above_sol,c = 'r', label = 'iptm > 0.7')
    plt.scatter(below_aff,below_sol,c = 'b', label = 'iptm <= 0.7')

    plt.xlabel(r'$\hat{f}_{\rm aff}$',fontsize = 18)
    plt.ylabel(r'$\hat{f}_{\rm sol}$',fontsize = 18)
    plt.title(rf'$\beta$ = {beta}',fontsize = 18)
    plt.ylim(2,7)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig(f'./pareto_final/alphafold_analysis_generated_pareto_beta:{beta}.png')

def count_above(sol_dict,gen_iptm,iptm_threshold,sol_threshold):
    count = 0
    for key in gen_iptm:
        if sol_dict[key][1] > sol_threshold and gen_iptm[key] > iptm_threshold:
            count += 1
    return count

def plot_blox_plots_alphafold():
    betas = [-1.0,0.0,1.0,2.0,'antBO']
    fig, ax = plt.subplots(1,2,figsize = (16,6))
    colors = ['blue','green','orange','red','purple']
    for k in range(len(betas)):
        b = betas[k]
        if b != 'antBO':
            gen_dict = get_generated_affinity(beta = b)
            gen_iptm = extract_iptm_generated(beta = b)
        else:
            gen_dict = get_generated_antBO(beta = 0.0)
            gen_iptm = extract_iptm_generated(beta = b)
        iptm_thresholds = np.linspace(0,1)
        count_no_sol = []
        count_sol = []
        for iptm_t in iptm_thresholds:
            count_no_sol.append(count_above(gen_dict,gen_iptm,iptm_t,-12))
        if b != 'antBO':
            ax[0].plot(iptm_thresholds,count_no_sol,label = rf'$\beta$ = {b}',c = colors[k])
        else:
            ax[0].plot(iptm_thresholds,count_no_sol,label = rf'antBO',c = 'purple',linestyle = ':')

        for iptm_t in iptm_thresholds:
            count_sol.append(count_above(gen_dict,gen_iptm,iptm_t,4.0))
        if b != 'antBO':
            ax[1].plot(iptm_thresholds,count_sol,label = rf'$\beta$ = {b}',c = colors[k])
        else:
            ax[1].plot(iptm_thresholds,count_sol,c = 'purple',linestyle = ':')
    ax[0].set_xlabel('iptm threshold',fontsize = 18)
    ax[0].set_ylabel('Number of sequences above thresholds',fontsize = 18)
    ax[0].set_title('No solubility threshold',fontsize = 18)
    ax[0].legend(fontsize = 12,frameon = False)

    ax[1].set_xlabel('iptm threshold',fontsize = 18)
    ax[1].set_ylabel('Number of sequences above thresholds',fontsize = 18)
    ax[1].set_title(r'$f_{\rm sol}^{\rm min}$ = 4.0',fontsize = 18)
    plt.tight_layout()
    plt.savefig('./pareto_final/alphafold_box_plots.png')

import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

def compute_diversity(beta = 0.0,ipTM_t = 0.7):
    root_path = './lib/dataset/alphafold_pred'

    iptm_scores = extract_iptm_generated(beta = beta)

    if beta == 0.0:
        cif_paths = [f"generated_{idx}/fold_generated_{idx}_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == -1.0:
        cif_paths = [f"generated_{idx}_1/fold_generated_{idx}_1_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == 1.0:
        cif_paths = [f"generated_{idx}_2/fold_generated_{idx}_2_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == 2.0:
        cif_paths = [f"generated_{idx}_3/fold_generated_{idx}_3_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == 'antBO':
        cif_paths = [f"generated_{idx}_4/fold_generated_{idx}_4_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    
    target_chain = "A"          # <- chain that all files share
    quiet      = True           # suppress Biopython's warnings

    subset_resnums = list(range(25,35)) + list(range(50,64)) + list(range(98,107)) 
    parser      = MMCIFParser(QUIET=True)
    atom_sets   = []          # [[Atom, Atom, …], …]
    resseq_sets = []          # [[10, 11, …], …] – keeps numbering parallel to atom_sets

    for idx, path in enumerate(cif_paths):
        chain = next(                         # first MODEL -> chosen chain
            parser.get_structure(f"m{idx}", os.path.join(root_path,path)).get_models()
        )[target_chain]

        ca_atoms  = []
        resseqs   = []
        for res in chain:
            if is_aa(res, standard=True) and "CA" in res:
                ca_atoms.append(res["CA"])
                resseqs.append(res.get_id()[1])      # (hetflag, resseq, icode)[1]

        atom_sets.append(ca_atoms)
        resseq_sets.append(resseqs)

    # sanity-check same length & same numbering across models
    if len({tuple(r) for r in resseq_sets}) != 1:
        raise ValueError("Residue lists differ between models – align them first!")

    # Pre-compute a *Boolean mask* telling us which positions belong to the user subset
    mask = np.array([r in subset_resnums for r in resseq_sets[0]])
    if not mask.any():
        raise ValueError("None of the requested residue numbers were found!")

    sup   = Superimposer()
    pairs = []
    for i in range(len(atom_sets)):
        for j in range(i + 1, len(atom_sets)):
            # 4a Align on the *whole* chain
            sup.set_atoms(atom_sets[i], atom_sets[j])
            rot, tran = sup.rotran    # rotation matrix & translation vector

            # 4b Extract coordinates *for the subset only*
            ref_coords   = np.array([a.get_coord() for a in atom_sets[i]])[mask]
            move_coords  = np.array([a.get_coord() for a in atom_sets[j]])[mask]
            move_coords  = np.dot(move_coords, rot) + tran    # apply alignment

            # 4c Compute subset RMSD by hand
            diff2 = ((ref_coords - move_coords) ** 2).sum(axis=1)
            rmsd_subset = np.sqrt(diff2.mean())

            pairs.append(((i, j), rmsd_subset))

    return np.mean([r for _, r in pairs]), np.std([r for _, r in pairs])

def compute_novelty(beta = 0.0,ipTM_t = 0.7):
    # ----------------- INPUT:  overwrite with your paths ---------------
    root_path = './lib/dataset/alphafold_pred'

    iptm_scores = extract_iptm_generated(beta = beta)

    if beta == 0.0:
        list_ref = [f"generated_{idx}/fold_generated_{idx}_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == -1.0:
        list_ref = [f"generated_{idx}_1/fold_generated_{idx}_1_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == 1.0:
        list_ref = [f"generated_{idx}_2/fold_generated_{idx}_2_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == 2.0:
        list_ref = [f"generated_{idx}_3/fold_generated_{idx}_3_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]
    elif beta == 'antBO':
        list_ref = [f"generated_{idx}_4/fold_generated_{idx}_4_model_0.cif" for idx in iptm_scores if iptm_scores[idx] >= ipTM_t]

    
    list_pool = [f"initial_{idx}/fold_initial_{idx}_model_0.cif" for idx in range(60)]   # second list
    chain_id  = "A"                           # chain present in every file

    # Optional: restrict RMSD to particular residue numbers
    subset_resnums = None                     # e.g. {10, 11, 15, 87}
                                            # → leave as None for whole chain

    # ---------------------------------------------------------------
    # 2.  Convenience helpers
    # ---------------------------------------------------------------

    def load_ca_atoms(cif_path, chain):
        """Return (atom_list, resseq_list) for the chosen chain."""
        struct = MMCIFParser(QUIET=True).get_structure("x", cif_path)
        chain  = next(struct.get_models())[chain]

        atoms, resseqs = [], []
        for res in chain:
            if is_aa(res, standard=True) and "CA" in res:
                atoms.append(res["CA"])
                resseqs.append(res.get_id()[1])               # resseq = integer
        return atoms, resseqs

    def rmsd_after_alignment(atoms_ref, atoms_mov, mask=None):
        """
        Align on *all* atoms in atoms_ref / atoms_mov.
        If mask is given (Boolean array), RMSD is computed only on masked positions.
        """
        sup = Superimposer()
        sup.set_atoms(atoms_ref, atoms_mov)       # Kabsch fit

        if mask is None:                          # whole chain RMSD already computed
            return sup.rms

        # otherwise: re-compute on the subset --------------------------
        rot, tran = sup.rotran

        A = np.array([a.get_coord() for a in atoms_ref])[mask]
        B = np.array([a.get_coord() for a in atoms_mov])[mask]
        B = np.dot(B, rot) + tran

        diff2 = ((A - B) ** 2).sum(axis=1)
        return float(np.sqrt(diff2.mean()))

    # ---------------------------------------------------------------
    # 3.  Parse all files once (faster for large sets)
    # ---------------------------------------------------------------
    atoms_ref, resseq_ref = zip(*(load_ca_atoms(os.path.join(root_path,p), chain_id) for p in list_ref))
    atoms_pool, resseq_pool = zip(*(load_ca_atoms(os.path.join(root_path,p), chain_id) for p in list_pool))

    # --- consistency check (same number & order of residues) --------
    if len({tuple(r) for r in resseq_ref}) != 1 or len({tuple(r) for r in resseq_pool}) != 1:
        raise ValueError("Residue numbering differs between files; "
                        "align sequences first or trim structures!")

    if resseq_ref[0] != resseq_pool[0]:
        raise ValueError("Reference and pool files do not share identical residue lists!")

    mask = None
    if subset_resnums is not None:               # build Boolean mask once
        mask = np.array([n in subset_resnums for n in resseq_ref[0]])
        if not mask.any():
            raise ValueError("None of the requested residue numbers were found!")

    # ---------------------------------------------------------------
    # 4.  Compute minimum RMSD for every reference structure
    # ---------------------------------------------------------------
    min_rmsd, best_match = {}, {}                # filename → rmsd / best partner

    for (fname_ref, atoms_r) in zip(list_ref, atoms_ref):
        best_val = np.inf
        best_file = None
        for (fname_pool, atoms_p) in zip(list_pool, atoms_pool):
            r = rmsd_after_alignment(atoms_r, atoms_p, mask)
            if r < best_val:
                best_val  = r
                best_file = fname_pool
        min_rmsd[fname_ref]  = best_val
        best_match[fname_ref] = best_file

    # ---------------------------------------------------------------
    # 5.  Nicely report
    # ---------------------------------------------------------------
    rmsds = [min_rmsd[ref] for ref in list_ref] 
    return np.mean(rmsds), np.std(rmsds)

def plot_diversity_novelty():
    """Plot diversity and novelty for two ipTM thresholds side-by-side."""

    betas = np.array([-1.0, 0.0, 1.0, 2.0])
    bar_width = 0.35         # a bit wider, so the two bars almost touch

    # ---- ipTM > 0.7 ----
    div_hi = [compute_diversity(beta=b)       for b in betas]
    nov_hi = [compute_novelty(beta=b)         for b in betas]
    div_hi_mean, div_hi_std = zip(*div_hi)
    nov_hi_mean, nov_hi_std = zip(*nov_hi)

    # ---- all trajectories (ipTM threshold = 0) ----
    div_all = [compute_diversity(beta=b, ipTM_t=0.0) for b in betas]
    nov_all = [compute_novelty(beta=b,  ipTM_t=0.0)  for b in betas]
    div_all_mean, div_all_std = zip(*div_all)
    nov_all_mean, nov_all_std = zip(*nov_all)

    # ── set up the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # Left panel – Diversity
    ax = axes[0]
    ax.bar(betas - bar_width / 2,
           div_hi_mean,
           yerr=div_hi_std,
           width=bar_width,
           capsize=5,
           label='ipTM > 0.7')
    ax.bar(betas + bar_width / 2,
           div_all_mean,
           yerr=div_all_std,
           width=bar_width,
           capsize=5,
           label='all')

    ax.set_xlabel(r'$\beta$', fontsize=18)
    ax.set_ylabel('Diversity (Å)', fontsize=18)
    ax.set_xticks(betas)
    ax.legend(frameon = False)

    # Right panel – Novelty
    ax = axes[1]
    ax.bar(betas - bar_width / 2,
           nov_hi_mean,
           yerr=nov_hi_std,
           width=bar_width,
           capsize=5,
           label='ipTM > 0.7')
    ax.bar(betas + bar_width / 2,
           nov_all_mean,
           yerr=nov_all_std,
           width=bar_width,
           capsize=5,
           label='all')

    ax.set_xlabel(r'$\beta$', fontsize=18)
    ax.set_ylabel('Novelty (Å)', fontsize=18)
    ax.set_xticks(betas)

    plt.tight_layout()
    plt.savefig('./pareto_final/alphafold_diversity.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    #list_sequences()
    #write_json_file()
    #write_json_file_antBO()
    #extract_iptm()
    '''
    d = get_initial_affinity()
    d2 = extract_iptm_initial()
    plot_initial_alphafold(d,d2)
    '''
    '''
    d = get_generated_affinity(beta = 2.0)
    d2 = extract_iptm_generated(beta = 2.0)
    plot_generated_alphafold(d,d2,beta = 2.0)
    '''
    #plot_blox_plots_alphafold()
    #compute_diversity(beta = 1.0)
    #compute_novelty(beta = 2.0)
    plot_diversity_novelty()