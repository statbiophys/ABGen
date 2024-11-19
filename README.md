# Generative model for monoclonal antibodies

This repo contains code for the paper [Generative model for monoclonal antibodies].
Some of the code was adapted from the repository [GflowNet for Biological Sequence Design] (https://github.com/MJ10/BioSeq-GFN-AL)
See Licence for more information

## Setup
The code has been tested with Python 3.7 with CUDA 10.2 and CUDNN 8.0.

1. We recommand setting up an anaconda environment before running the code
2. Before installing the requirements, ensure you have a c++ compiler available on your machine (apt-get install build-essential on ubuntu)
3. Install the dependencies (pip install -r requirements.txt)
4. Install anarci (conda install bioconda::anarci)

## Running the code
`mcmc_covid.py`, `mcmc_true_aff.py`, and `mcmc_true_aff_hard.py` are the entry points for the generation of sequences using MCMC.
`run_covid.py`, `run_true_aff.py`, and `run_true_aff_hard.py` are the entry points for the generation of sequences using GFlowNet.
`antBO_simple.py`, `antBO_hard.py` are the entry points for the generation of sequences using antBO.

Please reach out to Paul Pereira, [paul.pereira@phys.ens.fr](paul.pereira@phys.ens.fr) for any issues, comments, questions or suggestions.
