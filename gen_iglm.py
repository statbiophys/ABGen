from iglm import IgLM
import pandas as pd
from anarci import anarci
from Bio.Seq import Seq

def generate_iglm():
    iglm = IgLM()

    prompt_sequence = "EVQ"
    chain_token = "[HEAVY]"
    species_token = "[HUMAN]"
    num_seqs = 300

    generated_seqs = iglm.generate(
        chain_token,
        species_token,
        prompt_sequence=prompt_sequence,
        num_to_generate=num_seqs,
    )

    with open('./lib/dataset/full_IGLM.txt','w') as f:
        for i in generated_seqs:
            f.write(i+'\n')

def find_CDR3(seqs):
    sequences = []
    CDRs = []
    for i in range(len(seqs)):
        sequences.append(('seq'+str(i),seqs[i]))
    results = anarci(sequences, scheme="imgt", output=False)
    numbering, alignment_details, hit_tables = results

    for j in range(len(numbering)):
        if numbering[j] is None:
            CDRs.append('NO CDR')
        else:
            cdr = ''.join([i[1] for i in numbering[j][0][0] if int(i[0][0]) > 103 and int(i[0][0]) < 119 and i[1] != '-'])
        CDRs.append(cdr)
    with open('./lib/dataset/IGLM_seqs_CDR3.txt','w') as f:
        for i in CDRs:
            f.write(i+'\n')

def get_IGG_seqs():
    data = pd.read_csv('./lib/dataset/IGG.csv').values
    seqs = []
    for i in data:
        seqs.append(i[13])
    return seqs

def get_IGG_CDRs():
    data = pd.read_csv('./lib/dataset/IGG.csv').values
    seqs = []
    for i in data:
        seqs.append(i[49])
    return seqs
        

def save_thera_seqs():
    data = pd.read_csv('./lib/dataset/thera_antibodies.csv').values
    seqs = []
    clinical = []
    for i in data:
        if i[4] == 'Phase-III' or i[4] == 'Approved' or i[4] == 'Approved (w)':
            seqs.append(i[6])
            clinical.append(i[4])
    with open('./lib/dataset/thera_seqs.txt','w') as f:
        for i in seqs:
            f.write(i+'\n')

def get_thera_CDR():
    seqs = []
    with open('./lib/dataset/thera_seqs_CDR3.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs

def get_IGLM_seqs_2():
    seqs = []
    with open('./lib/dataset/full_IGLM.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs

def get_IGLM_CDR():
    seqs = []
    with open('./lib/dataset/IGLM_seqs_CDR3.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs

def get_thera_seqs():
    seqs = []
    with open('./lib/dataset/thera_seqs.txt','r') as f:
        for line in f:
            seqs.append(line.split('\n')[0])
    return seqs


if __name__  == '__main__':
    #save_thera_seqs()
    print(get_IGG_CDRs())
    #seqs = get_IGLM_seqs()
    #find_CDR3(seqs)
    #cdr = get_thera_CDR()
    #cdr_IGLM = get_IGLM_CDR()
    #print(cdr)