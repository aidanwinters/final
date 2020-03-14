import numpy as np
from Bio import SeqIO


#read in sequence file
#assumes each new line is a target sequence
def read_short_seqs(file):

    seqs = []
    with open(file) as f:
        for line in f:
            seqs.append(line.strip())
    return seqs

#dictionary to go from base to oneHot encoding
base_encode =  {'A': np.array([0,0,0,0,1]),
               'T': np.array([0,0,0,1,0]),
               'C': np.array([0,0,1,0,0]),
               'G': np.array([0,1,0,0,0]),
               'N': np.array([1,0,0,0,0])}

#dictionary to go from oneHot to Base,
#Only stores the position of 1 in the one hot encoding
base_decode = {4:'A', 3:'T', 2:'C', 1:'G',0:'N'}

#convert an array of sequence strings into one-hot encoding
def strList_oneHotMatr(seq_strs):
    return np.array([np.concatenate([base_encode[base] for base in seq]) for seq in seq_strs])


#get a dictionary of sequences using biopython
def get_neg_dict(neg_file ='data/yeast-upstream-1k-negative.fa'):

    neg_dict = {}

    for record in SeqIO.parse(neg_file, "fasta"):
        neg_dict[record.id] = record.seq

    return neg_dict

# using the dictionary of negative Sequences
#get the desired number of sequences using numpy sampler (random.choice)
def get_neg_seqs(neg_dict, num = 137,
                 pos_list = read_short_seqs('data/rap1-lieb-positives.txt'), oneHot = True, seed=0):

    np.random.seed(seed)
    seq_size = len(pos_list[0])
    seq_sample = np.random.choice(list(neg_dict.keys()), size=num)
    seqs = []

    for s in seq_sample:
        seq = neg_dict[s]
        is_pos = True

        #make sure that the selection is not found in the positive sequence list
        while is_pos:
            select = np.random.choice(len(seq) - seq_size)
            new_seq = seq[select:select+seq_size]
            is_pos =  new_seq in pos_list

        seqs.append(new_seq)
    if oneHot:
        return strList_oneHotMatr(seqs), seqs
    else:
        return seqs
