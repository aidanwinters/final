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

base_encode =  {'A': np.array([0,0,0,0,1]),
               'T': np.array([0,0,0,1,0]),
               'C': np.array([0,0,1,0,0]),
               'G': np.array([0,1,0,0,0]),
               'N': np.array([1,0,0,0,0])}

base_decode = {4:'A', 3:'T', 2:'C', 1:'G',0:'N'}

#convert an array of sequence strings into one-hot encoding
def strList_oneHotMatr(seq_strs):
    return np.array([np.concatenate([base_encode[base] for base in seq]) for seq in seq_strs])

#takes in a oneHot string representing one base and converts into a base
def modelOut_toBase(modelOut):
    #set all values to zero expect the max
    return base_decode[np.argmax(modelOut)]

def translate(oneHot):
    # check that sequence length is divisible by 5 in our case
    if len(oneHot) % len(base_decode) == 0:
        return [modelOut_toBase(oneHot[x:x+5]) for x in range(0,len(oneHot),len(base_decode))]
    else:
        raise ValueError('One Hot Sequence provided has incorrect length.')

#get a dictionary of sequences
def get_neg_dict(neg_file ='data/yeast-upstream-1k-negative.fa'):

    neg_dict = {}

    for record in SeqIO.parse(neg_file, "fasta"):
        neg_dict[record.id] = record.seq

    return neg_dict
#if a dictionary is passed, just parse and return sequences
#else, generate dictionary
def get_neg_seqs(neg_dict, num = 137,
                 pos_list = read_short_seqs('data/rap1-lieb-positives.txt'), oneHot = True, seed=0):

    np.random.seed(seed)
    seq_size = len(pos_list[0])
    seq_sample = np.random.choice(list(neg_dict.keys()), size=num)
    seqs = []

    for s in seq_sample:
        seq = neg_dict[s]
        is_pos = True

        while is_pos:
            select = np.random.choice(len(seq) - seq_size)
            new_seq = seq[select:select+seq_size]
            is_pos =  new_seq in pos_list

        seqs.append(new_seq)
    if oneHot:
        return strList_oneHotMatr(seqs), seqs
    else:
        return seqs
