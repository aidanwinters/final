from .io import read_short_seqs, strList_oneHotMatr, get_neg_seqs, get_neg_dict, translate
from .NN import NeuralNetwork, binarize
import numpy as np
import matplotlib.pyplot as plt

pos_seqs = read_short_seqs('data/rap1-lieb-positives.txt')
pos_oneHot = strList_oneHotMatr(pos_seqs)
pos_lbl = np.ones((len(pos_seqs),1))

neg_dict = get_neg_dict()
neg_oneHot, neg_seqs = get_neg_seqs(neg_dict, pos_list = pos_seqs)
neg_lbl = np.zeros((len(neg_seqs),1))

print(pos_oneHot.shape, neg_oneHot.shape)
merged = np.concatenate((pos_oneHot, neg_oneHot))
print(merged.shape)
response = np.concatenate((pos_lbl, neg_lbl))

# Testing kfold individually
# rap1_nn = NeuralNetwork(merged, response, hlayer_size=len(response)//2, lr =0.1)
# cv_auc, cv_pred = rap1_nn.kfold(5)

#Using kfold to test
# some parameters/architecture for best performance
for hl in [len(response)//1 - 40, len(response)//2, len(response)//4]:
    for lr in [1, 0.1, 0.01, 0.001]:
        for seed in [1,2,3]:
            nn = NeuralNetwork(merged, response, hlayer_size=hl, lr=lr, seed=seed)
            cv_auc, cv_pred = nn.kfold(10)
            print('Layer size:', hl, '\tLR: ', lr, '\tSeed: ', seed, '\tAUC:', cv_auc)

# rap1_nn.feedforward()
# rap1_nn.train()
# auc = rap1_nn.roc()
# print(auc)
# print(rap1_nn.predict(merged))
# print(rap1_nn.auc())

# rap1_nn.plot_mse()


#TESTING 8x3x8 auto encoder

# identity_matrix = np.array([[1,0,0,0,0,0,0,0],
#                             [0,1,0,0,0,0,0,0],
#                             [0,0,1,0,0,0,0,0],
#                             [0,0,0,1,0,0,0,0],
#                             [0,0,0,0,1,0,0,0],
#                             [0,0,0,0,0,1,0,0],
#                             [0,0,0,0,0,0,1,0],
#                             [0,0,0,0,0,0,0,1]])
#
# # # input and output should both be 2d matrices!!!!!
# nn = NeuralNetwork(identity_matrix, identity_matrix, hlayer_size=3, lr =0.1)
# nn.feedforward()
# nn.backprop()
# nn.train()
# nn.roc()
# print(nn.auc())
