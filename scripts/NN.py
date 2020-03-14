import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold

#used to pass inputs through
def sigmoid(vec):
    return 1/(1+np.exp(-vec))

#given a threshold and a vector of values, convert binary
def binarize(vec, split=0.5, upper=1, lower=0):
    return np.where(vec > split, upper, lower)

#this assumes that values are already coming from sigmoid function
#return the derivative of those values
def sigmoid_derv(vec):
    return vec * (1 - vec)

class NeuralNetwork:
    def __init__(self, x, y, hlayer_size = 3, lr = 0.01, seed=123):
        self.lr = lr
        self.seed = seed
        self.x = x
        self.y = y

        np.random.seed(seed)
        #create layers and bias units
        self.w1 = np.random.randn(x.shape[1], hlayer_size)
        self.w2 = np.random.randn(hlayer_size, y.shape[1])
        # self.b2 = np.full(y.shape[0], bias)

    #this simply takes all our weights (w1,w2), multiplies by our features (x)
    #and stores these as layer outputs (z1, z2)
    def feedforward(self):
        self.z1 = sigmoid(np.dot(self.x, self.w1))
        self.z2 = sigmoid(np.dot(self.z1, self.w2))

    def backprop(self):

        #error was calculated as the difference between predicted (z2) and actual (y)
        error = self.z2 - self.y
        #we first get the error associated with our last layer and multiply by the derivate of output
        w2_delta = error * sigmoid_derv(self.z2)

        #calculate error of first later using the error calculated on the last layer
        error_h1 = np.dot(w2_delta, self.w2.T)
        w1_delta = error_h1 * sigmoid_derv(self.z1)

        #update weight matrices using our learning rate as well as the changes (w1/w2 delta) calculated above
        self.w1 -= self.lr * np.dot(self.x.T, w1_delta)
        self.w2 -= self.lr * np.dot(self.z1.T, w2_delta)

    #training is done with rounds of feedforward and then backprop
    #this continues until we reach a specified number of epochs
    def train(self, num_epochs=2000, mse_cutoff=0.001):
        self.epoch = []
        self.mse = []

        for x in range(num_epochs):
            self.feedforward()
            self.backprop()
            if x % 50 == 0:
                self.epoch.append(x)
                self.mse.append(np.square(self.y - self.z2).sum())
                # if len(self.mse) > 1:
                #     print('Here')
                    # if (self.mse[-2] - self.mse[-1]) < mse_cutoff:
                    #     print('finishing after epoch: ', x)
                    #     return



    #get true positive rate and false positive rate
        #assumes that z2 is a 2d matrix (which is should be) even when output is a single node
        #assumes that y is a 2d matrix (which is should be)
        #also assumes that the possible values of y are 1 and 0
    def roc(self, plot=True, predicted=None,actual=None):

        if actual is None:
            actual = self.y
        if predicted is None:
            predicted = self.z2

        out, true = predicted.flatten(), actual.flatten() # flatten 2d arrays (needed for cases where model output is not a single value)
        thresh = np.arange(0,1.002, 0.002) #thresholding is overdone here but ensures we have a nice plot

        if len(out) != len(true):
            raise ValueError('Flattened Model out does not match flattened input')

        tpr,fpr = [], []

        for t in thresh:
            pred = np.where(out > t, 1, 0)
            tp_count = 0
            fp_count = 0
            for x in range(len(pred)):
                if pred[x] == 1: #when we predict 1
                    if true[x] == 1:
                        tp_count += 1 # predicted correctly, true positive
                    elif true[x] == 0:
                        fp_count += 1 #predicted false as true: false positive

            pos = np.count_nonzero(true)
            neg = len(pred) - pos

            #divide true positive and false positive by number of pos and neg respectively to get rates
            tpr.append(tp_count/pos)
            fpr.append(fp_count/neg)

        if plot:
            plt.plot(fpr, tpr, '-')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.xlim(-0.05,1.05)
            plt.ylim(-0.05,1.05)
            plt.plot([0,1], [0,1], '--', c='red')
            plt.show()

        return metrics.auc(fpr, tpr)


    def plot_mse(self):
        plt.plot(self.epoch, self.mse)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()

    def kfold(self, num_folds=10):
        kf = KFold(n_splits=num_folds, shuffle=True) #init kfold calculator
        master_x = self.x
        master_y = self.y

        pred_y_final = np.zeros((len(self.y),)) #placeholder for final predictions

        for train_index, test_index in kf.split(master_x): #get indices for test and train
            self.x, X_test = master_x[train_index], master_x[test_index] #create test/train sets
            self.y, y_test = master_y[train_index], master_y[test_index] #create train/test outpute
            self.train()

            new_pred = self.predict(X_test)
            pred_y_final[test_index] = new_pred.flatten() # store predictions

        self.x = master_x #reset x and y to original input and output
        self.y = master_y

        #return AUC as well as predicted values
        return self.roc(predicted = pred_y_final, plot=False), pred_y_final

#expects a 2d array of new data
#uses feedforward to predict
#assumes NN has already been trained
    def predict(self, new_dat):
        prev_x = self.x
        self.x = new_dat
        self.feedforward()
        self.x = prev_x
        return self.z2
