import torch
import torch.nn.functional as F
import numpy as np
from download_mnist import load
import operator
import math
import time

# Load MNIST Data
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28*28)
x_test  = x_test.reshape(10000,28*28)
x_train = (x_train/255.0).astype(float)
x_test = (x_test/255.0).astype(float)

x_test = torch.from_numpy(x_test)

# Get one hot label 
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
y_train = one_hot(y_train , 10)
y_test = one_hot(y_test , 10)
y_test = torch.from_numpy(y_test)

# Define Fully-connected layer
def fc_forward(z, W, b):
    return torch.matmul(z,W) + b

# Define ReLU layer as activation 
def relu_forward(z):
    return z.clamp(min=0)


# Define cross_entropy as function 
def cross_entropy_loss(y_predict, y_true):
    y_true = torch.argmax(y_true, dim=-1)
    return F.cross_entropy(y_predict, y_true)


# Define mean_square_error loss as function
def MSE_loss(y_predict, y_true):
    #print ((y_predict - y_true).data)
    loss = (y_predict - y_true).pow(2).sum(dim=-1)
    return loss.mean()

# Initialize Weights and bias same as in pytorch 
# model size 784-100-10
weights = {}
input_units = 784
std1 = 1. / 100.
std2 = 1. / 10.
weights["W1"] = torch.autograd.Variable(torch.randn(784,100, requires_grad=True, dtype=torch.double)*std1,requires_grad=True)
weights["b1"] = torch.autograd.Variable(torch.randn(100, requires_grad=True, dtype=torch.double)*std1,requires_grad=True)
weights["W2"]= torch.autograd.Variable(torch.randn(100,10, requires_grad=True, dtype=torch.double)*std2,requires_grad=True)
weights["b2"] = torch.autograd.Variable(torch.randn(10, requires_grad=True, dtype=torch.double)*std2,requires_grad=True)


# Define forward and backward computation 
gradients={}
def forward(x):
    fc1 = fc_forward(x, weights["W1"], weights["b1"]) 
    relu1 = relu_forward(fc1)
    fc2 = fc_forward(relu1, weights["W2"], weights["b2"]) 
    return fc2

def net(x, y):
    fc1 = fc_forward(x, weights["W1"], weights["b1"]) 
    relu1 = relu_forward(fc1)
    fc2 = fc_forward(relu1, weights["W2"], weights["b2"]) 
    #loss = MSE_loss(fc2, y)  # use mean_square_eroor loss
    loss = cross_entropy_loss(fc2,y)  #use cross_entropy loss
    loss.backward()
    gradients["W2"],gradients["b2"]= weights["W2"].grad, weights["b2"].grad 
    gradients["W1"],gradients["b1"]= weights["W1"].grad, weights["b1"].grad 
    return loss

# Define SGD as optimizer

class SGD(object):
    def __init__(self, weights, lr=0.01):
        self.v = weights
        self.iterations = 0 
        self.lr = lr

    def iterate(self, weights, gradients):
        for key in self.v.keys():
            with torch.no_grad():
                weights[key].data = weights[key] - self.lr * gradients[key] 
                weights[key].grad.data.zero_()
        self.iterations += 1


# Get prediction accuracy
def get_accuracy(X,y_true):
    y_predict=forward(X).argmax(dim=1, keepdim=True)
    acc = y_predict.eq(y_true.argmax(dim=1, keepdim=True)).sum()
    acc = acc.numpy()
    return acc/10000.0

# Random get next batch
train_num = len(x_train)
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return x_train[idx],y_train[idx]


# Set up hyperparameters
batch_size = 128
num_epoch = 10
sgd=SGD(weights,lr=0.01)

# Start Training
time0= time.time()
for e in range(num_epoch):
    for s in range(int(train_num/batch_size+1)):
        X,y=next_batch(batch_size)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        loss = net(X,y)
        sgd.iterate(weights,gradients)
    
    print("\n epoch:{} ; loss:{}".format(e+1,loss))
    #print (get_accuracy(x_test,y_test).item()) 
    print(" test_acc:{}".format(get_accuracy(x_test,y_test)))

time1=time.time()
#print("\n Final result test_acc:{}; ".format(get_accuracy(x_test,y_test)))
print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))