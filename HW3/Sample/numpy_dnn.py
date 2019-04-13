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

# Get one hot label 
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
y_train = one_hot(y_train , 10)
y_test = one_hot(y_test , 10)

# Define Fully-connected layer
def fc_forward(z, W, b):
    return np.dot(z, W) + b

def fc_backward(next_dz, W, z):
    N = z.shape[0]
    dz = np.dot(next_dz, W.T)  
    dw = np.dot(z.T, next_dz)  
    db = np.sum(next_dz, axis=0)  
    return dw / N, db / N, dz

# Define ReLU layer as activation 
def relu_forward(z):
    return np.maximum(0, z)

def relu_backward(next_dz, z):
    dz = np.where(np.greater(z, 0), next_dz, 0)
    return dz


# Define cross_entropy as function 
def cross_entropy_loss(y_predict, y_true):
    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1,keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  
    dy = y_probability - y_true
    return loss, dy


# Define mean_square_error loss as function
def MSE_loss(y_predict, y_true):
    loss = np.mean(np.square(y_predict - y_true))
    dy = (2.0 * (y_predict - y_true))
    return loss, dy

# Initialize Weights and bias same as in pytorch 
# model size 784-100-10
weights = {}
input_units = 784
std1 = 1. / math.sqrt(100)
std2 = 1. / math.sqrt(10)
weights["W1"] = np.random.uniform(-std1, std1,(input_units, 100)).astype(np.float64)
weights["b1"] = np.random.uniform(-std1, std1, 100).astype(np.float64)
weights["W2"] = np.random.uniform(-std2, std2,(100, 10)).astype(np.float64)
weights["b2"] = np.random.uniform(-std2, std2, 10).astype(np.float64) 


# Define forward and backward computation for dnn model
nuerons={}
gradients={}
def forward(X):
    nuerons["fc1"]=fc_forward(X.astype(np.float64),weights["W1"],weights["b1"])
    nuerons["fc1_relu"]=relu_forward(nuerons["fc1"])
    nuerons["y"]=fc_forward(nuerons["fc1_relu"],weights["W2"],weights["b2"])
    return nuerons["y"]

def backward(X,y_true):
    loss,dy=cross_entropy_loss(nuerons["y"],y_true) #use cross_entropy loss
    #loss, dy = MSE_loss(nuerons["y"],y_true) # use mean_square_eroor loss
    gradients["W2"],gradients["b2"],gradients["fc1_relu"]=fc_backward(dy,weights["W2"],nuerons["fc1_relu"])
    gradients["fc1"]=relu_backward(gradients["fc1_relu"],nuerons["fc1"])
    gradients["W1"],gradients["b1"],_=fc_backward(gradients["fc1"],weights["W1"],X)
    return loss

# Define SGD as optimizer
def _copy_weights_to_zeros(weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result
class SGD(object):
    def __init__(self, weights, lr=0.01):
        self.v = _copy_weights_to_zeros(weights)  
        self.iterations = 0 
        self.lr = lr

    def iterate(self, weights, gradients):
        for key in self.v.keys():
            self.v[key] = self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]
        self.iterations += 1


# Get prediction accuracy
def get_accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1),
                            np.argmax(y_true,axis=-1)))

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
        forward(X)
        loss=backward(X,y)
        sgd.iterate(weights,gradients)
    print("\n epoch:{} ; loss:{}".format(e+1,loss))
    print(" train_acc:{};  test_acc:{}".format(get_accuracy(X,y),get_accuracy(x_test,y_test)))

time1=time.time()
print("\n Final result test_acc:{}; ".format(get_accuracy(x_test,y_test)))
print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   