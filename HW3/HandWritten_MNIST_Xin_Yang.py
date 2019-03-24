#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import math
import numpy as np  
import operator  
import datetime
import time


# In[2]:


import pickle
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


# In[3]:


def one_hot_encoder(labels, class_nums=10):
    one_hot=np.zeros([len(labels), class_nums])
    for i in range(len(labels)):
        one_hot[i][labels[i]] = 1
    return one_hot


# In[4]:


x_train, y_train, x_test, y_test = load()
x_train=x_train
x_test=x_test


# In[5]:


y_train_one_hot = one_hot_encoder(y_train)
y_test_one_hot = one_hot_encoder(y_test)


# In[6]:


def relu(x):
    return np.maximum(x,0.0)
 
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
 
def d_relu(x):
    (len_x,len_y)=x.shape
    d_x = np.zeros([len_x,len_y])
    [inds_x,inds_y]=np.where(x>=0)
    inds = np.stack((inds_x,inds_y), axis=-1) 
    for ind in inds:
        d_x[ind[0],ind[1]] = 1
    return d_x


# In[7]:



def next_batch(train_x, train_y, batch_size, counter):
    batch_x = []
    batch_y = []
    rand_indices = []
    if len(train_x)-np.sum(counter) <= batch_size:
        for i in range(len(counter)):
            if counter[i]==0:
                rand_indices.append(i)
                counter[i]=1
    else:      
        while len(rand_indices) != batch_size:
            tmp=np.random.randint(0,len(train_x) - 1)
            if counter[tmp]==1:
                continue
            else:
                rand_indices.append(tmp)
                counter[tmp]=1
#     print("rand indices" , rand_indices)
    for index in rand_indices:
        batch_x.append(train_x[index])
        batch_y.append(train_y[index])
    return np.array(batch_x), np.array(batch_y),counter


# In[8]:


def initialize_variables(size):
    w=np.random.randn(size[0], size[1]) * 0.01
#     w=np.zeros([size[0], size[1]])
#     b=np.random.random([size[1]])
    b=np.zeros(size[1])
    return w,b


# In[9]:





# In[10]:


learning_rate=0.01

def forward_propagation(w_1,b_1,w_2,b_2,w,b,X,y):
    Z1=np.dot(X,w_1) + b_1
    A1=relu(Z1)
    Z2=np.dot(A1,w_2) + b_2
    A2=relu(Z2)
    Z3=np.dot(A2,w) + b
    y_=softmax(Z3)
    
    loss = np.square(y-y_).sum()/len(y)
    
    cache={'Z1':Z1, 'A1':A1, 'Z2':Z2, 'A2':A2, 'Z3':Z3,'y_':y_}
    return cache, loss


def back_propagation(forward_cache,w_1,w_2,w,b_1,b_2,b,x,y):
    dZ3 = forward_cache['y_']-y
    wT = np.transpose(w)
    dZ2 = np.dot(dZ3,wT)*d_relu(forward_cache['Z2'])
    
    w_2T=np.transpose(w_2)
    dZ1 = np.dot(dZ2,w_2T)*d_relu(forward_cache['Z1'])
    
    
    m = forward_cache['A2'].shape[1]
    A2T = np.transpose(forward_cache['A2'])
    dw = np.dot(A2T,dZ3)/m
    db = np.sum(dZ3, axis=1, keepdims= True)/m
    
    m = forward_cache['A1'].shape[1]
    A1T=np.transpose(forward_cache['A1'])
    dw_2=np.dot(A1T,dZ2)/m
    db_2 = np.sum(dZ3,axis=1, keepdims= True)/m
    
    
    xT = np.transpose(x)
    dw_1 = np.dot(xT,dZ1)/m
    db_1 = np.sum(dZ2, axis=1, keepdims= True)/m
    
    
    w_1=w_1-learning_rate*dw_1
    b_1=b_1-learning_rate*db_1
    
    w_2=w_2-learning_rate*dw_2
    b_2=b_2-learning_rate*db_2
    
    w=w-learning_rate*dw
    b=b-learning_rate*b
    
    return w_1,b_1,w_2,b_2,w,b


# In[11]:


print(y_train_one_hot.shape)


# In[12]:


num_hidden_untis_one=200
num_hidden_units_two=50
n_dim = x_train.shape[1]
# sd = 1/np.sqrt(n_dim)

# index=0
# X=np.array(features[index])
# y=np.array(train_labels[index])
# print(X)
# print(y)


w_1,b_1=initialize_variables([x_train.shape[1], num_hidden_untis_one])

w_2,b_2=initialize_variables([num_hidden_untis_one, num_hidden_units_two])

w,b=initialize_variables([num_hidden_units_two, y_train_one_hot.shape[1]])

# learning rate, max iteration
num_epochs = 1
batch_size = 1
starttime = datetime.datetime.now()

for i in range(num_epochs): 
    print("epoch", i)
    counter=np.zeros(x_train.shape[0])
    step=0
    while np.sum(counter) < len(x_train):
        input_x,y,counter = next_batch(x_train, y_train_one_hot, batch_size, counter)
    #     input_x=x_train
    #     y=y_train_one_hot
        forward_cache,loss = forward_propagation(w_1,b_1,w_2,b_2,w,b,input_x, y)

        if step % 100 == 0:
            print('step:',step)
    #         print(forward_cache['y_'])
    #         print(y)
            argmax_y=np.argmax(y,axis=1)
            argmax_y_=np.argmax(forward_cache['y_'],axis=1)
            equals=[]
            for i in range(len(argmax_y)):
                if argmax_y[i]==argmax_y_[i]:
                    equals.append(1)
                else:
                    equals.append(0)
            accuracy=np.sum(equals)/len(equals)
            print ("The loss is %f" % (np.mean(np.square(loss))))
            print ("The accuracy is %f\n" % accuracy)

            #print output_error.tolist()
    #         continue
        w_1,b_1,w_2,b_2,w,b=back_propagation(forward_cache,w_1,w_2,w,b_1,b_2,b,input_x,y)
        step+=1
endtime = datetime.datetime.now()
print ("Training Time:", (endtime - starttime).seconds,'s')


# In[13]:



equals=[]

for i in range(x_test.shape[0]):
    forward_cache,loss = forward_propagation(w_1,b_1,w_2,b_2,w,b,x_test[i], y_test_one_hot[i])
    
#     print(forward_cache['y_'])
#     print(loss)
    argmax_y=np.argmax(y_test_one_hot[i])
    argmax_y_=np.argmax(forward_cache['y_'])
    if argmax_y==argmax_y_:
        equals.append(1)
    else:
        equals.append(0)
accuracy=np.sum(equals)/len(equals)
print ("The accuracy is %f\n" % accuracy)


# In[ ]:




