#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np  
import operator  
import time
import tensorflow as tf


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
x_train=x_train/255
x_test=x_test/255


# In[5]:


y_train_one_hot = one_hot_encoder(y_train)
y_test_one_hot = one_hot_encoder(y_test)


# In[6]:


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


# In[ ]:





# In[8]:


class_num = 10
n_hidden_units_one = 200
n_hidden_units_two = 50

learning_rate = 0.01
n_dim = x_train.shape[1]

# Input Layer
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, n_dim],name='x_input')
    y_true = tf.placeholder(tf.float32, [None,class_num],name='y_input')

# Hidden Layers
with tf.name_scope('hidden_layer_1'):
    with tf.name_scope('W1'):
        W_1 = tf.Variable(tf.truncated_normal([n_dim,n_hidden_units_one], stddev=0.1, name='Weights'))
    with tf.name_scope('b1'):
        b_1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_units_one], name='biases'))
    with tf.name_scope('h1'):
        h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1, name='output')

with tf.name_scope('hidden_layer_2'):
    with tf.name_scope('W2'):
        W_2 = tf.Variable(tf.truncated_normal([n_hidden_units_one,n_hidden_units_two], stddev=0.1, name='Weights'))
    with tf.name_scope('b2'):
        b_2 = tf.Variable(tf.constant(0.1, shape=[n_hidden_units_two], name='biases'))
    with tf.name_scope('h2'):
        h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2, name='output')
        
# Output Layer
with tf.name_scope('output_layer'):
    with tf.name_scope('W'):
        W = tf.Variable(tf.truncated_normal([n_hidden_units_two,class_num], stddev=0.1, name='Weights'))
    with tf.name_scope('b'):
        b = tf.Variable(tf.constant(0.1, shape=[class_num], name='biases'))
    with tf.name_scope('output'):
        y = tf.nn.softmax(tf.matmul(h_2, W) + b, name='output')

# Cost Function
with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_sum(y_true*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    print(train_step)


# In[9]:


from tensorflow.python.framework import graph_util
import datetime


# In[10]:


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
epoch_num=5
batch_size=128
starttime = datetime.datetime.now()

for i in range(epoch_num):   
    print("epoch", i)
    counter=np.zeros(x_train.shape[0])
    step=0
    while np.sum(counter) < len(x_train):
        batch_x, batch_y,counter = next_batch(x_train, y_train_one_hot, batch_size, counter)

        train_step.run({x:batch_x, y_true:batch_y})
        if step%300==0:
            pre_num=tf.argmax(y,1,output_type='int32',name='output')
            correct_prediction = tf.equal(pre_num,tf.argmax(y_true,1,output_type='int32'))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            a_t = accuracy.eval({x:batch_x,y_true:batch_y})
            print('Step', step)
            print('Train Accuracy:{0}'.format(a_t),'\n')
            
        step+=1

endtime = datetime.datetime.now()
print ("Training Time:", (endtime - starttime).seconds,'s')
a = accuracy.eval({x:x_test,y_true:y_test_one_hot})
print('Test Accuracy:{0}'.format(a),'\n')
sess.close()


# In[ ]:




