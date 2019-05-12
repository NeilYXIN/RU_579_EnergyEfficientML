import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

csv_data = pd.read_csv('../../data/TF_FFT_DATA.csv', header=None)
print(csv_data.shape)
label_num = 10
features=[]
labels=[]

def get_data(label):
    condition = csv_data.loc[:,csv_data.shape[1]-1] == label
    filter_result = csv_data[condition]
    for i in range(filter_result.shape[0]):
        y = filter_result.values[i]
        feature = y[0:csv_data.shape[1]-1]
        label = int(y[csv_data.shape[1]-1])
        global features
        features=np.append(features,feature)
        global labels
        labels=np.append(labels,int(label))


for i in range(label_num):
    get_data(i)

features=features.reshape(-1,csv_data.shape[1]-1)
labels=np.array(labels, dtype=np.int32)
print(features)
print(labels)

def one_hot_encode(labels):
    n_labels = len(labels) # number of datasets
    one_hot_encode = np.zeros((n_labels,label_num)) # initialize label matrix
    one_hot_encode[np.arange(n_labels), labels] = 1 # generate one hot label matrix, row = numbers of samples, col = number of catagories
    return one_hot_encode

labels=one_hot_encode(labels)
print(len(labels),len(features))

train_test_split = np.random.rand(len(features)) < 0.70 # randomly divide training sets and test sets
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

# # Tensorflow Implementation
import tensorflow as tf

# ## Deep Neural Network

class_num = label_num
n_hidden_units_one = 120
n_hidden_units_two = 60
#30/20 for 9 points
#120/60 for 16

# n_hidden_units_three = 80
learning_rate = 0.01
n_dim = features.shape[1]
sd = 1/np.sqrt(n_dim)

# Input Layer
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, n_dim],name='x_input')
    y_true = tf.placeholder(tf.float32, [None,class_num],name='y_input')

# Hidden Layers
with tf.name_scope('hidden_layer_1'):
    with tf.name_scope('W1'):
        W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean=0, stddev=sd, name='Weights'))
    with tf.name_scope('b1'):
        b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd, name='biases'))
    with tf.name_scope('h1'):
        h_1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1, name='output')

with tf.name_scope('hidden_layer_2'):
    with tf.name_scope('W2'):
        W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean=0, stddev=sd, name='Weights'))
    with tf.name_scope('b2'):
        b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd, name='biases'))
    with tf.name_scope('h2'):
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2, name='output')
        
# Output Layer
with tf.name_scope('output_layer'):
    with tf.name_scope('W'):
        W = tf.Variable(tf.random_normal([n_hidden_units_two,class_num], mean=0, stddev=sd, name='Weights'))
    with tf.name_scope('b'):
        b = tf.Variable(tf.random_normal([class_num], mean=0, stddev=sd, name='biases'))
    with tf.name_scope('output'):
        y = tf.nn.softmax(tf.matmul(h_2, W) + b, name='output')

# Cost Function
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y), reduction_indices=[1]))
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    print(train_step)

from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
    
for step in range(200):
    batch_x, batch_y = train_x,train_y
    train_step.run({x:batch_x, y_true:batch_y})
    if step%10==0:
#         y_res = tf.identity(y,name='y_prediction') # Output
        pre_num=tf.argmax(y,1,output_type='int32',name='output')
        correct_prediction = tf.equal(pre_num,tf.argmax(y_true,1,output_type='int32'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        a = accuracy.eval({x:test_x,y_true:test_y})
        print('Accuracy：{0}'.format(a))
        
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['output_layer/output/output'])
with tf.gfile.FastGFile('../../data/VibAuthModel.pb', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())
sess.close()
