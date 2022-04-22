import pandas as pd
import numpy as np
from copy import deepcopy 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from random import shuffle
import time
from ast import literal_eval
import pickle as cPickle

import torch.nn.functional as F
import torch
####################################
# logistic dpp using to generate diverse item embedings based on item sets 
# change user_num, item_num, input files and output_files, according to your experiments
####################################

t0 = time.time()

np.random.seed(0)

####################################
# parameters 
####################################
user_num = 4641 
item_num = 2235
emb_dim = 64
set_length = 5
lr = 1e-3
decay_step = 100
decay = 0.95

sigmoid_lbda = 0.01
epochs = 50
runs = 1
batch_size = 1024
emb_init_mean = 0.
emb_init_std = 0.01
diag_init_mean = 1.
diag_init_std = 0.01
regu_weight = 0.

def get_sets(pos_set_file, neg_set_file):
    
    upos_sets = []
    with open(pos_set_file) as f:
        for l in f.readlines():
            sstr = l.strip().split(';')
            u, sets = int(sstr[0]), sstr[1:]

            for s in sets:
                a_set = []
                s1 = s.split(',')
                for id in s1:
                    a_set.append(int(id))
                if len(a_set) == set_length:
                    upos_sets.append(a_set)

    uneg_sets = []
    with open(neg_set_file) as f:
        for l in f.readlines():
            sstr = l.strip().split(';')
            u, sets = int(sstr[0]), sstr[1:]

            for s in sets:
                a_set = []
                s1 = s.split(',')
                for id in s1:
                    a_set.append(int(id))
                if len(a_set) == set_length:
                    uneg_sets.append(a_set)
    return np.array(upos_sets), np.array(uneg_sets)

################################
# create model 
################################
def set_det(item_sets):
    subV = tf.gather(weights['V'],item_sets)
    #subV = tf.Print(subV, ['subV=', subV, '  '])
    subD = tf.matrix_diag(tf.square(tf.gather(weights['D'],item_sets)))
    K1 = tf.matmul(subV, tf.transpose(subV,perm=[0,2,1]))
    K = tf.add(K1,subD)
    eps = tf.eye(tf.shape(K)[1],tf.shape(K)[1],[tf.shape(K)[0]])
    K = tf.add(K,eps)
    res = tf.matrix_determinant(K)
    #res = tf.Print(res, ['res=', res, '  '])
    return res

def logsigma(itemSet):
    return tf.reduce_mean(tf.log(1-tf.exp(-sigmoid_lbda*set_det(itemSet))))

def regularization(itemSet):
    itemsInBatch, _ = tf.unique(tf.reshape(itemSet,[-1]))
    subV = tf.gather(weights['V'],itemsInBatch)
    subD = tf.gather(weights['D'],itemsInBatch)
    subV_norm = tf.reduce_mean(tf.norm(subV,axis=1))
    subD_norm = tf.norm(subD)
    return subV_norm+subD_norm

################################
# tf graph
################################

pset_input = tf.placeholder(tf.int32, [None,None])   #item sets
nset_input = tf.placeholder(tf.int32, [None,None])   #item sets

#get processed sets
pos_sets, neg_sets = get_sets('pos_item_sets_3.txt', 'neg_item_sets_3.txt')
train_size = len(pos_sets)

print(pos_sets.shape, neg_sets.shape)
for run in range(runs):
    # Construct model
    pset_input = tf.placeholder(tf.int32, [None,None])   #item sets
    nset_input = tf.placeholder(tf.int32, [None,None])   #item sets

    # Store layers weight & bias
    initializer = tf.keras.initializers.glorot_normal()
    weights = {
        'V': tf.Variable(initializer([item_num, emb_dim]), name='item_embeddings'),
        'D': tf.Variable(initializer([item_num]), name='item_bias')
    }
    # Construct model
    loss = logsigma(pset_input) + tf.log(1 - logsigma(nset_input)) # - regu_weight*regularization(pset_input) + regu_weight*regularization(nset_input)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.01,beta2=0.01) # beta plus petit ?
    train_op = optimizer.minimize(-loss)

    # Initializing the variables
    init = tf.global_variables_initializer()

    print("start training...")
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        #print(test, 'for not train')
        for epoch in range(epochs):
            ave_cost = 0.
            nbatch = 0
            while True:
                if nbatch*batch_size <= train_size:
                    pos_batch = pos_sets[nbatch*batch_size: (nbatch+1)*batch_size]
                    neg_batch = neg_sets[nbatch*batch_size: (nbatch+1)*batch_size]
                else:
                    if train_size - (nbatch-1)*batch_size > 0:
                        pos_batch = pos_sets[(nbatch-1)*batch_size: train_size]
                        neg_batch = neg_sets[(nbatch-1)*batch_size: train_size]
                    break
                nbatch += 1

                _, c = sess.run([train_op, loss], feed_dict={pset_input: pos_batch, nset_input: neg_batch})
                #print(c)
                ave_cost += c / nbatch
            #print(epoch, ave_cost)

        param = sess.run(weights)
        cPickle.dump(param, open('item_kernel_3.pkl', 'wb'))
