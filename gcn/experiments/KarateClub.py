import networkx as nx
from networkx.algorithms import community
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

zkc = nx.karate_club_graph()

# create a new class
c = community.greedy_modularity_communities(zkc)
c = [sorted(x) for x in c]
l = {}
for i in range(len(c)):
    for j in range(len(c[i])):
        l[c[i][j]] = {'community': i}
nx.set_node_attributes(zkc, l)

s = {}
for i in range(len(c)):
    sample = np.random.choice(c[i], 2)
    for j in range(len(sample)):
        s[sample[j]] = {'sample': i}
nx.set_node_attributes(zkc, s)

data = pd.DataFrame.from_dict(dict(zkc.nodes(data=True)), orient='index')

# # visualization
# pos = nx.spring_layout(zkc)
# labels = {l: l for l in zkc.nodes().keys()}
# nx.draw_networkx_nodes(zkc, pos, nodelist=c[0], node_color='r', node_size=200)
# nx.draw_networkx_nodes(zkc, pos, nodelist=c[1], node_color='b', node_size=200)
# nx.draw_networkx_nodes(zkc, pos, nodelist=c[2], node_color='g', node_size=200)
# nx.draw_networkx_edges(zkc, pos, width=1)
# nx.draw_networkx_labels(zkc, pos, labels=labels, font_color='w', font_size=8)
# plt.axis('off')
# plt.show()


# input for neural network

order = sorted(list(zkc.nodes()))
adjacency = nx.to_numpy_array(zkc, nodelist=order)
identity = np.eye(zkc.number_of_nodes())
adjacency = adjacency + identity
degree = np.diag(np.sum(adjacency, axis=0))
labels = np.array(pd.factorize(data.club)[0])
labels_onehot = np.zeros((labels.shape[0], 2))
labels_onehot[np.arange(labels.shape[0]), labels] = 1

labels_mask = np.zeros((labels.shape[0]))
fn = lambda obj: np.random.choice(obj.index, 1, False)
sample = np.concatenate(data.groupby('club', as_index=False).apply(fn).values)
labels_mask[sample] = 1

n_features = identity.shape[1]
n_nodes = adjacency.shape[0]
n_filters_1 = 10
n_filters_2 = 6
n_filters_3 = 2

# placeholders
Features = tf.placeholder(tf.float32, shape=(n_nodes, n_features))
Adjacency = tf.placeholder(tf.float32, shape=(n_nodes, n_nodes))
Degree = tf.placeholder(tf.float32, shape=(n_nodes, n_nodes))
Labels = tf.placeholder(tf.float32, shape=(n_nodes, 2))
LabelsMask = tf.placeholder(tf.int32, shape=n_nodes)


def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial)


Weights_1 = glorot([n_features, n_filters_1])
Weights_2 = glorot([n_filters_1, n_filters_2])
Weights_3 = glorot([n_filters_2, n_filters_3])


def gcn_layer(features, adjacency, degree, weights, activation='relu'):
    d_ = tf.pow(tf.linalg.inv(degree), 0.5)
    y = tf.matmul(d_, tf.matmul(adjacency, d_))
    kernel = tf.matmul(features, weights)
    if activation == 'relu':
        out = tf.nn.relu(tf.matmul(y, kernel))
    elif activation == 'sigmoid':
        out = tf.nn.sigmoid(tf.matmul(y, kernel))
    else:
        out = tf.matmul(y, kernel)
    return out


H_1 = gcn_layer(Features, Adjacency, Degree, Weights_1, activation='relu')
H_2 = gcn_layer(H_1, Adjacency, Degree, Weights_2, activation='relu')
H_3 = gcn_layer(H_2, Adjacency, Degree, Weights_3, activation='id')
Output = tf.nn.softmax(H_3)

L = tf.nn.softmax_cross_entropy_with_logits(logits=H_3, labels=Labels)
LabelsMask = tf.cast(LabelsMask, tf.float32)
LabelsMask /= tf.reduce_mean(LabelsMask)
L *= LabelsMask
Loss = tf.reduce_mean(L)
TrainOp = tf.train.AdamOptimizer(0.05, 0.9).minimize(Loss)

n_iter = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(n_iter):
        feed_dict = {Features: identity, Adjacency: adjacency, Degree: degree, Labels: labels_onehot, LabelsMask: labels_mask}
        _, loss = sess.run([TrainOp, Loss], feed_dict=feed_dict)
        print(loss)
        res = sess.run(Output, feed_dict=feed_dict)

        if it % 1 == 0:
            pos = {}
            for i in range(len(res)):
                pos[i] = np.array(res[i])
            ids = {l: l for l in zkc.nodes().keys()}
            cat = list(nx.get_node_attributes(zkc, 'club').values())
            dic = dict(zip(set(cat), list(range(1, len(set(cat))+1))))
            cols = [dic[v] for v in cat]
            nx.draw_networkx_nodes(zkc, pos, node_color=cols, node_size=200)
            nx.draw_networkx_edges(zkc, pos, width=1)
            nx.draw_networkx_labels(zkc, pos, labels=ids, font_color='w', font_size=8)
            plt.axis('off')
            plt.show()

