import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN_AE, GCN_VAE

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from scipy.sparse import issparse

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_string('log_dir', 'logs', 'Log directory string')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

# Load data
adj, adj_train, features, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_citation_data(
    FLAGS.dataset,
    task='link_prediction')

# Some preprocessing
if FLAGS.features == 0:
    features = sp.identity(features.shape[0])
features = preprocess_features(features)

sum_adj = adj_train.sum()

adj_labels = adj_train + sp.eye(adj_train.shape[0])
adj_labels = sparse_to_tuple(adj_labels)

if FLAGS.model == 'gcn_ae':
    support = [preprocess_adj(adj_train)]
    num_supports = 1
    model_func = GCN_AE
elif FLAGS.model == 'gcn_1st_ae':
    identity = sparse_to_tuple(sp.eye(adj_train.shape[0]))
    adj_normalized = sparse_to_tuple(normalize_adj(adj_train))
    support = [identity, adj_normalized]
    num_supports = 2
    model_func = GCN_AE
elif FLAGS.model == 'gcn_vae':
    support = [preprocess_adj(adj_train)]
    num_supports = 1
    model_func = GCN_VAE
elif FLAGS.model == 'gcn_1st_vae':
    identity = sparse_to_tuple(sp.eye(adj_train.shape[0]))
    adj_normalized = sparse_to_tuple(normalize_adj(adj_train))
    support = [identity, adj_normalized]
    num_supports = 2
    model_func = GCN_VAE
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'adj_labels': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
    'sum_adj': tf.placeholder(tf.int32)
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True, name=FLAGS.model)

# Initialize session
sess = tf.Session()


# To write logs to Tensorboard
class Summarizer():
    def __init__(self, name):
        self.name = name
        self.summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/' + model.name + '/' + self.name,
                                                    graph=tf.get_default_graph())

    def __call__(self, epoch, feed_dict):
        summary = sess.run(model.sum_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, epoch)


summarize = Summarizer('val')


# Init variables
sess.run(tf.global_variables_initializer())


def get_roc_score(edges_pos, edges_neg, adj, emb):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if issparse(adj):
        adj = adj.toarray()

    # predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds_pos = []
    pos = []
    for e in edges_pos:
        preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    auc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return auc_score, ap_score


# Train model
for epoch in range(FLAGS.epochs):

    print(f'Epoch {epoch}:')

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict_link_prediction(features, support, adj_labels, sum_adj, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    loss = outs[1]
    acc = outs[2]

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    auc, ap = get_roc_score(val_edges, val_edges_false, adj, emb)
    print(f'\ttrain_loss={loss:.5f}\ttrain_acc={acc:.5f}\n\tval_auc_score={auc:.5f}\tval_ap_score={ap:.5f}'
          f'\n\ttime={time.time() - t:.5f}')

    summarize(epoch, feed_dict)

