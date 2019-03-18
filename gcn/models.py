from gcn.layers import *
from gcn.metrics import *


flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.sum_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()."""
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

        self.sum_op = tf.summary.merge_all()

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError('Tensorflow session not provided.')
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, f'tmp/{self.name}.ckpt')
        print(f"Model saved in file {save_path}")

    def load(self, sess=None):
        if not sess:
            raise AttributeError('Tensorflow session not provided')
        saver = tf.train.Saver(self.vars)
        save_path = f'tmp/{self.name}.ckpt'
        saver.restore(sess, save_path)
        print(f'Model restored from file: {save_path}')


class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """
    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)

    def _build(self):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()"""
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

        self.sum_op = tf.summary.merge_all()

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        tf.summary.scalar(self.name + '/loss', self.loss)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        tf.summary.scalar(self.name + '/accuracy', self.accuracy)

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        tf.summary.scalar(self.name + '/loss', self.loss)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        tf.summary.scalar(self.name + '/accuracy', self.accuracy)

    def _build(self):

        self.layers.append(GraphConv(input_dim=self.input_dim,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     dropout=True,
                                     sparse_inputs=True,
                                     logging=self.logging))

        self.layers.append(GraphConv(input_dim=FLAGS.hidden1,
                                     output_dim=self.output_dim,
                                     placeholders=self.placeholders,
                                     act=lambda x: x,
                                     dropout=True,
                                     logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_AE(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_AE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.placeholders = placeholders

        self.adj_labels = tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_labels'],
                                                               validate_indices=False), [-1])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        n_nodes = tf.cast(tf.shape(self.placeholders['adj_labels'])[0], tf.float32)
        sum_adj = tf.cast(self.placeholders['sum_adj'], tf.float32)
        pos_weight = (n_nodes * n_nodes - sum_adj) / sum_adj
        norm = n_nodes * n_nodes / ((n_nodes * n_nodes - sum_adj) * 2)

        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.outputs,
                                                                                    targets=self.adj_labels,
                                                                                    pos_weight=pos_weight))

        self.z_mean = self.activations[-2]

        tf.summary.scalar(self.name + '/loss', self.loss)

    def _accuracy(self):
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(self.outputs), 0.5), tf.int32),
                                           tf.cast(self.adj_labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        tf.summary.scalar(self.name + '/accuracy', self.accuracy)

    def _build(self):

        self.layers.append(GraphConv(input_dim=self.input_dim,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     dropout=True,
                                     sparse_inputs=True,
                                     logging=self.logging))

        self.layers.append(GraphConv(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden2,
                                     placeholders=self.placeholders,
                                     act=lambda x: x,
                                     dropout=True,
                                     logging=self.logging))

        self.layers.append(InnerProductDecoder(placeholders=self.placeholders,
                                               act=lambda x: x,
                                               dropout=False))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_VAE(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_VAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.placeholders = placeholders

        self.adj_labels = tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_labels'],
                                                               validate_indices=False), [-1])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        n_nodes = tf.cast(tf.shape(self.placeholders['adj_labels'])[0], tf.float32)
        sum_adj = tf.cast(self.placeholders['sum_adj'], tf.float32)
        pos_weight = (n_nodes * n_nodes - sum_adj) / sum_adj
        norm = n_nodes * n_nodes / ((n_nodes * n_nodes - sum_adj) * 2)

        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.outputs,
                                                                                    targets=self.adj_labels,
                                                                                    pos_weight=pos_weight))
        self.z_mean, self.z_log_std = self.activations[-3]
        self.kl = (0.5 / n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * self.z_log_std - tf.square(self.z_mean) -
                                                                 tf.square(tf.exp(self.z_log_std)), 1))
        self.loss -= self.kl

        tf.summary.scalar(self.name + '/loss', self.loss)

    def _accuracy(self):
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(self.outputs), 0.5), tf.int32),
                                           tf.cast(self.adj_labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        tf.summary.scalar(self.name + '/accuracy', self.accuracy)

    def _build(self):

        self.layers.append(GraphConv(input_dim=self.input_dim,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     dropout=True,
                                     sparse_inputs=True,
                                     logging=self.logging))

        self.layers.append(ParallelGraphConv(num=2,
                                             input_dim=FLAGS.hidden1,
                                             output_dim=FLAGS.hidden2,
                                             placeholders=self.placeholders,
                                             act=lambda x: x,
                                             dropout=True,
                                             logging=self.logging))

        self.layers.append(Normal())

        self.layers.append(InnerProductDecoder(placeholders=self.placeholders,
                                               act=lambda x: x,
                                               dropout=False))

    def predict(self):
        return tf.nn.softmax(self.outputs)
