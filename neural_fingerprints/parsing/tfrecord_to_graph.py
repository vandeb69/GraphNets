import operator, copy
import numpy as np
import tensorflow as tf


# Read graphs from TFRecord file
def read_graphs_from_tfrecord(filename):
    # Create record iterator
    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    # Initialize list of read graphs
    graphs = []
    # For each record in the file, parse the corresponding molecular graph (represented as a Python dictionary)
    for i, string_record in enumerate(record_iterator):
        print(f"\r\t{i + 1}", end='', flush=True)
        g = string_record_to_graph(string_record)
        graphs.append(g)

    return graphs


# Write all graphs to a TFRecord file
def write_graphs_to_tfrecord(graphs, filename):
    # Create writer
    writer = tf.python_io.TFRecordWriter(filename)

    # Iterate across all graphs in the dataset
    for g in graphs:
        # Convert graph to TensorFlow Example
        example = graph_to_example(g)
        # Write serialised example to file
        writer.write(example.SerializeToString())

    # Close writer
    writer.close()


# Decode a graph from a Tensorflow record
def string_record_to_graph(string_record):
    # Create a Tensorflow example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Represent molecular graph as a dictionary
    g = {}

    # Retrieve shape information
    shape_bytes = example.features.feature['shape'].bytes_list.value[0]
    g['shape'] = np.fromstring(shape_bytes, dtype=np.int64)
    n_nodes, n_edges, num_node_features, num_edge_features, n_targets, n_graphs = g['shape']

    # Retrieve NumPy arrays representing the graph
    if 'node_features' in example.features.feature:
        node_features_string = example.features.feature['node_features'].bytes_list.value[0]
        g['node_features'] = np.reshape(np.fromstring(node_features_string, dtype=np.float32),
                                        (n_nodes, num_node_features))
    if 'adj_mat' in example.features.feature:
        adj_mat_string = example.features.feature['adj_mat'].bytes_list.value[0]
        g['adj_mat'] = np.reshape(np.fromstring(adj_mat_string, dtype=np.int64), (-1, 2))
    if 'inc_mat' in example.features.feature:
        inc_mat_string = example.features.feature['inc_mat'].bytes_list.value[0]
        g['inc_mat'] = np.reshape(np.fromstring(inc_mat_string, dtype=np.int64), (-1, 2))
    if 'target' in example.features.feature:
        target_string = example.features.feature['target'].bytes_list.value[0]
        g['target'] = np.reshape(np.fromstring(target_string, dtype=np.float32), (n_graphs, n_targets))
    if 'node_graph_map' in example.features.feature:
        node_graph_map_string = example.features.feature['node_graph_map'].bytes_list.value[0]
        g['node_graph_map'] = np.fromstring(node_graph_map_string, dtype=np.int64)
    if 'edge_graph_map' in example.features.feature:
        edge_graph_map_string = example.features.feature['edge_graph_map'].bytes_list.value[0]
        g['edge_graph_map'] = np.fromstring(edge_graph_map_string, dtype=np.int64)
    if 'id' in example.features.feature:
        id_string = example.features.feature['id'].bytes_list.value[0]
        g['id'] = np.fromstring(id_string, dtype=np.int64)

    return g


# Encode a graph as a TensorFlow Example object
def graph_to_example(g):
    feature_dict = {k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[g[k].tobytes()])) for k in g.keys()}

    # Create TensorFlow Example object
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example

