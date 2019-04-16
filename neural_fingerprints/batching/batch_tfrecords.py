import argparse, os, time, operator
import numpy as np
from neural_fingerprints.parsing.tfrecord_to_graph import read_graphs_from_tfrecord


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecords_file", help="File with one TFRecord per graph in the dataset.", type=str)
    parser.add_argument("tfrecords_path", help="Directory where TFRecords representing the graph mini-batches will be "
                                               "stored.", type=str)
    parser.add_argument("batch_size", help="Number of graphs per batch.", type=int)
    # Train/Val/Test splits can be specified either with an input file containing the indices of the graphs to be
    # included in each dataset or via desired proportions. In the latter case, a random partition will be created
    split_args = parser.add_mutually_exclusive_group(required=True)
    split_args.add_argument("--split_indices_file", type=str, help="File containing indices of graphs assigned to"
                                                                   "training, validation and test datasets, in that"
                                                                   "order. One row per split (can be empty).")
    split_args.add_argument("--split_proportions", type=float, nargs=3, help="proportion of graphs to be used for "
                                                                             "training, validation and test, in that "
                                                                             "order. Must add up to 1.0.")
    parser.add_argument("--seed", type=int, help="Seed for train/val/test split. Unused if split indices are provided.",
                        default=0)
    args = parser.parse_args()

    return args


def read_splits(filename):
    # Check whether the file exists, raising an error if not
    if os.path.isfile(filename):
        # For now, we assume that the file format is correct if it exists
        with open(filename, 'r') as f:
            tr_idx = np.array(map(int, f.readline().strip().split()))
            val_idx = np.array(map(int, f.readline().strip().split()))
            tst_idx = np.array(map(int, f.readline().strip().split()))
    else:
        raise ValueError(f'Train/Val/Test indices file {filename} does not exist.')

    return tr_idx, val_idx, tst_idx


def make_random_splits(n_graphs, p_tr, p_val, p_tst, random_state=0):
    # Make sure the proportions are positive and add up to 1.0
    if not (p_tr > 0 and p_val > 0 and p_tst > 0 and np.allclose(p_tr + p_val + p_tst, 1.0)):
        raise ValueError('Train/Val/Test proportions must be positive and add up to 1.0.')

    # Set random seed
    np.random.seed(random_state)

    # Number of training, validation and test graphs
    n_tr, n_val = int(p_tr * n_graphs), int(p_val * n_graphs)
    n_tst = max(0, n_graphs - (n_tr + n_val))

    # random permutation of graph indices
    perm = np.random.permutation(n_graphs)
    tr_idx, val_idx, tst_idx = np.sort(perm[:n_tr]), np.sort(perm[n_tr:(n_tr + n_val)]), np.sort(perm[(n_tr + n_val):])

    return tr_idx, val_idx, tst_idx


def load_graphs(filename):
    print(f"Reading graph representations from TFRecords file {filename}...")
    tic = time.time()
    graphs = read_graphs_from_tfrecord(filename)
    n_graphs = len(graphs)
    toc = time.time()
    print(f"\nRead {n_graphs} graphs in {toc - tic:0.3f} seconds.\n")

    return graphs, n_graphs


def make_batches(graphs, batch_size, random_state=0):
    # Number of graphs
    n_graphs = len(graphs)
    # Number of batches
    n_batches = n_graphs/batch_size + ((n_graphs % batch_size) != 0)

    # Number of node features, edge features and targets that the graphs have, respectively
    num_node_features, num_edge_features, n_targets = graphs[0]['shape'][2:5]

    # Set random seed to shuffle examples prior to forming batches
    np.random.seed(random_state)
    perm = np.random.permutation(n_graphs)

    # Prepare batches
    batches = []
    for i in range(n_batches):
        # Create dictionary representing the batch of graphs
        batch = {}

        # Retrieve the graphs (randomly) assigned to the i-th batch
        batch_idx = perm[(i*batch_size):min((i+1)*batch_size, n_graphs)]
        batch_graphs = operator.itemgetter(*batch_idx)(graphs)
        n_graphs_batch = len(batch_graphs)

        # TODO: continue



def main():
    # PARSE INPUT ARGUMENTS
    args = parse_input_arguments()

    # LOAD GRAPHS FROM TFRECORDS FILE
    graphs, n_graphs = load_graphs(args.tfrecords_file)

    # MAKE TRAINING/VALIDATION/TEST SPLITS
    if args.split_indices_file is not None:
        tr_idx, val_idx, tst_idx = read_splits(args.split_indices_file)
    elif args.split_proportions is not None:
        p_tr, p_val, p_tst = args.split_proportions
        tr_idx, val_idx, tst_idx = make_random_splits(n_graphs, p_tr, p_val, p_tst, random_state=args.seed)

    # PREPARE MINI-BATCHES



if __name__ == "__main__":
    main()
