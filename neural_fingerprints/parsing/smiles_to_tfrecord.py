import argparse, os, time
import numpy as np
from neural_fingerprints.parsing.smiles_parser import SMILESParser
import tensorflow as tf


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", help="File with one SMILES string per line.", type=str)
    parser.add_argument("output_file", help="Output file where TFRecords representing the graphs will be stored.", type=str)
    parser.add_argument("config_file", help="JSON file with the configuration for the SMILES parser.", type=str)
    parser.add_argument("--targets_file", help="File with a set of comma-separated targets per line. "
                                               "Number of lines must match number of lines in smiles_file.", type=str)
    args = parser.parse_args()

    return args


def read_smiles(smiles_file):
    print("Reading SMILES strings...", flush=True)
    tic = time.time()
    smiles_array = []
    with open(smiles_file, 'r') as f:
        for line in f:
            smiles_array.append(line.strip())
    n_smiles = len(smiles_array)
    toc = time.time()
    print(f"Read {n_smiles} SMILES strings in {toc - tic:0.3f} seconds.\n", flush=True)
    return smiles_array


def read_targets(targets_file):
    targets_array, n_targets = None, None
    if targets_file is not None:
        print("Reading targets...", flush=True)
        tic = time.time()
        targets_array = np.loadtxt(targets_file, delimiter=',', dtype=np.float32)
        n_targets = len(targets_array)
        toc = time.time()
        print(f"Read {n_targets} targets in {toc - tic:0.3f} seconds.\n", flush=True)
    return targets_array


# Encode a graph as a TensorFlow Example object
def graph_to_example(g):
    feature_dict = {k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[g[k].tobytes()])) for k in g.keys()}

    # Create TensorFlow Example object
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


# Write all graphs to a TFRecord file
def write_graphs_to_tfrecord(graphs, filename):
    # Create writer
    writer = tf.python_io.TFRecordWriter(filename)

    # Iterate across all graphs in the dataset
    for i, g in enumerate(graphs):
        print(f"\r\t{i + 1}/{len(graphs)}", end='', flush=True)
        # Convert graph to TensorFlow Example
        example = graph_to_example(g)
        # Write serialised example to file
        writer.write(example.SerializeToString())

    # Close writer
    writer.close()


def main():
    # Parse input arguments
    args = parse_input_arguments()

    # Read SMILES strings
    print("Reading SMILES strings...", flush=True)
    tic = time.time()
    smiles_array = []
    with open(args.smiles_file, 'r') as f:
        for line in f:
            smiles_array.append(line.strip())
    n_smiles = len(smiles_array)
    toc = time.time()
    print(f"Read {n_smiles} SMILES strings in {toc - tic:0.3f} seconds.\n", flush=True)

    # Read targets (if any)
    targets_array, n_targets = None, None
    if args.targets_file is not None:
        print("Reading targets...", flush=True)
        tic = time.time()
        targets_array = np.loadtxt(args.targets_file, delimiter='\n', dtype=np.float32)
        n_targets = len(targets_array)
        toc = time.time()
        print(f"Read {n_targets} targets in {toc - tic:0.3f} seconds.\n", flush=True)
    if targets_array is not None and n_smiles != n_targets:
        raise ValueError('smiles_file must have same number of lines as target_file')

    # Create graph representation of dataset
    print("Parsing SMILES strings...", flush=True, end="")
    tic = time.time()
    smiles_parser = SMILESParser(config=args.config_file)
    graphs = smiles_parser.parse_smiles(smiles_array=smiles_array, targets_array=targets_array)
    n_graphs = len(graphs)
    toc = time.time()
    print(f"\rParsed {n_smiles} SMILES strings in {toc - tic:0.3f} seconds.", flush=True)
    print(f"Parsing failed for {n_smiles - n_graphs}/{n_smiles} SMILES strings.\n", flush=True)

    # Create output directory (if it does not exist)
    output_dir = os.path.split(args.output_file)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Writing graph data to TFRecords file...", flush=True, end='')
    tic = time.time()
    write_graphs_to_tfrecord(graphs, os.path.join(args.output_file))
    toc = time.time()
    print(f"\rWrote graph data to TFRecords file {args.output_file} in {toc-tic:0.03f} seconds.")


if __name__ == "__main__":
    main()
