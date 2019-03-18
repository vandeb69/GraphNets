import time
import os
import urllib.request
import pandas as pd

print("Downloading toxin dataset...", end=' ', flush=True)

tic = time.time()

# Path where data will be stored
TOXIN_PATH = "data/smiles/toxin"

# Create output folders (if they do not exist)
if not os.path.exists(TOXIN_PATH):
    os.makedirs(TOXIN_PATH)

# Download delaney dataset
url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-05-22-tox/sr-mmp.smiles-processed.csv"
urllib.request.urlretrieve(url, 'sr-mmp.smiles-processed.csv')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
data = pd.read_csv('sr-mmp.smiles-processed.csv')
data['target'] = data['target'].apply(lambda x: 1 - x)
data['target'].to_csv(f"{TOXIN_PATH}/targets.dat", header=False, index=False, sep=' ')
data['smiles'].to_csv(f"{TOXIN_PATH}/smiles.dat", header=False, index=False, sep=' ')

# Delete original file
os.remove('sr-mmp.smiles-processed.csv')

toc = time.time()

print(f"Done in {toc - tic:.02f} seconds", flush=True)
