import os
import urllib.request
import pandas as pd
import time

print("Downloading delaney dataset...", end=' ', flush=True)

tic = time.time()

# Path where data will be stored
DELANEY_PATH = "data/smiles/delaney"

# Create output folders (if they do not exist)
if not os.path.exists(DELANEY_PATH):
    os.makedirs(DELANEY_PATH)

# Download delaney dataset
url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-05-24-delaney/delaney-processed.csv"
urllib.request.urlretrieve(url, 'delaney-processed.csv')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
data = pd.read_csv('delaney-processed.csv')
data['measured log solubility in mols per litre'].to_csv(f"{DELANEY_PATH}/targets.dat",
                                                         header=False, index=False, sep=' ')
data['smiles'] = data['smiles'].apply(lambda x: x.strip())
data['smiles'].to_csv(f"{DELANEY_PATH}/smiles.dat", header=False, index=False, sep=' ')

# Delete original file
os.remove('delaney-processed.csv')

toc = time.time()

print(f"Done in {toc - tic:.02f} seconds", flush=True)
