import time
import os
import urllib.request
import pandas as pd
from math import log

# Raw malaria dataset
#####################

print("Downloading raw malaria dataset...", end=' ', flush=True)

tic = time.time()

# Path where data will be stored
MALARIA_PATH = "data/smiles/malaria"

# Create output folders (if they do not exist)
if not os.path.exists(MALARIA_PATH):
    os.makedirs(MALARIA_PATH)

# Download delaney dataset
url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/raw_csv.csv"
urllib.request.urlretrieve(url, 'raw_csv.csv')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
data = pd.read_csv('raw_csv.csv')
data['Activity (EC50 uM)'].apply(lambda x: log(x))
data['Activity (EC50 uM)'].to_csv(f"{MALARIA_PATH}/targets.dat", header=False, index=False, sep=' ')
data['Canonical_Smiles'].to_csv(f"{MALARIA_PATH}/smiles.dat", header=False, index=False, sep=' ')

# Delete original file
os.remove('raw_csv.csv')

toc = time.time()

print(f"Done in {toc - tic:.02f} seconds", flush=True)

# Reduced malaria dataset
#########################

print("Downloading reduced malaria dataset...", end=' ', flush=True)

tic = time.time()

# Path where data will be stored
REDUCED_MALARIA_PATH = "data/smiles/reduced_malaria"

# Create output folders (if they do not exist)
if not os.path.exists(REDUCED_MALARIA_PATH):
    os.makedirs(REDUCED_MALARIA_PATH)

# Download delaney dataset
url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv"
urllib.request.urlretrieve(url, 'malaria-processed.csv')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
data = pd.read_csv('malaria-processed.csv')
data['activity'].to_csv(f"{REDUCED_MALARIA_PATH}/targets.dat", header=False, index=False, sep=' ')
data['smiles'].to_csv(f"{REDUCED_MALARIA_PATH}/smiles.dat", header=False, index=False, sep=' ')

# Delete original file
os.remove('malaria-processed.csv')

toc = time.time()

print(f"Done in {toc - tic:.02f} seconds", flush=True)
