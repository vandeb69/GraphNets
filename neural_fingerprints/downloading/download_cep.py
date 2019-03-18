import os
import urllib.request
import tarfile
import pandas as pd
import time

# Complete CEP dataset
######################
print("Downloading complete CEP dataset...", end=' ', flush=True)

tic = time.time()

# Path where data will be stored
CEP_PATH = "data/smiles/cep"
CEP_NO_OUTLIERS_PATH = "data/smiles/cep_no_outliers"

# Create output folders (if they do not exist)
if not os.path.exists(CEP_PATH):
    os.makedirs(CEP_PATH)
if not os.path.exists(CEP_NO_OUTLIERS_PATH):
    os.makedirs(CEP_NO_OUTLIERS_PATH)

# Download complete CEP dataset
url = "https://github.com/HIPS/neural-fingerprint/raw/master/data/2015-06-02-cep-pce/data_cep.tar.gz"
urllib.request.urlretrieve(url, 'data_cep.tar.gz')

# Uncompress and delete original compressed file
tar = tarfile.open('data_cep.tar.gz', 'r:gz')
tar.extractall()
tar.close()
os.remove("data_cep.tar.gz")

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
data_tmp_moldata = pd.read_csv("data_tmp_moldata.csv")
data_tmp_moldata['smile'].to_csv(f"{CEP_PATH}/smiles.dat", header=False, index=False, sep=' ')
data_tmp_moldata['target'].to_csv(f"{CEP_PATH}/targets.dat", header=False, index=False, sep=' ')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
# removing all samples whose target value is 0.0 (which appear to behave as outliers in the CEP dataset)
data_tmp_moldata = data_tmp_moldata[data_tmp_moldata.target != 0]
data_tmp_moldata['smile'].to_csv(f"{CEP_NO_OUTLIERS_PATH}/smiles.dat", header=False, index=False, sep=' ')
data_tmp_moldata['target'].to_csv(f"{CEP_NO_OUTLIERS_PATH}/targets.dat", header=False, index=False, sep=' ')

# Delete original file
os.remove("data_tmp_moldata.csv")

toc = time.time()

print(f"Done in {toc - tic:.02f} seconds", flush=True)

# Reduced CEP dataset
#####################

print("Downloading reduced CEP dataset...", end=' ', flush=True)

tic = time.time()

# Path where data will be stored
REDUCED_CEP_PATH = "data/smiles/reduced_cep"
REDUCED_CEP_NO_OUTLIERS_PATH = "data/smiles/reduced_cep_no_outliers"

# Create output folders (if they do not exist)
if not os.path.exists(REDUCED_CEP_PATH):
    os.makedirs(REDUCED_CEP_PATH)
if not os.path.exists(REDUCED_CEP_NO_OUTLIERS_PATH):
    os.makedirs(REDUCED_CEP_NO_OUTLIERS_PATH)

# Download reduced CEP dataset
url = "https://github.com/HIPS/neural-fingerprint/raw/master/data/2015-06-02-cep-pce/cep-processed.csv"
urllib.request.urlretrieve(url, 'cep-processed.csv')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
cep_processed = pd.read_csv("cep-processed.csv")
cep_processed['smiles'].to_csv(f"{REDUCED_CEP_PATH}/smiles.dat", header=False, index=False, sep=' ')
cep_processed['PCE'].to_csv(f"{REDUCED_CEP_PATH}/targets.dat", header=False, index=False, sep=' ')

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
# removing all samples whose target value is 0.0 (which appear to behave as outliers in the CEP dataset)
cep_processed = cep_processed[cep_processed.PCE != 0]
cep_processed['smiles'].to_csv(f"{REDUCED_CEP_NO_OUTLIERS_PATH}/smiles.dat", header=False, index=False, sep=' ')
cep_processed['PCE'].to_csv(f"{REDUCED_CEP_NO_OUTLIERS_PATH}/targets.dat", header=False, index=False, sep=' ')

# Delete original file
os.remove("cep-processed.csv")

toc = time.time()

print(f"Done in {toc - tic:.02f} seconds", flush=True)
