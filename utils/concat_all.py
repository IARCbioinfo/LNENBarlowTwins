import numpy as np
import os
import cv2
import pandas as pd
import argparse
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Concat encoded vectors')
parser.add_argument( "--enc_vectors_dir", required=True, help="folder containing the encoded vectors to concat")

args = parser.parse_args()
root_enc = args.enc_vectors_dir
   
encoded_df = pd.DataFrame()
print('1 ----------------- Concatenation ----------------------------')
for tne_proj in os.listdir(root_enc):
    if tne_proj.find('TNE') != -1:
        print(tne_proj)
        print(os.path.join(root_enc, tne_proj))
        c_df = pd.read_csv(os.path.join(root_enc, tne_proj)) 
        encoded_df = encoded_df.append(c_df) 

print("********************************************************************")
print('2 ----------------- Write all ----------------------------')

print(os.path.join(root_enc, "barlowTwins_trainset_z128_all_projectors.csv"))
print(encoded_df.head())
encoded_df.to_csv(os.path.join(root_enc, "barlowTwins_trainset_z128_all_projectors.csv"))