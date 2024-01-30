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
parser.add_argument( "--concat_dir", required=True, help="folder where concatenated projector are save")
parser.add_argument( "--patient_id", required=True, help="patient id to concatenate")


args = parser.parse_args()
root_enc = args.enc_vectors_dir
   
os.makedirs(args.concat_dir, exist_ok=True)
ori_tne_list  = len(os.listdir(root_enc))               
encoded_df = pd.DataFrame()
first = True 
print('1 ----------------- Concatenation ----------------------------')
count = 1
tne_folder_count = 0
ele_in_c_df = 0 
tot_nb_ele = 0
list_tne_id = []
duplicated_tne = []
tneforlder2ignore = ["TNE0387-HPS", "TNE1007-HES", "TNE1007-HES.svs", "TNE0308-HPS", "TNE0868", "TNE1344", "TNE1344-HPS", "TNE0181-HPS", "TNE0591-HPS", "TNE0952.svs" ]
for tnefolder in os.listdir(root_enc):
   if tnefolder not in tneforlder2ignore:
        tne_folder_count += 1
        if tnefolder.find('TNE') != -1 and tnefolder.find(args.patient_id) != -1:
            print("tnefolder  ", tnefolder)
            ele_in_tne_folder  = 0
            if tnefolder[:7] in list_tne_id:
                print(tnefolder)
                duplicated_tne.append(tnefolder)
            list_tne_id.append(tnefolder[:7])
            for ele in os.listdir(os.path.join(root_enc, tnefolder)):
                ele_in_tne_folder += 1
                ele_in_c_df +=1 
                tot_nb_ele +=1
                encoded_v = np.load(os.path.join(root_enc, tnefolder, ele))
                encoded_v = np.transpose(encoded_v.flatten())
                c_df = pd.DataFrame(encoded_v) 
                c_df = c_df.T
                c_df['img_id']  = ele[:-4]
                c_df['tne_id']  = tnefolder[:7]
                encoded_df = encoded_df.append(c_df)
            print(os.path.join(args.concat_dir, f"barlowTwins_trainset_z128_{tnefolder[:7]}.csv"))
            encoded_df.to_csv(os.path.join(args.concat_dir, f"barlowTwins_trainset_z128_{tnefolder[:7]}.csv"), index = False)
            encoded_df = pd.DataFrame()
            count += 1
