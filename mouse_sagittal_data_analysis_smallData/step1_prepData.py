


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
from scanpy import pp

import load_data

dpth="/dcs04/hansen/data/ywang/ST/data_10X_ST//mouse_Sagittal/put/"

dir_mNSF_functions='/dcs04/hansen/data/ywang/ST/mNSF_package/functions/'
sys.path.append(dir_mNSF_functions)
import load_packages



##### load data
## sample 1
Y=pd.read_csv(path.join(dpth,'Y_sample1_smallData.csv.csv'))
X=pd.read_csv(path.join(dpth,'X_sample1_smallData.csv.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

#os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/SeqFish_JoePaper/data/')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/'
ad.write_h5ad(path.join(dpth,"data_s1.h5ad"),compression="gzip")


## sample 2
Y=pd.read_csv(path.join(dpth,'Y_sample2_smallData.csv.csv')
X=pd.read_csv(path.join(dpth,'X_sample2_smallData.csv.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

#os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/SeqFish_JoePaper/data/')
ad.write_h5ad(path.join(dpth,"data_s2.h5ad"),compression="gzip")



## sample 4
## sample 5
## sample 6
## sample 7





