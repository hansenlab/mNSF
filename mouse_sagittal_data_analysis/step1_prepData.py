#module avail python
#module load python/3.9.10

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
from scanpy import pp

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/')
from utils import misc,preprocess,visualize




##### load data
## sample 1
Y=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_sample1.csv')
X=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/X_sample1.csv')
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
Y=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_sample2.csv')
X=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/X_sample2.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

#os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/SeqFish_JoePaper/data/')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/'
ad.write_h5ad(path.join(dpth,"data_s2.h5ad"),compression="gzip")



## sample 4
## sample 5
## sample 6
## sample 7





