
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
from scanpy import pp
from NSF import misc,preprocess,visualize




##### load data
Y=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/Jan3_2022_LukasData_mgcv_correctedModel/data/counts_sample_tmp_HVG_t_sample1_shared.csv')
X=pd.read_csv("/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/Jan3_2022_LukasData_mgcv_correctedModel/data/X_allSpots_sample1.csv")
X = preprocess.rescale_spatial_coords(X_)
X=X.to_numpy()


##### format data
ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/')
ad.write_h5ad(path.join(dpth,"ggblocks_lr.h5ad"),compression="gzip")








