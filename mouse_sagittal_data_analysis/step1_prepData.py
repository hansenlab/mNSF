

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
from scanpy import pp
from mNSF.NSF import preprocess

# %%
#### User inputs
dpth="data"

# %%
##### load data, create annData objects, save
## sample 1
Y=pd.read_csv(path.join(dpth,'Y_sample1.csv'))
X=pd.read_csv(path.join(dpth,'X_sample1.csv'))
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

ad.write_h5ad(path.join(dpth,"data_s1.h5ad"),compression="gzip")


## sample 2
Y=pd.read_csv(path.join(dpth,'Y_sample2.csv'))
X=pd.read_csv(path.join(dpth,'X_sample2.csv'))
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

ad.write_h5ad(path.join(dpth,"data_s2.h5ad"),compression="gzip")

## sample 3 
Y=pd.read_csv(path.join(dpth,'Y_sample3.csv'))
X=pd.read_csv(path.join(dpth,'X_sample3.csv'))
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

ad.write_h5ad(path.join(dpth,"data_s3.h5ad"),compression="gzip")

## sample 1
Y=pd.read_csv(path.join(dpth,'Y_sample4.csv'))
X=pd.read_csv(path.join(dpth,'X_sample4.csv'))
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

ad.write_h5ad(path.join(dpth,"data_s4.h5ad"),compression="gzip")

# %%
