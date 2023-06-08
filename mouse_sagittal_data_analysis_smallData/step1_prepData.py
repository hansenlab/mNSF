import sys
sys.path.append("..")
from mNSF.NSF import preprocess
from os import path
import anndata
import scanpy
import pandas

dpth="data"

##### load data
## sample 1
Y = pandas.read_csv(path.join(dpth, 'Y_sample1_smallData.csv'))
X = pandas.read_csv(path.join(dpth, 'X_sample1_smallData.csv'))
X = mNSF.NSF.preprocess.rescale_spatial_coords(X)
X = X.to_numpy()

ad = anndata.AnnData(Y, obsm={"spatial":X})
ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
scanpy.pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
scanpy.pp.log1p(ad)
ad.write_h5ad(path.join(dpth, "data_s1.h5ad"), compression="gzip")


## sample 2
Y=pandas.read_csv(path.join(dpth, 'Y_sample2_smallData.csv'))
X=pandas.read_csv(path.join(dpth, 'X_sample2_smallData.csv'))
X=mNSF.NSF.preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = anndata.AnnData(Y,obsm={"spatial":X})
ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
scanpy.pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
scanpy.pp.log1p(ad)
ad.write_h5ad(path.join(dpth, "data_s2.h5ad"), compression="gzip")
