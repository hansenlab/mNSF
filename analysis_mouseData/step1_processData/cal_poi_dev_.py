#module avail python
#module load python/3.9.10
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
from scanpy import pp

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/')
from utils import misc,preprocess,visualize

from utils import preprocess
from scipy import sparse


##############

def deviancePoisson(X, sz=None):
  """
  X is matrix-like with observations in ROWs and features in COLs,
  sz is a 1D numpy array that is same length as rows of X
    sz are the size factors for each observation
    sz defaults to the row means of X
    set sz to 1.0 to have "no size factors" (ie Poisson for absolute counts instead of relative).

  Note that due to floating point errors the results may differ slightly
  between deviancePoisson(X) and deviancePoisson(X.todense()), use higher
  floating point precision on X to reduce this.
  """
  dtp = X.dtype
  X = X.astype("float64") #convert dtype to float64 for better numerics
  if sz is None:
    sz = np.ravel(X.mean(axis=1))
  else:
    sz = np.ravel(sz).astype("float64")
  if len(sz)==1:
    sz = np.repeat(sz,X.shape[0])
  feature_sums = np.ravel(X.sum(axis=0))
  print(feature_sums)
  print(sz.sum())
  with np.errstate(divide='raise'):
  	ll_null=feature_sums *np.log(100000+feature_sums/(sz.sum())) 
  	ll_null[feature_sums>0] = feature_sums[feature_sums>0] *np.log(feature_sums[feature_sums>0]/(sz.sum()))
  if sparse.issparse(X): #sparse scipy matrix
    LP = sparse.diags(1./sz)@X
    #LP is a NxJ matrix with same sparsity pattern as X
    LP.data = np.log(LP.data) #log transform nonzero elements only
    ll_sat = np.ravel(X.multiply(LP).sum(axis=0))
  else: #dense numpy array
    X = np.asarray(X) #avoid problems with numpy matrix objects
    sz = np.expand_dims(sz,-1) #coerce to Nx1 array for broadcasting
    with np.errstate(divide='ignore',invalid='ignore'):
      ll_sat = X*np.log(X/sz) #sz broadcasts across rows of X
    ll_sat = ll_sat.sum(axis=0, where=np.isfinite(ll_sat))
  return (2*(ll_sat-ll_null)).astype(dtp)





##### load data
############ sample 1
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_alllGenes_2percentSparseFilter_sample1_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample1.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)


## cal deviance_poisson
ad.var['deviance_poisson'] = deviancePoisson(ad.layers["counts"])
#o = np.argsort(-ad.var['deviance_poisson'])
#idx = list(range(ad.shape[0]))

dev_=ad.var['deviance_poisson']
DF = pd.DataFrame(dev_) 
DF.to_csv(path.join("/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/dev_poisson_sample1.csv"))


############ sample 2
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_alllGenes_2percentSparseFilter_sample2_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample2.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)


## cal deviance_poisson
ad.var['deviance_poisson'] = deviancePoisson(ad.layers["counts"])
#o = np.argsort(-ad.var['deviance_poisson'])
#idx = list(range(ad.shape[0]))

dev_=ad.var['deviance_poisson']
DF = pd.DataFrame(dev_) 
DF.to_csv(path.join("/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/dev_poisson_sample2.csv"))




############ sample 3
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_alllGenes_2percentSparseFilter_sample3_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample3.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)



## cal deviance_poisson
ad.var['deviance_poisson'] = deviancePoisson(ad.layers["counts"])
#o = np.argsort(-ad.var['deviance_poisson'])
#idx = list(range(ad.shape[0]))

dev_=ad.var['deviance_poisson']
DF = pd.DataFrame(dev_) 
DF.to_csv(path.join("/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/dev_poisson_sample3.csv"))





############ sample 4
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_alllGenes_2percentSparseFilter_sample4_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample4.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)



## cal deviance_poisson
ad.var['deviance_poisson'] = deviancePoisson(ad.layers["counts"])
#o = np.argsort(-ad.var['deviance_poisson'])
#idx = list(range(ad.shape[0]))

dev_=ad.var['deviance_poisson']
DF = pd.DataFrame(dev_) 
DF.to_csv(path.join("/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/dev_poisson_sample4.csv"))


