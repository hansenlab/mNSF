
# %%
# KW move all imports to top
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
from scanpy import pp
from scanpy import read_h5ad
import pickle
from mNSF.NSF import preprocess,training, pf, misc
from mNSF import pf_multiSample, training_multiSample, training_multiSample_perSample

from tensorflow.data import Dataset
from tensorflow_probability import math as tm
import tensorflow_probability as tfp
import tensorflow as tf

tv = tfp.util.TransformedVariable
tv = tfp.util.TransformedVariable
tfb = tfp.bijectors
tfk = tm.psd_kernels
ker = tfk.MaternThreeHalves

## KW - Richard added check here
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
try:
  tf.config.experimental.set_virtual_device_configuration(
      tf.config.experimental.list_physical_devices('GPU')[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16384)])
except RuntimeError as e:
  print(e)

# %% 
#### User inputs
dpth="data"
L = 5 # num patterns
nsample = 4
J=500
nIter = 50

run_ID = "KW_mouseSagittal_4samples" # to know what run our output is for
outDir = run_ID + "_out"
misc.mkdir_p(outDir)

mpth = path.join(outDir,"_models_mNSF_")
misc.mkdir_p(mpth)




# %%
#### Data loading
# KW already saved annData objects in step1 -- no need to recreate
# KW load in data same as in step2

## sample 1 
ad = read_h5ad(os.path.join(dpth,"data_s1.h5ad"))
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D1=D


## sample 2
ad = read_h5ad(os.path.join(dpth,"data_s2.h5ad"))
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D2=D


## sample 3
ad = read_h5ad(os.path.join(dpth,"data_s3.h5ad"))
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D3=D


## sample 4
ad = read_h5ad(os.path.join(dpth,"data_s4.h5ad"))
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D4=D

#%%
# step1, create D12
Ntr = D1["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D1)
D1_train = Dtrain.batch(round(Ntr)+1)

Ntr = D2["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D2)
D2_train = Dtrain.batch(round(Ntr)+1)

Ntr = D3["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D3)
D3_train = Dtrain.batch(round(Ntr)+1)


Ntr = D4["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D4)
D4_train = Dtrain.batch(round(Ntr)+1)

D1["Z"]=D1['X']
D2["Z"]=D2['X']
D3["Z"]=D3['X']
D4["Z"]=D4['X']

list_D__=[D1,D2,D3,D4]

#D12=list_D__[0]  KW note - Yi defined and never used, do we need this?

# KW none of these used?
#rng = np.random.default_rng()
#dtp = "float32"
#pth = "simulations/ggblocks_lr"
#dpth = path.join(pth,"data")
#mpth = path.join(pth,"models")
#plt_pth = path.join(pth,"results/plots")
#misc.mkdir_p(plt_pth)
#mpth = path.join(pth,"_models_mNSFH_")

#misc.mkdir_p(plt_pth) 
#misc.mkdir_p(mpth)

list_Ds_train_batches_=[D1_train,D2_train,D3_train,D4_train]


#%%
# process factorization
fit_ini=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit_ini.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)

picklePath = fit_ini.generate_pickle_path("constant",base=mpth) # renaming since we import pp package 
misc.mkdir_p(picklePath)


tro_ = training_multiSample_perSample.ModelTrainer(fit_ini,pickle_path=None)
tro_ini = training.ModelTrainer(fit_ini,pickle_path=None)


#%%
for iter in range(0,nIter):
	for ksample in range(0,nsample):
              fit =tro_.train_model([tro_ini],ksample,list_Ds_train_batches_,list_D__)## 1 iteration of parameter updatings
			  
### KW NOTE THIS ERRORS - "ResourceExhaustedError: {{function_node __wrapped__ConcatV2_N_2_device_/job:localhost/replica:0/task:0/device:GPU:0}} OOM when allocating tensor with shape[20,7263025] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:ConcatV2] name: concat"
# %%
