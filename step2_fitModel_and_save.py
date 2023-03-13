####
#### mNSF: data loading and formating, mNSF model fitting, result saving
####


import numpy as np
import os
import sys 
import pickle
import pandas as pd


import random
from os import path
from scanpy import read_h5ad
from importlib import reload  
import numpy as np
from tensorflow_probability import math as tm
tfk = tm.psd_kernels



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}



## load functions for  mNSF package
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')

from utils import preprocess
import numpy as np
from os import path
from scanpy import read_h5ad
from models import pf_ori_mod
from utils.misc import t2np


def interpret_npf_v3(list_fit,list_X,S=10,**kwargs):
  """
  fit: object of type PF with non-negative factors
  X: spatial coordinates to predict on
  returns: interpretable loadings W, factors eF, and total counts vector
  """
  kk=0
  for fit_tmp in list_fit:
    kk=kk+1
  for kkk in range(0,kk):
    Fhat_tmp = misc.t2np(list_fit[kkk].sample_latent_GP_funcs(list_X[kkk],S=S,chol=False)).T #NxL
    if kkk==0:
      Fhat_c=Fhat_tmp
    else:
      Fhat_c=np.concatenate((Fhat_c,Fhat_tmp), axis=0)
  return interpret_nonneg(np.exp(Fhat_c),list_fit[kkk].W.numpy(),sort=False,**kwargs)



def interpret_nonneg(factors,loadings,lda_mode=False,sort=True):
  """
  Rescale factors and loadings from a nonnegative factorization
  to improve interpretability. Two possible rescalings:

  1. Soft clustering of observations (lda_mode=True):
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "factor_sums"
  factor_sums is the "n" in the multinomial
  (ie the sum of the counts per observations)

  2. Soft clustering of features (lda_mode=False):
  Rows of loadings matrix sum to one, cols of factors matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "feature_sums"
  feature_sums is similar to an intercept term for each feature
  """
  if lda_mode:
    W,eF,eFsum,Wsum = rescale_as_lda(factors,loadings,sort=sort)##!!!!
    return {"factors":eF,"loadings":W,"totals1":eFsum,"totals2":Wsum}
  else: #spatialDE mode
    eF,W,Wsum,eFsum = rescale_as_lda(loadings,factors,sort=sort)
    return {"factors":eF,"loadings":W,"totals1":Wsum,"totals2":eFsum}


def rescale_as_lda(factors,loadings,sort=True):
  """
  Rescale nonnegative factors and loadings matrices to be
  comparable to LDA:
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  """
  W = postprocess.deepcopy(loadings)
  eF = postprocess.deepcopy(factors)
  W,wsum = postprocess.normalize_cols(W)
  #eF1,eFsum = postprocess.normalize_rows(eF*wsum)##??
  eF,eFsum = postprocess.normalize_rows(eF*wsum)##??
  if sort:
    o = np.argsort(-eF.sum(axis=0))
    return W[:,o],eF[:,o],eFsum
  else:
    return W,eF,eFsum,wsum




from models import cf,pf,pfh

from utils import preprocess,misc,training_fullData_12,visualize,postprocess
from models import pf_fit12

from models import pf_fit12_fullSample_2_3

from utils import training_oneSample

from utils import training_ori




## load functions for one-sample NSF package
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')

from utils import training


pth = "simulations/ggblocks_lr"
dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)



###############################################################################################################
###############################################################################################################
#%% Data loading
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/')
#from models import cf,pf,pfh
#from models import cf,pf,pfh
rng = np.random.default_rng()
dtp = "float32"
pth = "simulations/ggblocks_lr"

dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)

ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori_20_percentage.h5ad"))

J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D1=D


#%% Data loading
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample2/')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample2/')
#from models import cf,pf,pfh
#from models import cf,pf,pfh
rng = np.random.default_rng()
dtp = "float32"
pth = "simulations/ggblocks_lr"

dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)

ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D2=D


#%% Data loading
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample3/')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample3/')
#from models import cf,pf,pfh
#from models import cf,pf,pfh
rng = np.random.default_rng()
dtp = "float32"
pth = "simulations/ggblocks_lr"

dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)

ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D3=D


#%% Data loading
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample4/')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample4/')
#from models import cf,pf,pfh
#from models import cf,pf,pfh
rng = np.random.default_rng()
dtp = "float32"
pth = "simulations/ggblocks_lr"

dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)

ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D4=D


################################################################################################################
################################################################################################################
################################################################################################################

L = 7 # number of spatial factors

M = N 


### data formatting

nsample_=4

import statistics
import shutil
from tensorflow.data import Dataset
from tensorflow_probability import math as tm
import tensorflow_probability as tfp
import tensorflow as tf

tv = tfp.util.TransformedVariable
tv = tfp.util.TransformedVariable
tfb = tfp.bijectors
tfk = tm.psd_kernels
ker = tfk.MaternThreeHalves

# step1, create formated data
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


#################
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')



## mNSF initialization
fit12=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit12.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)




## train mNSF 
mpth = path.join(pth,"mnsf_")
pp = fit12.generate_pickle_path("constant",base=mpth)
misc.mkdir_p(plt_pth) 
misc.mkdir_p(mpth)
misc.mkdir_p(pp)


list_Ds_train_batches_=[D1_train,D2_train,D3_train,D4_train]


tro_ = training_fullData_12.ModelTrainer(fit12,pickle_path=pp)

tro_1 = training_ori.ModelTrainer(fit12,pickle_path=pp)
del fit12
for iter in range(0,50):
  fit =tro_.train_model([tro_1],
        list_Ds_train_batches_[0:4],list_D__[0:4])## 1 iteration of parameter updatings



## save mNSF results
nsample=4


for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_fitted_ind1.pkl')




## save mNSF results
hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}


reload(pf_ori_mod)
fit1=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf_ori_mod.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf_ori_mod.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit4=pf_ori_mod.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")


fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)


##  mNSF iterations
list_fit__=[fit1,fit2,fit3,fit4]
nsample=4
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
            training.assign_paras_from_np_to_tf(list_fit__[kkk],list_para_tmp)
            list_fit__[kkk].Z=list_D__[kkk]["Z"]
            list_fit__[kkk].sample_latent_GP_funcs(list_D__[kkk]['X'],S=100,chol=True)

save_object(list_fit__, 'fit_mNSF_ind1.pkl')




## extract factors from the model fitting output
list_X__=[D1["X"],D2["X"],D3["X"],D4["X"]]
M1_Y_=D1["Y"].shape[0]
M2_Y_=M1_Y_+D2["Y"].shape[0]
M3_Y_=M2_Y_+D3["Y"].shape[0]
M4_Y_=M3_Y_+D4["Y"].shape[0]

inpf12 = interpret_npf_v3(list_fit__,list_X__,S=100,lda_mode=False)
Fplot = inpf12["factors"][:,[0,1,2,3,4,5,6]]



## plot the factors in the 2-dim space

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')

plt_pth="plot__fullData_50iterations_ind1"
misc.mkdir_p(plt_pth)

fig,axes=visualize.multiheatmap(D1["X"],Fplot[: M1_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(plt_pth,"sample1_v2.png"),bbox_inches='tight')
fig,axes=visualize.multiheatmap(D2["X"], Fplot[M1_Y_:M2_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(plt_pth,"sample2_v2.png"),bbox_inches='tight')
fig,axes=visualize.multiheatmap(D3["X"], Fplot[M2_Y_:M3_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(plt_pth,"sample3_v2.png"),bbox_inches='tight')
fig,axes=visualize.multiheatmap(D4["X"], Fplot[M3_Y_:M4_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(plt_pth,"sample4_v2.png"),bbox_inches='tight')


fig,axes=visualize.multiheatmap(D12["X"], Fplot[:D12["X"].shape[0],:], (1,7), cmap="Blues", **hmkw)


## save the loading of each gene into csv filr
loadings=visualize.get_loadings(list_fit__[1])


## convert array into dataframe
import pandas as pd

DF = pd.DataFrame(loadings) 

## save the factors as a csv file
DF.to_csv(path.join(plt_pth,"loadings_v2.csv"))


DF = pd.DataFrame(Fplot[: M1_Y_,:]) 
DF.to_csv(path.join(plt_pth,"sample1.csv"))
DF = pd.DataFrame(Fplot[M1_Y_: M2_Y_,:]) 
DF.to_csv(path.join(plt_pth,"sample2.csv"))
DF = pd.DataFrame(Fplot[M2_Y_: M3_Y_,:]) 
DF.to_csv(path.join(plt_pth,"sample3.csv"))
DF = pd.DataFrame(Fplot[M3_Y_: M4_Y_,:]) 
DF.to_csv(path.join(plt_pth,"sample4.csv"))




