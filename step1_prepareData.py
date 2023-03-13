

##################################################################
##################################################################


##################################################################
##################################################################

import psutil
psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
import numpy as np
import os
import sys 
import pickle
import pandas as pd

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}



#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/mNSF')


from utils import preprocess

import numpy as np

from os import path
from scanpy import read_h5ad

#from models import cf,pf,pfh
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


###############################################################################################################
###############################################################################################################
from utils import preprocess,misc,training,visualize,postprocess
from models import cf,pf,pfh

import numpy as np

from os import path
from scanpy import read_h5ad
from tensorflow_probability import math as tm
tfk = tm.psd_kernels

from models import cf,pf,pfh
from utils import preprocess,misc,training,visualize,postprocess
from models import pf_fit12

#from models import pf_fit12_fullSample_2_3
from models import pf_fit12_fullSample_2_3

#fit12_ini=pf_ori_mod.ProcessFactorization_self12(fit1,fit2,J,L,Z1=D1['X'],Z2=D2['X'], psd_kernel=ker,nonneg=True,lik="poi") #run without error message


from utils import training_oneSample

from utils import training_ori

#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/NSF')


from utils import preprocess,misc,training,visualize,postprocess

from models import cf,pf,pfh


from tensorflow_probability import math as tm
tfk = tm.psd_kernels

from models import cf,pf,pfh
from utils import preprocess,misc,training,visualize,postprocess
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

ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))
#ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori_20_percentage.h5ad"))

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




#############
################################################################################################################
################################################################################################################
################################################################################################################
##############
# %% Initialize inducing poi_sz-constantnts
import random

L = 7
#cwd = os.getcwd()

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/NSF')
from utils import preprocess

from utils import preprocess,misc,training,visualize,postprocess

from models import cf,pf,pfh


import numpy as np

from os import path
from scanpy import read_h5ad
from tensorflow_probability import math as tm

from models import cf,pf,pfh
from utils import preprocess,misc,training,visualize,postprocess
from models import pf_fit12


# %% Initialize inducing poi_sz-constantnts
M = N #number of inducing points
#Z = X

from importlib import reload  


#########

nsample_=4


import statistics

import shutil


#Dval=None
from tensorflow.data import Dataset

from tensorflow_probability import math as tm

import tensorflow_probability as tfp
import tensorflow as tf

tv = tfp.util.TransformedVariable
tv = tfp.util.TransformedVariable
tfb = tfp.bijectors
tfk = tm.psd_kernels
ker = tfk.MaternThreeHalves


# step1, create D12
Ntr = D1["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D1)
D1_train = Dtrain.batch(round(Ntr)+1)
Ntr
Ntr = D2["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D2)
D2_train = Dtrain.batch(round(Ntr)+1)
Ntr
Ntr = D3["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D3)
D3_train = Dtrain.batch(round(Ntr)+1)
Ntr
Ntr = D4["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D4)
D4_train = Dtrain.batch(round(Ntr)+1)
Ntr


D1["Z"]=D1['X']
D2["Z"]=D2['X']
D3["Z"]=D3['X']
D4["Z"]=D4['X']

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/mNSF')



###################
#### mNSF model initialization
###################


list_D__=[D1,D2,D3,D4]


fit1=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf_ori_mod.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf_ori_mod.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit4=pf_ori_mod.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")



fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)



##
fit12_=pf_fit12_fullSample_2_3.ProcessFactorization_fit12(J,L,
  np.concatenate((D1['Z'], D2['Z'],D3['Z'],D4['Z']), axis=0),
  nsample=len(list_D__),
  psd_kernel=ker,nonneg=True,lik="poi")


fit12_.init_loadings(np.concatenate((D1['Y'], D2['Y'],D3['Y'],D4['Y']), axis=0),
  list_X=[D1['X'], D2['X'],D3['X'],D4['X']],
  list_Z=[D1['Z'],D2['Z'],D3['Z'],D4['Z']],
  sz=np.concatenate((D1['sz'], D2['sz'], D3['sz'],D4['sz']), axis=0),shrinkage=0.3)



M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]
M3_Z_=M2_Z_+D3["Z"].shape[0]
M4_Z_=M3_Z_+D4["Z"].shape[0]


delta1=fit12_.delta.numpy()[:,0:M1_Z_]
beta01=fit12_.beta0.numpy()[0:L,:]
beta1=fit12_.beta.numpy()[0:L,:]
W=fit12_.W.numpy()


delta2=fit12_.delta.numpy()[:,M1_Z_:M2_Z_]
beta02=fit12_.beta0.numpy()[L:(2*L),:]
beta2=fit12_.beta.numpy()[L:(2*L),:]

delta3=fit12_.delta.numpy()[:,M2_Z_:M3_Z_]
beta03=fit12_.beta0.numpy()[(2*L):(3*L),:]
beta3=fit12_.beta.numpy()[(2*L):(3*L),:]

delta4=fit12_.delta.numpy()[:,M3_Z_:M4_Z_]
beta04=fit12_.beta0.numpy()[(3*L):(4*L),:]
beta4=fit12_.beta.numpy()[(3*L):(4*L),:]

import tensorflow as tf
fit1.delta.assign(delta1) #LxM
fit1.beta0.assign(beta01) #LxM
fit1.beta.assign(beta1) #LxM
fit1.W.assign(W) #LxM

import pandas as pd

fit2.delta.assign(delta2) #LxM
fit2.beta0.assign(beta02) #LxM
fit2.beta.assign(beta2) #LxM
fit2.W.assign(W) #LxM

fit3.delta.assign(delta3) #LxM
fit3.beta0.assign(beta03) #LxM
fit3.beta.assign(beta3) #LxM
fit3.W.assign(W) #LxM

fit4.delta.assign(delta4) #LxM
fit4.beta0.assign(beta04) #LxM
fit4.beta.assign(beta4) #LxM
fit4.W.assign(W) #LxM


# save the parameters
reload(training)

kkk=0
list_para_tmp=training.store_paras_from_tf_to_np(fit1)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
list_para_tmp["Kuu_chol"].shape

kkk=1
list_para_tmp=training.store_paras_from_tf_to_np(fit2)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

kkk=2
list_para_tmp=training.store_paras_from_tf_to_np(fit3)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

kkk=3
list_para_tmp=training.store_paras_from_tf_to_np(fit4)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')


