# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .ipy
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---



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
import psutil
psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
import numpy as np
import os
import sys 
import pickle
import pandas as pd
from scanpy import pp

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}



#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')

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
#from utils import preprocess,misc,training,visualize,postprocess
from models import cf,pf,pfh

import numpy as np

from os import path
from scanpy import read_h5ad
from tensorflow_probability import math as tm
tfk = tm.psd_kernels

from models import cf,pf,pfh
from utils import preprocess,misc,training_fullData_12,visualize,postprocess
from models import pf_fit12

#from models import pf_fit12_fullSample_2_3
from models import pf_fit12_fullSample_2_3

#fit12_ini=pf_ori_mod.ProcessFactorization_self12(fit1,fit2,J,L,Z1=D1['X'],Z2=D2['X'], psd_kernel=ker,nonneg=True,lik="poi") #run without error message


from utils import training_oneSample

from utils import training_ori

#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')

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


###############################################################################################################

###############################################################################################################


############## Data loading
from scanpy import read_h5ad
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/'

## sample 1
#ad = read_h5ad((dpth,"data_s1.h5ad"))
Y=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_sample1.csv')
X=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/X_sample1.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D1=D


## sample 2
#ad = read_h5ad((dpth,"data_s2.h5ad"))
## sample 2
Y=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/Y_sample2.csv')
X=pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/X_sample2.csv')
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D2=D





#############
################################################################################################################
################################################################################################################
################################################################################################################
##############
# %% Initialize inducing poi_sz-constantnts
import random

L = 8
#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')
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


##################################################################
##################################################################
##################################################################
##################################################################

##################################################################
##################################################################

##################################################################
##################################################################
################### split into batches
##################################################################
##################################################################


nsample_=2
nbatch_eachDim=2

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





D1["Z"]=D1['X']
D2["Z"]=D2['X']

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
#D12['Z']=D12['X']



###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')


###################
###################
###################


list_D__=[D1,D2]
psutil.Process().memory_info().rss / (1024 * 1024 * 1024)

D12=list_D__[1]


fit12=pf_ori_mod.ProcessFactorization(J,L,D12['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit12.init_loadings(D12["Y"],X=D12['X'],sz=D12["sz"],shrinkage=0.3)

## 1500 spots per sample show memory issues after running 15mins (a few iterations throughout the 12 samples)
mpth = path.join(pth,"_models_L8_")#100 batches - finished running one round of sample 1-12, stuck at sample 2 at the 2nd round
#mpth = path.join(pth,"models__nsample_memorySaving_qsub_gpu_fullDasta_L7_byBatches_v9_rep2_")#10 batches

pp = fit12.generate_pickle_path("constant",base=mpth)
misc.mkdir_p(plt_pth) 
misc.mkdir_p(mpth)
misc.mkdir_p(pp)


list_Ds_train_batches_=[D1_train,D2_train]


#del ad
#del D1,D2,D3

tro_ = training_fullData_12.ModelTrainer(fit12,pickle_path=pp)#run without error message
#tro = training.ModelTrainer.from_pickle(pp)




tro_1 = training_ori.ModelTrainer(fit12,pickle_path=pp)#run without error message
del fit12



#tro_ = training_fullData_12.ModelTrainer(fit12,pickle_path=pp)#run without error message
#tro = training.ModelTrainer.from_pickle(pp)
#tro_1 = training_ori.ModelTrainer(fit12,pickle_path=pp)#run without error message
#del fit12
for kkk in range(0,50):
	fit =tro_.train_model([tro_1],
        	list_Ds_train_batches_,list_D__)## 1 iteration


#tro_1 = training_ori.ModelTrainer(fit12,pickle_path=pp)#run without error message
#del fit12
#fit =tro_.train_model([tro_1],
#        list_Ds_train_batches_[0:4],list_D__[0:4])## 0190 training complete, converged.





### update the restored result
#nsample=12
#for kkk in range(0,nsample):
#            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
#              list_para_tmp = pickle.load(inp)
#            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore.pkl')


### restore

os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')



##nsample=12
##for kkk in range(0,nsample):
##            with open('list_para_'+ str(kkk+1) +'_restore.pkl', 'rb') as inp:
##              list_para_tmp = pickle.load(inp)
##            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

##nsample=12
##for kkk in range(0,nsample):
##            with open('list_para_'+ str(kkk+1) +'_restore_ori.pkl', 'rb') as inp:
##              list_para_tmp = pickle.load(inp)
##            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

##nsample=12
##for kkk in range(0,nsample):
##            with open('list_para_'+ str(kkk+1) +'_restore_ori.pkl', 'rb') as inp:
##              list_para_tmp = pickle.load(inp)
##            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore.pkl')



##nsample=12
##for kkk in range(0,nsample):
##            with open('list_para_'+ str(kkk+1) +'_restore.pkl', 'rb') as inp:
##              list_para_tmp = pickle.load(inp)
##            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore_ori.pkl')



