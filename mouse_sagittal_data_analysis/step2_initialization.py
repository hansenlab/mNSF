

##################################################################
##################################################################


from anndata import AnnData
import sys
dpth="/dcs04/hansen/data/ywang/ST/mNSF_package/data/"

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSFH_2023_06_04_mNSFPerSampleTraining/functions'
sys.path.append(dir_mNSF_functions)

#import load_packages # load functions from mNSF and NSF, as well as the dependencies


##################################################################



###############################################################################################################

###############################################################################################################


############## Data loading
from scanpy import read_h5ad
#dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/'

## sample 1
#ad = read_h5ad((dpth,"data_s1.h5ad"))
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample1_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample1.csv')
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
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample2_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample2.csv')
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




## sample 3
#ad = read_h5ad((dpth,"data_s2.h5ad"))
## sample 2
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample3_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample3.csv')
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
D3=D



## sample 4
#ad = read_h5ad((dpth,"data_s2.h5ad"))
## sample 2
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample4_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample4.csv')
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
D4=D





################################################################################################################
################################################################################################################
################################################################################################################
##############
# %% Initialize inducing poi_sz-constantnts
import random

L = 20
#cwd = os.getcwd()
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_LukasData_sample2_3_qsub_fasterVersion_sub500_v2_topleft_500genes_layer1to4')
from utils import preprocess

from utils import preprocess,misc,training,visualize,postprocess

from models import cf,pf,pfh


import numpy as np

from os import path

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
################### split into batches
##################################################################
##################################################################


nsample_=4
nbatch_eachDim=1

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


list_D__=[D1,D2,D3,D4]



fit1=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit4=pf.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")


#fit1.init_loadings(D["Y"],X=X,sz=D["sz"],shrinkage=0.3)
#fit1.init_loadings(D1["Y"],X=X,sz=None,shrinkage=0.3)
fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)



##could remove omega in this step
fit12_=pf_multiSample.ProcessFactorization_fit12(J,L,
  np.concatenate((D1['Z'], D2['Z'], D3['Z'], D4['Z']), axis=0),
  nsample=nsample_,
  psd_kernel=ker,nonneg=True,lik="poi")


fit12_.init_loadings(np.concatenate((D1['Y'], D2['Y'], D3['Y'], D4['Y']), axis=0),
  list_X=[D1['X'], D2['X'], D3['X'], D4['X']],
  list_Z=[D1['Z'],D2['Z'], D3['X'], D4['X']],
  sz=np.concatenate((D1['sz'], D2['sz'], D3['sz'], D4['sz']), axis=0),shrinkage=0.3)
#save_object(fit12_, 'fit12___npf_fulldata_nsamele_memorySaving_qsub_fullData.pkl')





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






hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}




reload(training)

kkk=0
list_para_tmp=training_multiSsmpl.store_paras_from_tf_to_np(fit1)
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




kkk=0
list_para_tmp=training.store_paras_from_tf_to_np(fit1)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore_L12_STdata_mouse_Sagittal_addedMarkerGenes.pkl')

kkk=1
list_para_tmp=training.store_paras_from_tf_to_np(fit2)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore_L12_STdata_mouse_Sagittal_addedMarkerGenes.pkl')

kkk=2
list_para_tmp=training.store_paras_from_tf_to_np(fit3)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore_L12_STdata_mouse_Sagittal_addedMarkerGenes.pkl')


kkk=3
list_para_tmp=training.store_paras_from_tf_to_np(fit4)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_restore_L12_STdata_mouse_Sagittal_addedMarkerGenes.pkl')













