


##################################################################
import sys

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/2023_05_08_mNSFH/mNSFH_May8_2023/functions'
sys.path.append(dir_mNSF_functions)

import load_packages # load functions from mNSF and NSF, as well as the dependencies




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







#############
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


nsample_=4
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


# step2, initiate fit1, fit2 
##################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

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
#del D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12
psutil.Process().memory_info().rss / (1024 * 1024 * 1024)

#D12=list_D__[11]


#fit12=pf_ori_mod.ProcessFactorization(J,L,D12['Z'],psd_kernel=ker,nonneg=True,lik="poi")
#fit12.init_loadings(D12["Y"],X=D12['X'],sz=D12["sz"],shrinkage=0.3)

## 1500 spots per sample show memory issues after running 15mins (a few iterations throughout the 12 samples)
#mpth = path.join(pth,"models__nsample_memorySaving_qsub_gpu_fullDasta_L7_introducings_1000_qsub_")#100 batches - finished running one round of sample 1-12, stuck at sample 2 at the 2nd round
#mpth = path.join(pth,"models__nsample_memorySaving_qsub_gpu_fullDasta_L7_byBatches_v9_rep2_")#10 batches

#pp = fit12.generate_pickle_path("constant",base=mpth)
#misc.mkdir_p(plt_pth) 
#misc.mkdir_p(mpth)
#misc.mkdir_p(pp)


list_Ds_train_batches_=[D1_train,D2_train,D3_train,D4_train]




#tro_ = training.ModelTrainer(fit12,pickle_path=pp)#run without error message
#tro = training.ModelTrainer.from_pickle(pp)
#tro_1 = training_ori.ModelTrainer(fit12,pickle_path=pp)#run without error message
#fit =tro_.train_model([tro_1],
#        list_Ds_train_batches_,list_D__)## 0190 training complete, converged.

###


### update the restored result
### nsample=12
### for kkk in range(0,nsample):
###             with open('list_para_'+ str(kkk+1) +'_1000spotsPerSample.pkl', 'rb') as inp:
###               list_para_tmp = pickle.load(inp)
###            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')

##nsample=12
##for kkk in range(0,nsample):
##            with open('list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl', 'rb') as inp:
##              list_para_tmp = pickle.load(inp)
##            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample.pkl')


### restore

##nsample=12
##for kkk in range(0,nsample):
##            with open('list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl', 'rb') as inp:
##              list_para_tmp = pickle.load(inp)
##            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample.pkl')


## save the result permernantly

nsample=4
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'__fullData_fitted_L20_mouse_Sagittal_4samples_addmarkergenes_olfIncluded.pkl')# 50 iterations



##################################################################################################################
##################################################################################################################
##################################################################################################################
######################################################### plot ###################################################
##################################################################################################################
##################################################################################################################

hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}

#list_fit__=[fit12,fit12,fit12,fit12,fit12,fit12,fit12,fit12,fit12,fit12,fit12,fit12]


reload(pf_ori_mod)
fit1=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit4=pf.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")


psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
3.7205276489257812

#fit1.init_loadings(D["Y"],X=X,sz=D["sz"],shrinkage=0.3)
#fit1.init_loadings(D1["Y"],X=X,sz=None,shrinkage=0.3)
fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)


psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
3.7584152221679688



list_fit__=[fit1,fit2,fit3,fit4]
nsample=4
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'__fullData_fitted_L20_mouse_Sagittal_4samples_addmarkergenes_olfIncluded.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
            training_multiSample.assign_paras_from_np_to_tf(list_fit__[kkk],list_para_tmp)
            list_fit__[kkk].Z=list_D__[kkk]["Z"]
            list_fit__[kkk].sample_latent_GP_funcs(list_D__[kkk]['X'],S=100,chol=True)




#########################################################
dpth='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/'
os.chdir(dpth)


save_object(list_fit__, 'fit_threeSampleNSF_list_fit_L20_mouse_Sagittal_addmarkergenes_olfIncluded.pkl')

with open('fit_threeSampleNSF_list_fit_L20_mouse_Sagittal_addmarkergenes_olfIncluded.pkl', 'rb') as inp:
	list_fit__ = pickle.load(inp)
	
	

######################################################### deviance
nsample=4

for kkk in range(0,nsample):
  fit_tmp=list_fit__[kkk]
  D_tmp=list_D__[kkk]
  Mu_tr,Mu_val = fit_tmp.predict(D_tmp,Dval=None,S=10)
  Mu_tr_df = pd.DataFrame(Mu_tr) 
  #misc.poisson_deviance(D_tmp["Y"],Mu_tr)
  Mu_tr_df.to_csv("Mu_tr_df_twoSampleNSF_sample_"+str(kkk+1)+"_L20_addmarkergenes_olfIncluded.csv")




######################################################### factors
list_X__=[D1["X"],D2["X"],D3["X"],D4["X"]]

M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]
M3_Z_=M2_Z_+D3["Z"].shape[0]
M4_Z_=M3_Z_+D4["Z"].shape[0]


M1_Y_=D1["Y"].shape[0]
M2_Y_=M1_Z_+D2["Y"].shape[0]
M3_Y_=M2_Z_+D3["Y"].shape[0]
M4_Y_=M3_Z_+D4["Y"].shape[0]


  
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



#interpret_nonneg(np.exp
inpf12 = interpret_npf_v3(list_fit__,list_X__,S=100,lda_mode=False)
Fplot = inpf12["factors"][:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]



#cwd = os.getcwd()



#fig,axes=visualize.multiheatmap(D12["X"], Fplot[:D12["X"].shape[0],:], (1,12), cmap="Blues", **hmkw)

# save the loading of each gene into csv filr
loadings=visualize.get_loadings(list_fit__[0])
#>>> loadings.shape
#(900, 7)
# convert array into dataframe
import pandas as pd

DF = pd.DataFrame(loadings) 
# save the dataframe as a csv file
DF.to_csv(("loadings_NPF_sample1_2_v2_L20__fullData_50iterations__addmarkergenes_olfIncluded.csv"))



#Wdf=pd.DataFrame(loadings*inpf12['factors'][:,None], index=ad.var.index, columns=range(1,L+1))

#Wdf=pd.DataFrame(loadings*inpf12['factors'][:,None],  columns=range(1,L+1))


W = inpf12["loadings"]#*inpf["totals"][:,None]
Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_NPF_sample1_2_v2_L20__fullData_50iterations_spde__addmarkergenes_olfIncluded.csv"))



DF = pd.DataFrame(Fplot[: M1_Y_,:]) 
#DF.to_csv("NPF_sample2_v2_edited_L7_v3_fulldata_nsample.csv")
DF.to_csv(path.join("NPF_sample1_L20_v2_fulldata_fourSample___addmarkergenes_olfIncluded.csv"))
DF = pd.DataFrame(Fplot[M1_Y_: M2_Y_,:]) 
DF.to_csv(path.join("NPF_sample2_L20_v2_fulldata_fourSample___addmarkergenes_olfIncluded.csv"))
DF = pd.DataFrame(Fplot[M2_Y_: M3_Y_,:]) 
DF.to_csv(path.join("NPF_sample3_L20_v2_fulldata_fourSample___addmarkergenes_olfIncluded.csv"))
DF = pd.DataFrame(Fplot[M3_Y_: M4_Y_,:]) 
DF.to_csv(path.join("NPF_sample4_L20_v2_fulldata_fourSample___addmarkergenes_olfIncluded.csv"))




