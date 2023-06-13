## single analysis script for Yi's steps 2-4

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
from mNSF.NSF import preprocess,training, pf, misc, visualize, postprocess
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
J = 500
nIter = 50
nbatch_eachDim = 1 # KW - Yi defined this but never used?

run_ID = "KW_mouseSagittal_4samples_singleScript" # to know what run our output is for

outDir = run_ID + "_out" # make output directory using the run ID
misc.mkdir_p(outDir)

mpth = path.join(outDir,"_models_mNSF_") # in Yi code but nothing saved here
misc.mkdir_p(mpth)



# %%  ###### START STEP 2 ######
#### Data loading
# KW already saved annData objects in step1 -- no need to recreate

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


# %% Initialize inducing poi_sz-constantnts

M = N # KW number of inducing points equal to all points

### KW batch size is entire dataset -- do we want to batch here?
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
list_Ds_train_batches_=[D1_train,D2_train,D3_train,D4_train]



# %%
# KW fit each sample individually -- why? we overwrite the parameters later?
fit1=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit4=pf.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")

fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)


# KW fit samples together
##could remove omega in this step
fit12_=pf_multiSample.ProcessFactorization_fit12(J,L,
  np.concatenate((D1['Z'], D2['Z'], D3['Z'], D4['Z']), axis=0),
  nsample=nsample,
  psd_kernel=ker,nonneg=True,lik="poi")

fit12_.init_loadings(np.concatenate((D1['Y'], D2['Y'], D3['Y'], D4['Y']), axis=0),
  list_X=[D1['X'], D2['X'], D3['X'], D4['X']],
  list_Z=[D1['Z'],D2['Z'], D3['X'], D4['X']],
  sz=np.concatenate((D1['sz'], D2['sz'], D3['sz'], D4['sz']), axis=0),shrinkage=0.3)


# KW parse results and save to individual fits (why did we fit individually if overwriting here? are there other parameters learned?)
M1_Z_=D1["Z"].shape[0]
print(M1_Z_)
M2_Z_=M1_Z_+D2["Z"].shape[0]
M3_Z_=M2_Z_+D3["Z"].shape[0]
M4_Z_=M3_Z_+D4["Z"].shape[0]

M1_Y_=D1["Y"].shape[0]
print(M1_Y_)
M2_Y_=M1_Z_+D2["Y"].shape[0]
M3_Y_=M2_Z_+D3["Y"].shape[0]
M4_Y_=M3_Z_+D4["Y"].shape[0]

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

fit1.delta.assign(delta1) #LxM
fit1.beta0.assign(beta01) #LxM
fit1.beta.assign(beta1) #LxM
fit1.W.assign(W) #LxM

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

# %%
# KW eliminated duplicates

## KW added save_object funciton here from load_packages.py -- should abstract away (ask Yi if there is a preference where)
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# KW changed training.store... to traning_multiSample.store... since error "module 'mNSF.NSF.training' has no attribute 'store_paras_from_tf_to_np'"
kkk=0
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit1)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
list_para_tmp["Kuu_chol"].shape

kkk=1
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit2)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

kkk=2
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit3)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

kkk=3
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit4)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

#%% ###### START STEP 3 #########
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
# KW note - this updates the list_para_#.pkl files , right??

#%%
######## START STEP 4 #########

# KW - why are we resaving each one? just so we have saved output?
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, os.path.join(outDir,'list_para_' + str(kkk+1) +'_' + run_ID + '.pkl'))# 50 iterations


#%%
# KW - do we need to fit each one individually again?? I think we just overwrite them anyway

#fit1=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
#fit2=pf.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
#fit3=pf.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
#fit4=pf.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")

#fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
#fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
#fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
#fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)

#psutil.Process().memory_info().rss / (1024 * 1024 * 1024) # KW need to have this save somewhere instead of just printing it


list_fit__=[fit1,fit2,fit3,fit4]
for kkk in range(0,nsample):
            with open(os.path.join(outDir, 'list_para_' + str(kkk+1) +'_' + run_ID + '.pkl'), 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
            training_multiSample.assign_paras_from_np_to_tf(list_fit__[kkk],list_para_tmp) #KW note: assign pickled parameters to each in list_fit
            list_fit__[kkk].Z=list_D__[kkk]["Z"]
            list_fit__[kkk].sample_latent_GP_funcs(list_D__[kkk]['X'],S=100,chol=True)



# %%

##### KW - ASK YI, WHAT IS THIS FOR? saves object and then immediately reloads it? 
save_object(list_fit__, os.path.join(outDir, 'list_fit' + run_ID + '.pkl'))

#with open('fit_threeSampleNSF_list_fit_L20_mouse_Sagittal_addmarkergenes_olfIncluded.pkl', 'rb') as inp:
#	list_fit__ = pickle.load(inp)
	
	
for kkk in range(0,nsample):
  fit_tmp=list_fit__[kkk]
  D_tmp=list_D__[kkk]
  Mu_tr,Mu_val = fit_tmp.predict(D_tmp,Dval=None,S=10)
  Mu_tr_df = pd.DataFrame(Mu_tr) 
  #misc.poisson_deviance(D_tmp["Y"],Mu_tr)
  Mu_tr_df.to_csv(os.path.join(outDir,"Mu_tr_df_sample_"+str(kkk+1)+"_" + run_ID + ".csv"))


######################################################### factors
list_X__=[D1["X"],D2["X"],D3["X"],D4["X"]]

  
#%%
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
#Fplot = inpf12["factors"][:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
Fplot = inpf12["factors"][:, np.arange(L)]


# save the loading of each gene into csv filr
loadings=visualize.get_loadings(list_fit__[0]) # KW check all have same loadings?
loadingsDF = pd.DataFrame(loadings) 
# save the dataframe as a csv file
loadingsDF.to_csv(os.path.join(outDir, "loadings_NPF_" + run_ID + ".csv"))


W = inpf12["loadings"]#*inpf["totals"][:,None]
Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))
Wdf.to_csv(os.path.join(outDir, "loadings_NPF_spde" + run_ID + ".csv")) # KW question what is this


DF = pd.DataFrame(Fplot[: M1_Y_,:]) 
#DF.to_csv("NPF_sample2_v2_edited_L7_v3_fulldata_nsample.csv")
DF.to_csv(os.path.join(outDir, "NPF_sample1_" + run_ID + ".csv"))
DF = pd.DataFrame(Fplot[M1_Y_: M2_Y_,:]) 
DF.to_csv(os.path.join(outDir, "NPF_sample2_" + run_ID + ".csv"))
DF = pd.DataFrame(Fplot[M2_Y_: M3_Y_,:]) 
DF.to_csv(os.path.join(outDir, "NPF_sample3_" + run_ID + ".csv"))
DF = pd.DataFrame(Fplot[M3_Y_: M4_Y_,:]) 
DF.to_csv(os.path.join(outDir, "NPF_sample4_" + run_ID + ".csv"))





# %%





