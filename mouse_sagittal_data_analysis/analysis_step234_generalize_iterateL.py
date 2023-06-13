## single analysis script for Yi's steps 2-4
# Attempt at iterate through num patterns (L)
# NOTE - errors at line 176 fit = tro_.train_model because of shape mismatch between list_self[0] and list_para_tmp

# %%
# KW move all imports to top
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
from anndata import AnnData
import scanpy as sc
from scanpy import pp
from scanpy import read_h5ad
import time
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
#L = 5 # num patterns
L_list = [2,3,4,5,6,7,8,9,10,11,12]
J = 500
nIter = 50
nbatch_eachDim = 1 # KW - Yi defined this but never used?

## Load in samples - create list of h5ad files and sample names
# Load in samples
D1 = read_h5ad(os.path.join(dpth,"data_s1.h5ad"))
D2 = read_h5ad(os.path.join(dpth,"data_s2.h5ad"))
D3 = read_h5ad(os.path.join(dpth,"data_s3.h5ad"))
D4 = read_h5ad(os.path.join(dpth,"data_s4.h5ad"))

# create list of AnnData objects and corresponding names (for saving)
samples = [D1, D2] # include all samples for analysis in list
sample_names = ['D1', 'D2']
nsample = len(samples)
saveRes = False

run_ID = "KW_mouseSagittal_2samples_iterateL"

outDir = run_ID + "_out" # make output directory using the run ID
misc.mkdir_p(outDir)

mpth = path.join(outDir,"_models_mNSF_") # in Yi code but nothing saved here
misc.mkdir_p(mpth)

output_memory = pd.DataFrame(np.zeros((len(L_list), 5)))
output_memory.columns = ["L", "peak_before_first_iteration", "peak_after_first_iteration", "peak_after_all_iterations", "runtime"]
output_memory_file = "memoryUsage_ " + run_ID + ".csv"


###### START STEP 2 ######
#### Preprocess every sample
# KW already saved annData objects in step1 -- no need to recreate
for i in range(len(L_list)):
    L = L_list[i]
    print(L)

    list_D__ = []
    list_Ds_train_batches_  = []
    fits = []

    for s in samples:
        J = s.shape[1]
        D, _ = preprocess.anndata_to_train_val(s, layer="counts", train_frac=1.0,
                                        flip_yaxis=False)
        D_n,_ = preprocess.anndata_to_train_val(s,train_frac=1.0,flip_yaxis=False)
        fmeans,D_c,_ = preprocess.center_data(D_n)
        X = D["X"] #note this should be identical to Dtr_n["X"]
        N = X.shape[0]
        M = N # KW number of inducing points equal to all points
        
        D["Z"]=D['X']
        list_D__.append(D)

        Ntr = D["Y"].shape[0]
        Dtrain = Dataset.from_tensor_slices(D)
        D_train = Dtrain.batch(round(Ntr)+1)
        list_Ds_train_batches_.append(D_train)

        fit=pf.ProcessFactorization(J,L,D['Z'],psd_kernel=ker,nonneg=True,lik="poi")
        fit.init_loadings(D["Y"],X=D['X'],sz=D["sz"],shrinkage=0.3)
        fits.append(fit)

    sample_Xs = [D['X'] for D in list_D__]
    sample_Ys = [D['Y'] for D in list_D__]
    sample_Zs = [D['Z'] for D in list_D__]
    sample_szs = [D['sz'] for D in list_D__]






   
    print("fitting with multisample")

    fit12_=pf_multiSample.ProcessFactorization_fit12(J,L,
        np.concatenate(sample_Zs, axis=0),
        nsample=nsample,
        psd_kernel=ker,nonneg=True,lik="poi")

    fit12_.init_loadings(np.concatenate(sample_Ys, axis=0),
        list_X=sample_Xs,
        list_Z=sample_Zs,
        sz=np.concatenate(sample_szs, axis=0),shrinkage=0.3)

# KW parse results and save to individual fits (why did we fit individually if overwriting here? are there other parameters learned?)
    nSpots_perSample = [X.shape[0] for X in sample_Xs]
    W=fit12_.W.numpy()

    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    for i in range(nsample):
        print(i)
        delta = fit12_.delta.numpy()[:,sum(nSpots_perSample[:i]):sum(nSpots_perSample[:i+1])]
        beta0 = fit12_.beta0.numpy()[L*(i):L*(i+1),:]
        beta = fit12_.beta.numpy()[L*(i):L*(i+1),:]

        fit_i = fits[i]
        fit_i.delta.assign(delta)
        fit_i.beta0.assign(beta0) #LxM
        fit_i.beta.assign(beta) #LxM
        fit_i.W.assign(W) #LxM
        fits[i] = fit_i

        list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit_i)
        save_object(list_para_tmp, 'list_para_'+ str(nsample+1) +'.pkl')
        list_para_tmp["Kuu_chol"].shape



    ###### START STEP 3 #########
    # process factorization
    fit_ini=pf.ProcessFactorization(J,L,list_D__[0]['Z'],psd_kernel=ker,nonneg=True,lik="poi")
    fit_ini.init_loadings(list_D__[0]["Y"],X=list_D__[0]['X'],sz=list_D__[0]["sz"],shrinkage=0.3)

    picklePath = fit_ini.generate_pickle_path("constant",base=mpth) # renaming since we import pp package 
    misc.mkdir_p(picklePath)

    tro_ = training_multiSample_perSample.ModelTrainer(fit_ini,pickle_path=None)
    tro_ini = training.ModelTrainer(fit_ini,pickle_path=None)


    output_memory.iloc[i, 0] = L
    start = time.time()
    output_memory.iloc[i, 1] = tf.config.experimental.get_memory_info("GPU:0")['peak']/(1024*1024*1024)

    for iter in range(0,nIter):
        for ksample in range(0,nsample):
            fit =tro_.train_model([tro_ini],ksample,list_Ds_train_batches_,list_D__)## 1 iteration of parameter updatings
        if iter == 0:
            print('after first iteration' , tf.config.experimental.get_memory_info("GPU:0"))
            output_memory.iloc[i, 2] = tf.config.experimental.get_memory_info("GPU:0")['peak']/(1024*1024*1024)

    print('done fitting', tf.config.experimental.get_memory_info("GPU:0"))
    output_memory.iloc[i, 3] = tf.config.experimental.get_memory_info("GPU:0")['peak']/(1024*1024*1024)


    end = time.time()
    output_memory.iloc[i, 4] = end-start

    print(end-start)

    ######## START STEP 4 #########
    ## dont need to save outputs for memory check
    # KW - why are we resaving each one? just so we have saved output?
    #for i in range(0,nsample):
    #    with open('list_para_'+ str(i+1) +'.pkl', 'rb') as inp:
    #        list_para_tmp = pickle.load(inp)
    #        save_object(list_para_tmp, os.path.join(outDir,'list_para_' + str(i+1) +'_' + run_ID + '.pkl'))# 50 iterations


    # KW - do we need to fit each one individually again?? I think we just overwrite them anyway

    #list_fit__=[fit1,fit2,fit3,fit4] this is fits

    #for i in range(0,nsample):
    #        with open(os.path.join(outDir, 'list_para_' + str(i+1) +'_' + run_ID + '.pkl'), 'rb') as inp:
    #            list_para_tmp = pickle.load(inp)
    #            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
    #            training_multiSample.assign_paras_from_np_to_tf(fits[i],list_para_tmp) #KW note: assign pickled parameters to each in list_fit
    #            fits[i].Z=list_D__[i]["Z"]
    #            fits[i].sample_latent_GP_funcs(list_D__[i]['X'],S=100,chol=True)

    ##### KW - ASK YI, WHAT IS THIS FOR? saves object and then immediately reloads it? 
    #save_object(fits, os.path.join(outDir, 'list_fit' + run_ID + '.pkl'))

    #with open('fit_threeSampleNSF_list_fit_L20_mouse_Sagittal_addmarkergenes_olfIncluded.pkl', 'rb') as inp:
    #	list_fit__ = pickle.load(inp)
        
        
    #for i in range(0,nsample):
    #    fit_tmp=fits[i]
    #    D_tmp=list_D__[i]
    #    Mu_tr,Mu_val = fit_tmp.predict(D_tmp,Dval=None,S=10)
    #    Mu_tr_df = pd.DataFrame(Mu_tr) 
    #    #misc.poisson_deviance(D_tmp["Y"],Mu_tr)
    #    Mu_tr_df.to_csv(os.path.join(outDir,"Mu_tr_df_sample_"+str(i+1)+"_" + run_ID + ".csv"))


######################################################### factors

  
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
    #inpf12 = interpret_npf_v3(fits,sample_Xs,S=100,lda_mode=False)
    #Fplot = inpf12["factors"][:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
    #Fplot = inpf12["factors"][:, np.arange(L)]


    # save the loading of each gene into csv filr
    #loadings=visualize.get_loadings(fits[0]) # KW check all have same loadings?
    #loadingsDF = pd.DataFrame(loadings) 
    # save the dataframe as a csv file
    #loadingsDF.to_csv(os.path.join(outDir, "loadings_NPF_" + run_ID + ".csv"))


    #W = inpf12["loadings"]#*inpf["totals"][:,None]
    #Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1)) 
    #Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=None, columns=range(1,L+1)) #KW find way to retain index for saving
    #Wdf.to_csv(os.path.join(outDir, "loadings_NPF_spde" + run_ID + ".csv")) # KW question what is this


    #for i in range(nsample):
    #    DF = pd.DataFrame(Fplot[sum(nSpots_perSample[:i]): sum(nSpots_perSample[:i+1]),:]) 
    #    DF.to_csv(os.path.join(outDir, "NPF_sample" + str(i+1) + "_" + run_ID + ".csv"))

output_memory.to_csv(output_memory_file, index=False)







