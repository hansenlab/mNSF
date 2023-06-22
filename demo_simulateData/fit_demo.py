## Testing mNSF on simulated data

## Data from https://github.com/willtownes/nsf-paper/blob/main/simulations/bm_sp/data/S1.h5ad
## Data loading based on https://github.com/willtownes/nsf-paper/blob/main/demo.ipynb
## Fitting based on https://github.com/hansenlab/mNSF/blob/a8508703224cf0bdae3bc8e4dd261b1f5ebaa36b/mouse_sagittal_data_analysis_smallData/fit.py

#%%
####### Imports #################################################################
import sys
import mNSF
from mNSF import process_multiSample, training_multiSample
from mNSF.NSF import preprocess, misc, pf, visualize

from os import path
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import scanpy as sc
import sys 
import pickle

#%%
########################################################################
########################################################################
L=4
nsample=2

pth=""
mpth = path.join(pth,"models")
misc.mkdir_p(mpth)
pp = path.join(mpth,"pp")

misc.mkdir_p(pp)

legacy = True # Use legacy optimizer if tensorflow 2.12.0 +

epochs = 500

## Currently using exact same simulated data -- KW TODO: add noise
ad_filelist = ['data/S1.h5ad', 'data/S1.h5ad'] # from https://github.com/willtownes/nsf-paper/blob/main/simulations/bm_sp/data/S1.h5ad
ad_list = [sc.read_h5ad(ad)[:,:80] for ad in ad_filelist]
nsample = len(ad_list)

#%%
########################################################################3
################### step 0  Data loading
########################################################################

list_D=list()
list_X=list()
for ksample in range(0,nsample):
	ad = ad_list[ksample]
	X = ad.obsm["spatial"]
	D=process_multiSample.get_D_fromAnnData(ad)
	list_D.append(D)
	list_X.append(X)

	
list_Dtrain=process_multiSample.get_listDtrain(list_D)
list_sampleID=process_multiSample.get_listSampleID(list_D)


########################################################################3
################### step 1 initialize model
########################################################################

list_fit=process_multiSample.ini_multiSample(list_D,L)
#indices=indices.astype(int)



########################################################################3
################### step 2 fit model
########################################################################

list_fit=training_multiSample.train_model_mNSF(list_fit,pp,list_Dtrain,list_D, legacy=legacy, num_epochs=epochs)



process_multiSample.save_object(list_fit, 'list_fit_smallData.pkl') 

########################################################################3
################### step 3 save and plot results
########################################################################
## save the loadings
#loadings=visualize.get_loadings(list_fit[0])
#DF = pd.DataFrame(loadings) 
#DF.to_csv(("loadings.csv"))
inpf12=process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)


W = inpf12["loadings"]
#Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))

Wdf=pd.DataFrame(W*inpf12["totalsW"][:,None],  columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_spde_smallData.csv"))



## save the factors
inpf12 = process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join("","factors_sample"+str(k+1)+"_smallData.csv"))


#%%
## Heatmap of true values
hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}
for ksample in range(0,nsample):
	Ftrue = ad_list[ksample].obsm["spfac"]
	Xtr = list_X[ksample]
	fig,axes=visualize.multiheatmap(Xtr, Ftrue, (1,4), cmap="Blues", **hmkw)
	fig.suptitle("True data, sample " + str(ksample+1),fontsize=8)

## Heatmap of sampled data
print("True data")
hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}
for ksample in range(0,nsample):
	Yss = ad.layers["counts"][:,(4,0,1,2)]
	Xtr = list_X[ksample]
	fig,axes=visualize.multiheatmap(Xtr, Yss, (1,4), cmap="Blues", **hmkw)
	fig.suptitle("Sampled data, sample " + str(ksample+1),fontsize=8)

## plot the factors
hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}
for ksample in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	fig,axes=visualize.multiheatmap(list_D[ksample]["X"],Factors[indices,:], (1,L), cmap="Blues", **hmkw)
	fig.suptitle("Factors, sample " + str(ksample+1),fontsize=8)
	fig.savefig(path.join("","sample"+str(ksample+1)+"_smallData.png"),bbox_inches='tight')


# %%
