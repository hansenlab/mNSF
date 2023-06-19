

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSFH_2023_06_14/mNSF-main'
dir_output='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/'

########################################################################
########################################################################
import sys
sys.path.append(dir_mNSF_functions)


import mNSF

from mNSF import process_multiSample

from mNSF.NSF import preprocess
from mNSF.NSF import misc
#from mNSF.NSF import visualize


from mNSF import training_multiSample_perSample
from scanpy import read_h5ad
#from tensorflow.data import Dataset
from os import path
#import pandas
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import sys 
import pickle


sys.path.append(dir_output)


########################################################################
########################################################################
L=3
nsample=2

dpth='data'

pth="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/"
mpth = path.join(pth,"models")
misc.mkdir_p(mpth)
pp = path.join(mpth,"pp")#list_fit[0].generate_pickle_path("constant",base=mpth)
misc.mkdir_p(pp)

########################################################################3
################### step 0  Data loading
########################################################################

list_D=list()
list_X=list()
for ksample in range(0,nsample):
	Y=pd.read_csv(path.join(dpth,'/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_sample'+ str(ksample+1) +'_smallData.csv'))
	X=pd.read_csv(path.join(dpth,'/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample'+ str(ksample+1) +'_smallData.csv'))
	D=process_multiSample.get_D(X,Y)
	list_D.append(D)
	list_X.append(X)


	
list_Dtrain=process_multiSample.get_listDtrain(list_D)
list_sampleID=process_multiSample.get_listSampleID(list_D)


########################################################################3
################### step 1 initialize model
########################################################################

list_fit = process_multiSample.ini_multiSample(list_D,L)
#indices=indices.astype(int)


########################################################################3
################### step 2 fit model
########################################################################
list_fit = training_multiSample_perSample.train_model_mNSF(list_fit,pp,
        		list_Dtrain,list_D,1)


# save the fitted model
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

Wdf=pd.DataFrame(W*inpf12["totals1"][:,None],  columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_spde_smallData.csv"))



## save the factors
inpf12 = process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join("factors_sample"+str(k+1)+"_smallData.csv"))













	

