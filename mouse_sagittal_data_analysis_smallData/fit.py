import sys
sys.path.append('..')
import mNSF

from mNSF.NSF import preprocess

from mNSF.NSF import visualize
from mNSF import training_multiSample


from scanpy import read_h5ad
from tensorflow.data import Dataset
from os import path
#import pandas
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import sys 
import pickle



########################################################################
L=6
nsample=2

dpth='data'

mpth = path.join(pth,"_models")
pp = list_fit[0].generate_pickle_path("constant",base=mpth)
misc.mkdir_p(mpth)
misc.mkdir_p(pp)

########################################################################3
################### step 0  Data loading
########################################################################



list_D=list()

for kkk in range(0,nsample):
	Y=pd.read_csv(path.join(dpth,'Y_sample'+ str(k+1) +'.csv'))
	X=pd.read_csv(path.join(dpth,'X_sample'+ str(k+1) +'.csv')
	D=process_multiSample.get_D(X,Y)
	list_D.append(D)


	
list_Dtrain=process_multiSample.get_listDtrain(list_D)
list_sampleID=process_multiSample.get_listSampleID(list_D)


########################################################################3
################### step 1 initialize model
########################################################################

list_fit = process_multiSample.ini_multiSample(list_D,L)


########################################################################3
################### step 2 fit model
########################################################################
list_fit = training_multiSample.train_model_mNSF(list_fit,pickle_path=pp,
        		list_Dtrain,list_D,niter_=50)


# save the fitted model
postprprocess_multiSample.save_object(list_fit, 'list_fit.pkl') 

########################################################################3
################### step 3 save and plot results
########################################################################
## save the loadings
#loadings=visualize.get_loadings(list_fit[0])
#DF = pd.DataFrame(loadings) 
#DF.to_csv(("loadings.csv"))

W = inpf12["loadings"]
Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_spde.csv"))


## save the factors
inpf12 = postprprocess_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join("factors_sample"+str(k+1)+".csv"))

	

