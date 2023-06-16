

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSFH_2023_06_14/mNSF-main'
dir_output='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/'

########################################################################
########################################################################
import sys
sys.path.append(dir_mNSF_functions)

#from scanpy import read_h5ad


import mNSF

from mNSF import process_multiSample

from mNSF.NSF import preprocess
from mNSF.NSF import misc
#from mNSF.NSF import visualize
#from mNSF import training_multiSample
from mNSF import training_multiSample
from mNSF import process_multiSample
from mNSF.NSF import visualize

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
os.chdir(dir_output)



########################################################################
########################################################################
L=12

## nsample = 2:
#worked on L12

##nsample = 4:
#worked on L12



nsample=4

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
	Y=pd.read_csv(path.join(dpth,'/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_sample'+ str(ksample+1) +'.csv'))
	X=pd.read_csv(path.join(dpth,'/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample'+ str(ksample+1) +'.csv'))
	D=process_multiSample.get_D(X,Y)
	list_D.append(D)
	list_X.append(X)

list_D=[list_D[3],list_D[2],list_D[1],list_D[0]]
list_X=[list_X[3],list_X[2],list_X[1],list_X[0]]

	
list_Dtrain=process_multiSample.get_listDtrain(list_D)
list_sampleID=process_multiSample.get_listSampleID(list_D)




########################################################################3
################### step 1 initialize model
########################################################################

list_fit = process_multiSample.ini_multiSample(list_D,L)


########################################################################
################### step 2 fit model
########################################################################


list_fit=training_multiSample.train_model_mNSF(list_fit,pp,list_Dtrain,list_D)

gc.collect()


#during traning:
#print(tf.config.experimental.get_memory_info('GPU:0'))
#{'current': 4684887040, 'peak': 29394932480}


# save the fitted model
process_multiSample.save_object(list_fit, 'list_fit.pkl') 




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
Wdf.to_csv(path.join("loadings_spde.csv"))



## save the factors
#inpf12 = process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join("factors_sample"+str(k+1)+".csv"))


#



## plot the factors
hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}
for ksample in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	fig,axes=visualize.multiheatmap(list_D[ksample]["X"],Factors[indices,:], (1,L), cmap="Blues", **hmkw)
	fig.savefig(path.join(dir_output,"sample"+str(ksample+1)+".png"),bbox_inches='tight')






	

