sample1 = 10

########################################################################
dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSF_2023_10_20/'
dir_output="/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_dist005/"

########################################################################
########################################################################
import sys
sys.path.append(dir_mNSF_functions)



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
L=3


nsample=2


mpth = path.join("models")
misc.mkdir_p(mpth)
pp = path.join(mpth,"pp",str(sample1))
misc.mkdir_p(pp)




########################################################################3
################### step 0  Data loading
########################################################################

list_D=list()
list_X=list()


for ksample in range(0,nsample):
	print(ksample)
	Y=pd.read_csv(path.join('/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_dist005/Y_kp_s'+ str(sample1) +'_s'+str(sample1+1)+'_filterDist005_s'+str(ksample+1)+'_corrected.csv'))
	X=pd.read_csv(path.join('/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_dist005/X_kp_s'+ str(sample1) +'_s'+str(sample1+1)+'_filterDist005_s'+str(ksample+1)+'_corrected.csv'))
	D=process_multiSample.get_D(X,Y)
	list_D.append(D)
	list_X.append(D["X"])


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


# save the fitted model
process_multiSample.save_object(list_fit, 'list_fit'+str(sample1)+"_"+str(sample1+1)+'_szmean.pkl') 


########################################################################
with open( 'list_fit'+str(sample1)+"_"+str(sample1+1)+'_szmean.pkl', 'rb') as inp:
              list_fit = pickle.load(inp)


########################################################################
################### step 3 save and plot results
########################################################################
inpf12=process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)


W = inpf12["loadings"]
#Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))

Wdf=pd.DataFrame(W*inpf12["totalsW"][:,None],  columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_spde_s"+str(sample1)+"_"+str(sample1+1)+"_szmean.csv"))



## save the factors
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join("factors_sample"+str(sample1)+"_"+str(sample1+1)+"_s"+str(k+1)+"_szmean.csv"))


#



	

