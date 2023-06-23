### KW edit of Yi fit.py, to benchmark memory usage
# iterate through various N
# Note: in training_multiSample.py line 271 in _train_model_fixed_lr, set num_epochs = 10 for benchmarking

#%%
########################################################################
########################################################################
import sys
import mNSF
from mNSF.NSF import preprocess, misc
from mNSF import process_multiSample, training_multiSample
from scanpy import read_h5ad
import os
from os import path
import numpy as np
import tensorflow as tf
import pandas as pd
import sys 
import pickle
import time
from csv import writer


########################################################################
########################################################################
L=20
nsample=7
epochs = 500
legacy = True # Use legacy optimizer if tensorflow 2.12.0 +


possibleSamples = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
dpth='data'
pth="iterateN_" + str(nsample)
misc.mkdir_p(pth)
mpth = path.join(pth,"models")
misc.mkdir_p(mpth)
pp = path.join(mpth,"pp")#list_fit[0].generate_pickle_path("constant",base=mpth)
misc.mkdir_p(pp)

output_memory_list = []
output_memory_columns = ["nSample", "peak_before_first_iteration", "peak_after_10_iterations", "runtime"]

memory_outfile = "memoryUsage_mouseSagittal_L20_iterNsample_runSeperately.csv"

if not os.path.isfile(memory_outfile):
    with open(memory_outfile, 'a') as file:
        writer_object = writer(file)
        writer_object.writerow(output_memory_columns)
        file.close()

      


########################################################################3
################### step 0  Data loading
########################################################################

list_D=list()
list_X=list()
for ksample in possibleSamples[:nsample]:
	Y=pd.read_csv(path.join(dpth,'Y_sample'+ str(ksample+1) +'_smallData.csv'))
	X=pd.read_csv(path.join(dpth,'X_sample'+ str(ksample+1) +'_smallData.csv'))
	D=process_multiSample.get_D(X,Y)
	list_D.append(D)
	list_X.append(X)


	
list_Dtrain=process_multiSample.get_listDtrain(list_D)
list_sampleID=process_multiSample.get_listSampleID(list_D)


########################################################################3
################### step 1 initialize model
########################################################################

list_fit = process_multiSample.ini_multiSample(list_D,L)

# measuring here - would need to build into train_model_mNSF step
# can set max iteration to 10 and then measure
########################################################################3
################### step 2 fit model
########################################################################

output_memory_list.append(nsample)

start = time.time()
output_memory_list.append(tf.config.experimental.get_memory_info("GPU:0")['peak']/(1024*1024*1024))


list_fit=training_multiSample.train_model_mNSF(list_fit,pp,list_Dtrain,list_D, legacy=legacy, num_epochs=epochs)


output_memory_list.append(tf.config.experimental.get_memory_info("GPU:0")['peak']/(1024*1024*1024))
end = time.time()
output_memory_list.append(end-start)

with open(memory_outfile, 'a') as file:
    writer_object = writer(file)
    writer_object.writerow(output_memory_list)
    file.close()


# save the fitted model
process_multiSample.save_object(list_fit, os.path.join(pth,'list_fit_smallData.pkl'))

########################################################################3
################### step 3 save and plot results
########################################################################

inpf12=process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)

## save the loadings
W = inpf12["loadings"]
Wdf=pd.DataFrame(W*inpf12["totalsW"][:,None],  columns=range(1,L+1))
Wdf.to_csv(path.join(pth, "loadings_spde_smallData.csv"))


## save the factors
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join(pth, "factors_sample"+str(k+1)+"_smallData.csv"))













	


# %%
