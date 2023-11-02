

########################################################################
dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSF_2023_10_20/'
#dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSF_2023_10_20/mNSF-main'
dir_output="/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_keepAll_scTransform/"

########################################################################
########################################################################
import sys
sys.path.append(dir_mNSF_functions)

#from scanpy import read_h5ad

import random
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

from scanpy import pp

sys.path.append(dir_output)
os.chdir(dir_output)



########################################################################
########################################################################
L=10


nsample=12

mpth = path.join("models")
misc.mkdir_p(mpth)
pp = path.join(mpth,"pp",str(2))#list_fit[0].generate_pickle_path("constant",base=mpth)
misc.mkdir_p(pp)


########################################################################3
################### step 0  Data loading
########################################################################

list_D=list()
list_X=list()

for ksample in range(0,nsample):
	Y=pd.read_csv(path.join('//dcs04/hansen/data/ywang/ST/DLPFC/processed_Data//Y_features_sele_sample'+str(ksample+1)+'_500genes.csv'))
	X=pd.read_csv(path.join('//dcs04/hansen/data/ywang/ST/DLPFC/processed_Data///X_allSpots_sample'+str(ksample+1)+'.csv'))
	D=process_multiSample.get_D(X,Y)
	list_D.append(D)
	list_X.append(D["X"])
	

list_Dtrain=process_multiSample.get_listDtrain(list_D)
list_sampleID=process_multiSample.get_listSampleID(list_D)


# inducing points, 70% of total spots for each sample
for ksample in range(0,nsample):
	random.seed(111)
	ninduced=round(list_D[ksample]['X'].shape[0] * 0.35)
	random.seed(222)
	print(ninduced)
	D=list_D[ksample]
	rd_ = random.sample(range(0, D['X'].shape[0]), ninduced)
	list_D[ksample]["Z"]=D['X'][rd_ ,:]





########################################################################3
################### step 1 initialize model
########################################################################
#lik="nb"

list_fit=process_multiSample.ini_multiSample(list_D,L,"nb")


########################################################################
################### step 2 fit model
########################################################################


list_fit=training_multiSample.train_model_mNSF(list_fit,pp,list_Dtrain,list_D)



# save the fitted model
process_multiSample.save_object(list_fit, 'list_fit_nb_12samples_szMean_L10_fullData.pkl') 



########################################################################
with open( 'list_fit_nb_12samples_szMean_L10_fullData.pkl', 'rb') as inp:
              list_fit = pickle.load(inp)

    
              
########################################################################
################### step 3 save and plot results
########################################################################
inpf12=process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)


W = inpf12["loadings"]
#Wdf=pd.DataFrame(W*inpf12["totals1"





Wdf=pd.DataFrame(W*inpf12["totalsW"][:,None],  columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_spde_nb_szMean_12samples_L10_fullData.csv"))



## save the factors
#inpf12 = process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Factors = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	indices=indices.astype(int)
	Factors_df = pd.DataFrame(Factors[indices,:]) 
	Factors_df.to_csv(path.join(dir_output,"factors_nb_szMean_sample_s"+str(k+1)+"_L10_fullData.csv"))


#



	

