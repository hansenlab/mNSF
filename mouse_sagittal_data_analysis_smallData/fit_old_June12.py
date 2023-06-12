import sys
sys.path.append('..')
import mNSF

from mNSF.NSF import preprocess

from mNSF.NSF import training
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


#interpret_npf_v3



########################################################################
L=6
nsample=2

dpth='data'


########################################################################3
################### step 0  Data loading
########################################################################



list_D=list()
list_Dtrain=list()
list_X=list()
list_Z=list()
list_sampleID=list()

index_=0

for kkk in range(0,nsample):
	ad = read_h5ad(path.join(dpth, 'data_s'+ str(nsample+1) +'.h5ad'))
	D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,flip_yaxis=False)
	D["Z"]=D['X']
	list_D.append(D)
	list_X.append(D['X'])
	list_Z.append(D['Z'])
	list_sampleID.append(D['Z'])
	Ntr = D["Y"].shape[0]
	Dtrain = Dataset.from_tensor_slices(D)
	D_train = Dtrain.batch(round(Ntr)+1)
	list_Dtrain.append(D_train)
	list_sampleID.append([index_:Ntr])
	index_=index_+Ntr
                                    



J=list_D[0]["Y"].shape[1] # number of geens



########################################################################3
################### step 1 initialize model
########################################################################

list_fit = ini_multiSample(list_D,L=L)






list_fit=list()

for k in range(0,nsample):
	D=list_D[k]
	fit=pf.ProcessFactorization(J,L,D['Z'],psd_kernel=ker,nonneg=True,lik="poi")
	fit.init_loadings(D["Y"],X=D['X'],sz=D["sz"],shrinkage=0.3)
	list_fit.append(fit)]
	if k==0:
		X_concatenated=D['X']
		Z_concatenated=D['Z']
		Y_concatenated=D['Y']
		sz_concatenated=D['sz']
	else:
		X_concatenated=np.concatenate((X_concatenated, D['X']), axis=0)
		Z_concatenated=np.concatenate((Z_concatenated, D['Z']), axis=0)
		Y_concatenated=np.concatenate((Y_concatenated, D['Y']), axis=0)
		sz_concatenated=np.concatenate((sz_concatenated, D['sz']), axis=0)


fit12_=pf_multiSample.ProcessFactorization_fit12(J,L,
  Z_concatenated,
  nsample=nsample,
  psd_kernel=ker,nonneg=True,lik="poi")


fit12_.init_loadings(Y_concatenated,
  list_X=list_X,
  list_Z=list_Z,
  sz=sz_concatenated), axis=0),shrinkage=0.3)



for k in range(0,nsample):
	indices=list_sampleID[k]
	delta=fit12_.delta.numpy()[:,indices]
	beta0=fit12_.beta0.numpy()[((k-1)*L):(k*L),:]
	beta=fit12_.beta.numpy()[((k-1)*L):(k*L),:]
	W=fit12_.W.numpy()
	list_fit[k].delta.assign(delta) 
	list_fit[k].beta0.assign(beta0)
	list_fit[k].beta.assign(beta) 
	list_fit[k].W.assign(W) 
	list_para_tmp=training_multiSample.store_paras_from_tf_to_np(list_fit[k])
	postprprocess_multiSample.save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')






########################################################################3
################### step 2 fit model
########################################################################
D12=list_D__[0]


fit12=pf.ProcessFactorization(J,L,D12['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit12.init_loadings(D12["Y"],X=D12['X'],sz=D12["sz"],shrinkage=0.3)


mpth = path.join(pth,"_models")

pp = fit12.generate_pickle_path("constant",base=mpth)
misc.mkdir_p(plt_pth) 
misc.mkdir_p(mpth)
misc.mkdir_p(pp)



tro_ = training_multiSample.ModelTrainer(fit12,pickle_path=pp)


tro_1 = training.ModelTrainer(fit12,pickle_path=pp)



for kiter in range(0,50):
	fit =tro_.train_model([tro_1],
        	list_Dtrain,list_D)



########################################################################3
################### step 3 save and plot results
########################################################################


# save the fitted model
for k in range(0,nsample):
            with open('list_para_'+ str(k+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            training_multiSample.assign_paras_from_np_to_tf(list_fit[k],list_para_tmp)
            list_fit[k].Z=list_D__[k]["Z"]
            list_fit[k].sample_latent_GP_funcs(list_D[kkk]['X'],S=100,chol=True)


postprprocess_multiSample.save_object(list_D, 'list_fit.pkl')


## save the loadings
loadings=visualize.get_loadings(list_fit__[0])
#DF = pd.DataFrame(loadings) 
#DF.to_csv(("loadings.csv"))

W = inpf12["loadings"]
Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_spde.csv"))



## save the factors
inpf12 = postprprocess_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Fplot = inpf12["factors"][:,0:L]

for k in range(0,nsample):
	indices=list_sampleID[k]
	DF = pd.DataFrame(Fplot[indices,:]) 
	DF.to_csv(path.join("factors_sample"+str(k+1)+".csv"))

	









