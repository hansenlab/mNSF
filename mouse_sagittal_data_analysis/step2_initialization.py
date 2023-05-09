




dir_mNSF_functions='/dcs04/hansen/data/ywang/ST/mNSF_package/functions/'
sys.path.append(dir_mNSF_functions)
import load_packages



###############################################################################################################

###############################################################################################################


############## Data loading

dpth='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/'

## sample 1
#ad = read_h5ad((dpth,"data_s1.h5ad"))
Y=pd.read_csv(path.join(dpth,'Y_sample1.csv'))
X=pd.read_csv(path.join(dpth,'X_sample1.csv'))
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D1=D


## sample 2
#ad = read_h5ad((dpth,"data_s2.h5ad"))
## sample 2
Y=pd.read_csv(path.join(dpth,'Y_sample2.csv'))
X=pd.read_csv(path.join(dpth,'X_sample2.csv'))
X = preprocess.rescale_spatial_coords(X)
X=X.to_numpy()

ad = AnnData(Y,obsm={"spatial":X})

ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X

pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
pp.log1p(ad)

J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
D2=D





################################################################################################################
################################################################################################################
################################################################################################################
##############
# %% Initialize inducing poi_sz-constantnts


##################################################################
##################################################################

##################################################################
##################################################################
################### split into batches
##################################################################
##################################################################


nsample_=2




# step1, create D12
Ntr = D1["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D1)
D1_train = Dtrain.batch(round(Ntr)+1)
Ntr
Ntr = D2["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D2)
D2_train = Dtrain.batch(round(Ntr)+1)
Ntr
Ntr = D3["Y"].shape[0]


D1["Z"]=D1['X']
D2["Z"]=D2['X']




###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
#cwd = os.getcwd()


###################
###################
###################

L=8
J=500

list_D__=[D1,D2]



fit1=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")


#fit1.init_loadings(D["Y"],X=X,sz=D["sz"],shrinkage=0.3)
#fit1.init_loadings(D1["Y"],X=X,sz=None,shrinkage=0.3)
fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)




##could remove omega in this step
fit12_=pf_multiSample.ProcessFactorization_fit12(J,L,
  np.concatenate((D1['Z'], D2['Z']), axis=0),
  nsample=nsample_,
  psd_kernel=ker,nonneg=True,lik="poi")


fit12_.init_loadings(np.concatenate((D1['Y'], D2['Y']), axis=0),
  list_X=[D1['X'], D2['X']],
  list_Z=[D1['Z'],D2['Z']],
  sz=np.concatenate((D1['sz'], D2['sz']), axis=0),shrinkage=0.3)
#save_object(fit12_, 'fit12___npf_fulldata_nsamele_memorySaving_qsub_fullData.pkl')





M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]



delta1=fit12_.delta.numpy()[:,0:M1_Z_]
beta01=fit12_.beta0.numpy()[0:L,:]
beta1=fit12_.beta.numpy()[0:L,:]
W=fit12_.W.numpy()


delta2=fit12_.delta.numpy()[:,M1_Z_:M2_Z_]
beta02=fit12_.beta0.numpy()[L:(2*L),:]
beta2=fit12_.beta.numpy()[L:(2*L),:]




import tensorflow as tf
fit1.delta.assign(delta1) #LxM
fit1.beta0.assign(beta01) #LxM
fit1.beta.assign(beta1) #LxM
fit1.W.assign(W) #LxM

import pandas as pd

fit2.delta.assign(delta2) #LxM
fit2.beta0.assign(beta02) #LxM
fit2.beta.assign(beta2) #LxM
fit2.W.assign(W) #LxM




kkk=0
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit1)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
list_para_tmp["Kuu_chol"].shape

kkk=1
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit2)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')











