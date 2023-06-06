

##################################################################
##################################################################


from anndata import AnnData
import sys
dpth="/dcs04/hansen/data/ywang/ST/mNSF_package/data/"

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/2023_05_08_mNSFH/mNSFH_May8_2023/functions'
sys.path.append(dir_mNSF_functions)

#import load_packages # load functions from mNSF and NSF, as well as the dependencies


##################################################################


##### load data

dpth="/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/"

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
Y=pd.read_csv(path.join(dpth,'Y_sample3.csv'))
X=pd.read_csv(path.join(dpth,'X_sample3.csv'))
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
L = 20
T=12
nsample_=2
J=500

random.seed(10)
# step2, initiate fit1, fit2 
D1["Z"]=D1['X'][random.sample(range(0, D1['X'].shape[0]-1), 300) ,:]
random.seed(10)

D2["Z"]=D2['X'][random.sample(range(0, D2['X'].shape[0]-1), 300) ,:]
random.seed(10)



D1["Z"]=D1['X']
D2["Z"]=D2['X']



#################################################
#### mNSF model initialization
#################################################


list_D__=[D1,D2]


#fit = pfh.ProcessFactorizationHybrid(N, J, L, D["Z"], T=T, psd_kernel=ker,
#                                       nonneg=True, lik="poi")
                                       
fit1=pfh.ProcessFactorizationHybrid(D1['Y'].shape[0],J,L,D1['Z'],T=T, psd_kernel=ker,nonneg=True,lik="poi")
fit2=pfh.ProcessFactorizationHybrid(D2['Y'].shape[0],J,L,D2['Z'],T=T, psd_kernel=ker,nonneg=True,lik="poi")



#fit.init_loadings(D["Y"],X=X,sz=D["sz"],shrinkage=0.3)
fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)






######################################
######################################
Z_c=np.concatenate((D1['Z'], D2['Z']), axis=0)
list_Z=[D1["Z"],D2["Z"]]

reload(pfh_multiSample)


fit12_=pfh_multiSample.ProcessFactorizationHybrid(Z_c.shape[0],J,L,
  Z=Z_c,
  list_Z=list_Z,
  T=T,
  #nsample=len(list_D__),
  psd_kernel=ker,nonneg=True,lik="poi")




fit12_.init_loadings(np.concatenate((D1['Y'], D2['Y']), axis=0),
  [D1['X'], D2['X']],
  #list_Z=[D1['Z'],D2['Z']],
  sz=np.concatenate((D1['sz'], D2['sz']), axis=0),shrinkage=0.3)


#fit12_.spat.beta0
#fit12_.spat.delta



### assign parameters to each samples' model 
M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]





# spatial parameters

delta1=fit12_.spat.delta.numpy()[:,0:M1_Z_]
beta01=fit12_.spat.beta0.numpy()[0:T,:]
beta1=fit12_.spat.beta.numpy()[0:T,:]
W=fit12_.spat.W.numpy()



delta2=fit12_.spat.delta.numpy()[:,M1_Z_:M2_Z_]
beta02=fit12_.spat.beta0.numpy()[T:(2*T),:]
beta2=fit12_.spat.beta.numpy()[T:(2*T),:]


fit1.spat.delta.assign(delta1) #LxM
fit1.spat.beta0.assign(beta01) #LxM
fit1.spat.beta.assign(beta1) #LxM
fit1.spat.W.assign(W) #LxM



fit2.spat.delta.assign(delta2) #LxM
fit2.spat.beta0.assign(beta02) #LxM
fit2.spat.beta.assign(beta2) #LxM
fit2.spat.W.assign(W) #LxM




# non-spatial parameters
L_nosp=L-T

fit12_.nsp.qloc.shape
fit12_.nsp.qscale.shape
fit12_.nsp.ploc.shape




qloc1=fit12_.nsp.qloc[0:M1_Z_,:]
qscale1=fit12_.nsp.qscale[0:M1_Z_,:]
V=fit12_.nsp.V #shared between samples
ploc=fit12_.nsp.ploc[0:L_nosp] #shared between samples

qloc2=fit12_.nsp.qloc[M1_Z_:M2_Z_,:]
ploc2=fit12_.nsp.ploc[0:L_nosp]
qscale2=fit12_.nsp.qscale[M1_Z_:M2_Z_,:]



fit1.nsp.qloc.assign(qloc1) #LxM
fit1.nsp.qscale.assign(qscale1) #LxM
fit1.nsp.V.assign(V) #LxM



fit2.nsp.qloc.assign(qloc2) #LxM
fit2.nsp.qscale.assign(qscale2) #LxM
fit2.nsp.V.assign(V) #LxM
fit2.nsp.ploc.assign(ploc) #LxM



D1["X"].shape
D2["X"].shape



# save the parameters
kkk=0
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit1)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
list_para_tmp["Kuu_chol"].shape

kkk=1
list_para_tmp=training_multiSample.store_paras_from_tf_to_np(fit2)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')



