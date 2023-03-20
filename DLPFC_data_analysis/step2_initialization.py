

##################################################################
##################################################################

import load_packages # load functions from mNSF and NSF, as well as the dependencies



##################################################################
##################################################################
#%% Data loading, sample 1
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data/'
ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D1=D


#%% Data loading, sample 1
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data/'
ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

D1,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)


#%% Data loading, sample 2
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample2/simulations/ggblocks_lr/data/'
ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

D2,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)


#%% Data loading, sample 3
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample3/simulations/ggblocks_lr/data/'
ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

D3,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)


#%% Data loading, sample 4
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample1/simulations/ggblocks_lr/data')
dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample4/simulations/ggblocks_lr/data/'
ad = read_h5ad(path.join(dpth,"ggblocks_lr_ori.h5ad"))

D4,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)



################################################################################################################
##############
L = 7
nsample_=4


D1["Z"]=D1['X']
D2["Z"]=D2['X']
D3["Z"]=D3['X']
D4["Z"]=D4['X']



########################################################################################################################
########################################################################################################################

###################
#### mNSF model initialization
###################


list_D__=[D1,D2,D3,D4]


fit1=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf_ori_mod.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf_ori_mod.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit4=pf_ori_mod.ProcessFactorization(J,L,D4['Z'],psd_kernel=ker,nonneg=True,lik="poi")



fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)
fit4.init_loadings(D4["Y"],X=D4['X'],sz=D4["sz"],shrinkage=0.3)



######################################3
fit12_=pf_fit12_fullSample_2_3.ProcessFactorization_fit12(J,L,
  np.concatenate((D1['Z'], D2['Z'],D3['Z'],D4['Z']), axis=0),
  nsample=len(list_D__),
  psd_kernel=ker,nonneg=True,lik="poi")


fit12_.init_loadings(np.concatenate((D1['Y'], D2['Y'],D3['Y'],D4['Y']), axis=0),
  list_X=[D1['X'], D2['X'],D3['X'],D4['X']],
  list_Z=[D1['Z'],D2['Z'],D3['Z'],D4['Z']],
  sz=np.concatenate((D1['sz'], D2['sz'], D3['sz'],D4['sz']), axis=0),shrinkage=0.3)



M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]
M3_Z_=M2_Z_+D3["Z"].shape[0]
M4_Z_=M3_Z_+D4["Z"].shape[0]


delta1=fit12_.delta.numpy()[:,0:M1_Z_]
beta01=fit12_.beta0.numpy()[0:L,:]
beta1=fit12_.beta.numpy()[0:L,:]
W=fit12_.W.numpy()


delta2=fit12_.delta.numpy()[:,M1_Z_:M2_Z_]
beta02=fit12_.beta0.numpy()[L:(2*L),:]
beta2=fit12_.beta.numpy()[L:(2*L),:]

delta3=fit12_.delta.numpy()[:,M2_Z_:M3_Z_]
beta03=fit12_.beta0.numpy()[(2*L):(3*L),:]
beta3=fit12_.beta.numpy()[(2*L):(3*L),:]

delta4=fit12_.delta.numpy()[:,M3_Z_:M4_Z_]
beta04=fit12_.beta0.numpy()[(3*L):(4*L),:]
beta4=fit12_.beta.numpy()[(3*L):(4*L),:]


fit1.delta.assign(delta1) #LxM
fit1.beta0.assign(beta01) #LxM
fit1.beta.assign(beta1) #LxM
fit1.W.assign(W) #LxM



fit2.delta.assign(delta2) #LxM
fit2.beta0.assign(beta02) #LxM
fit2.beta.assign(beta2) #LxM
fit2.W.assign(W) #LxM

fit3.delta.assign(delta3) #LxM
fit3.beta0.assign(beta03) #LxM
fit3.beta.assign(beta3) #LxM
fit3.W.assign(W) #LxM

fit4.delta.assign(delta4) #LxM
fit4.beta0.assign(beta04) #LxM
fit4.beta.assign(beta4) #LxM
fit4.W.assign(W) #LxM


# save the parameters


kkk=0
list_para_tmp=training.store_paras_from_tf_to_np(fit1)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
list_para_tmp["Kuu_chol"].shape

kkk=1
list_para_tmp=training.store_paras_from_tf_to_np(fit2)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

kkk=2
list_para_tmp=training.store_paras_from_tf_to_np(fit3)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')

kkk=3
list_para_tmp=training.store_paras_from_tf_to_np(fit4)
save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')


