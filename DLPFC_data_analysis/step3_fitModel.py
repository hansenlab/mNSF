

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


L = 7 # number of spatial factors
nsample_=4



D1["Z"]=D1['X']
D2["Z"]=D2['X']
D3["Z"]=D3['X']
D4["Z"]=D4['X']


Ntr = D1["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D1)
D1_train = Dtrain.batch(round(Ntr)+1)

Ntr = D2["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D2)
D2_train = Dtrain.batch(round(Ntr)+1)

Ntr = D3["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D3)
D3_train = Dtrain.batch(round(Ntr)+1)

Ntr = D4["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D4)
D4_train = Dtrain.batch(round(Ntr)+1)
   
   
list_D__=[D1,D2,D3,D4]
list_Ds_train_batches_=[D1_train,D2_train,D3_train,D4_train]


########### train models
dir_output_modelFitting='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches'
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')

#fit12=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit_ini=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit_ini.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)

## train mNSF 
mpth = path.join(dir_output_modelFitting,"mnsf_")
pp = fit_ini.generate_pickle_path("constant",base=mpth)

tro_ = training_multiSample.ModelTrainer(fit_ini,pickle_path=pp)

tro_ini = training.ModelTrainer(fit_ini,pickle_path=pp)
del fit1

for iter in range(0,50):
  fit =tro_.train_model([tro_ini],
        list_Ds_train_batches_[0:4],list_D__[0:4])## 1 iteration of parameter updatings











