

##################################################################
##################################################################
import sys

dir_mNSF_functions='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/mNSF_package/functions'
sys.path.append(dir_mNSF_functions)
import load_packages # load functions from mNSF and NSF, as well as the dependencies


##################################################################
##################################################################

#%% Data loading
dpth="/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/mNSF_package/data/"

ad = read_h5ad(path.join(dpth,"ggblocks_lr_1.h5ad"))

D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D1=D


ad = read_h5ad(path.join(dpth,"ggblocks_lr_5.h5ad"))

D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D2=D


ad = read_h5ad(path.join(dpth,"ggblocks_lr_9.h5ad"))

D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D3=D






################################################################################################################


L = 7 # number of spatial factors

nsample_=3
J=500






Ntr = D1["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D1)
D1_train = Dtrain.batch(round(Ntr)+1)

Ntr = D2["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D2)
D2_train = Dtrain.batch(round(Ntr)+1)

Ntr = D3["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D3)
D3_train = Dtrain.batch(round(Ntr)+1)


random.seed(10)
# step2, initiate fit1, fit2 
D1["Z"]=D1['X'][random.sample(range(0, D1['X'].shape[0]-1), 1000) ,:]
random.seed(10)

D2["Z"]=D2['X'][random.sample(range(0, D2['X'].shape[0]-1), 1000) ,:]
random.seed(10)

D3["Z"]=D3['X'][random.sample(range(0, D3['X'].shape[0]-1), 1000) ,:]
random.seed(10)


list_D__=[D1,D2,D3]
list_Ds_train_batches_=[D1_train,D2_train,D3_train]


########### train models
dir_output_modelFitting='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/mNSF_package/model_fitting/'
sys.path.append(dir_output_modelFitting)

#fit12=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit_ini=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit_ini.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)


## train mNSF 
mpth = path.join(dir_output_modelFitting,"mnsf_")
pp = fit_ini.generate_pickle_path("constant",base=mpth)

tro_ = training_multiSample.ModelTrainer(fit_ini,pickle_path=pp)

tro_ini = training.ModelTrainer(fit_ini,pickle_path=pp)
#del fit1

for iter in range(0,50):
  fit =tro_.train_model([tro_ini],
        list_Ds_train_batches_[0:3],list_D__[0:3])## 1 iteration of parameter updatings











