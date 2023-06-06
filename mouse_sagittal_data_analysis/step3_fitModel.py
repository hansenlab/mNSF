

##################################################################
import sys

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSFH_2023_06_04_mNSFPerSampleTraining/functions'
sys.path.append(dir_mNSF_functions)

#import load_packages # load functions from mNSF and NSF, as well as the dependencies
from mNSF import training_multiSample_perSample



#################################################



###############################################################################################################

###############################################################################################################


############## Data loading
from scanpy import read_h5ad
#dpth='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/data_10X_ST/mouse_Sagittal/put/'

## sample 1
#ad = read_h5ad((dpth,"data_s1.h5ad"))
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample1_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample1.csv')
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
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample2_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample2.csv')
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




## sample 3
#ad = read_h5ad((dpth,"data_s2.h5ad"))
## sample 2
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample3_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample3.csv')
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
D3=D



## sample 4
#ad = read_h5ad((dpth,"data_s2.h5ad"))
## sample 2
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/Y_addedMarkerGenes_sample4_v2.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/X_sample4.csv')
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
D4=D




# step1, create D12
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



D1["Z"]=D1['X']
D2["Z"]=D2['X']
D3["Z"]=D3['X']
D4["Z"]=D4['X']



###################
###################
###################

L = 20

nsample_=4
J=500



list_D__=[D1,D2,D3,D4]


D12=list_D__[0]


rng = np.random.default_rng()
dtp = "float32"
pth = "simulations/ggblocks_lr"
dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)
mpth = path.join(pth,"_models_mNSFH_")

misc.mkdir_p(plt_pth) 
misc.mkdir_p(mpth)


list_Ds_train_batches_=[D1_train,D2_train,D3_train,D4_train]


#del ad
#del D1,D2,D3


fit_ini=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")

fit_ini.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)

pp = fit_ini.generate_pickle_path("constant",base=mpth)
misc.mkdir_p(pp)




reload(pfh_multiSample)

tro_ = training_multiSample_perSample.ModelTrainer(fit_ini,pickle_path=pp)
tro_ini = training.ModelTrainer(fit_ini,pickle_path=pp)
#del fit1

nsample=len(list_Ds_train_batches_)


dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/mNSFH_2023_06_04_mNSFPerSampleTraining/functions'
sys.path.append(dir_mNSF_functions)

for iter in range(0,50):
	for ksample in range(0,nsample):
		fit =tro_.train_model([tro_ini],ksample,list_Ds_train_batches_,list_D__)## 1 iteration of parameter updatings
			  

#list_self: tro_ini


list_self=[tro_ini]
#WARNING:tensorflow:Gradients do not exist for variables ['variational/covar_tril:0', 'variational/mean:0', 'variational_location:0', 'variational_scale:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?


