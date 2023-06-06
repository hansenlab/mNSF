

##################################################################
import sys

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/2023_05_08_mNSFH/mNSFH_May8_2023/functions'
sys.path.append(dir_mNSF_functions)
import load_packages # load functions from mNSF and NSF, as well as the dependencies





##### load data
#dpth="/dcs04/hansen/data/ywang/ST/data_10X_ST//mouse_Sagittal/put/"

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



D1["Z"]=D1['X']
D2["Z"]=D2['X']



###################
###################
###################

L = 20
T=12
nsample_=2
J=500





list_D__=[D1,D2]


D12=list_D__[1]




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




list_Ds_train_batches_=[D1_train,D2_train]




#del ad
#del D1,D2,D3

fit_ini=pfh.ProcessFactorizationHybrid(D1['Y'].shape[0],J,L,D1['Z'],T=T, psd_kernel=ker,nonneg=True,lik="poi")

fit_ini.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)





pp = fit_ini.generate_pickle_path("constant",base=mpth)
misc.mkdir_p(pp)




reload(pfh_multiSample)

tro_ = training_multiSample.ModelTrainer(fit_ini,pickle_path=pp)
tro_ini = training.ModelTrainer(fit_ini,pickle_path=pp)
#del fit1

for iter in range(0,50):
  fit =tro_.train_model([tro_ini],
        list_Ds_train_batches_,list_D__)## 1 iteration of parameter updatings

#list_self: tro_ini


list_self=[tro_ini]
#WARNING:tensorflow:Gradients do not exist for variables ['variational/covar_tril:0', 'variational/mean:0', 'variational_location:0', 'variational_scale:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?


