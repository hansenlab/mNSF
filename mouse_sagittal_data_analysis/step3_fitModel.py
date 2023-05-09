
##################################################################


dir_mNSF_functions='/dcs04/hansen/data/ywang/ST/mNSF_package/functions/'
sys.path.append(dir_mNSF_functions)




import load_packages # load functions from mNSF and NSF, as well as the dependencies




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


L=8

###################
###################
###################


list_D__=[D1,D2]


D12=list_D__[1]


fit12=pf.ProcessFactorization(J,L,D12['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit12.init_loadings(D12["Y"],X=D12['X'],sz=D12["sz"],shrinkage=0.3)

## 1500 spots per sample show memory issues after running 15mins (a few iterations throughout the 12 samples)
mpth = path.join("models_L8_")#100 batches - finished running one round of sample 1-12, stuck at sample 2 at the 2nd round
#mpth = path.join(pth,"models__nsample_memorySaving_qsub_gpu_fullDasta_L7_byBatches_v9_rep2_")#10 batches

pp = fit12.generate_pickle_path("constant",base=mpth)

misc.mkdir_p(mpth)
misc.mkdir_p(pp)


list_Ds_train_batches_=[D1_train,D2_train]


#del ad
#del D1,D2,D3

tro_ = training_multiSample.ModelTrainer(fit12,pickle_path=pp)#run without error message
#tro = training.ModelTrainer.from_pickle(pp)




tro_1 = training.ModelTrainer(fit12,pickle_path=pp)#run without error message
#del fit12

import time

time.time()
1683640214.0678272

for kkk in range(0,50):
	fit =tro_.train_model([tro_1],
        	list_Ds_train_batches_,list_D__)## 1 iteration
        	

time.time()
1683640273.4912817

#59.4 seconds

