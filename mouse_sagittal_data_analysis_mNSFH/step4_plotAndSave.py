


##################################################################
import sys

dir_mNSF_functions='/users/ywang/Hansen_projects/scRNA/2023_05_08_mNSFH/mNSFH_May8_2023/functions'
sys.path.append(dir_mNSF_functions)

import load_packages # load functions from mNSF and NSF, as well as the dependencies





##################################################################



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

###################

L = 20
T=12
nsample_=2
J=500



# step1, create D12
Ntr = D1["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D1)
D1_train = Dtrain.batch(round(Ntr)+1)
Ntr
Ntr = D2["Y"].shape[0]
Dtrain = Dataset.from_tensor_slices(D2)
D2_train = Dtrain.batch(round(Ntr)+1)





D1["Z"]=D1['X']
D2["Z"]=D2['X']



list_D__=[D1,D2]

list_X=[D1['X'], D2['X']]

list_Ds_train_batches_=[D1_train,D2_train]



## save the result permernantly
nsample=2
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'__fullData_fitted_mNSFH_mouse_Sagittal.pkl')# 50 iterations


##################################################################################################################


#fit = pfh.ProcessFactorizationHybrid(N, J, L, D["Z"], T=T, psd_kernel=ker,
#                                       nonneg=True, lik="poi")
                                       
fit1=pfh.ProcessFactorizationHybrid(D1['Y'].shape[0],J,L,D1['Z'],T=T, psd_kernel=ker,nonneg=True,lik="poi")
fit2=pfh.ProcessFactorizationHybrid(D2['Y'].shape[0],J,L,D2['Z'],T=T, psd_kernel=ker,nonneg=True,lik="poi")



#fit.init_loadings(D["Y"],X=X,sz=D["sz"],shrinkage=0.3)
fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)



list_fit__=[fit1,fit2]
nsample=2
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'__fullData_fitted_mNSFH_mouse_Sagittal.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
            training_multiSample.assign_paras_from_np_to_tf_spat(list_fit__[kkk].spat,list_para_tmp)
            training_multiSample.assign_paras_from_np_to_tf_nonsp(list_fit__[kkk].nsp,list_para_tmp)
            list_fit__[kkk].Z=list_D__[kkk]["Z"]
            list_fit__[kkk].spat.sample_latent_GP_funcs(list_D__[kkk]['X'],S=100,chol=True)





#########################################################
dpth='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/'
os.chdir(dpth)
save_object(list_fit__, 'fit_threeSampleNSF_list_fit_mNSFH_mouse_Sagittal.pkl')

######################################################### deviance
nsample=2



#########################################################



#########################################################


M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]



inpfh = interpret_npfh_v3(list_fit__,list_X,S=100,lda_mode=False)


Fplot = inpfh["spatial"]["factors"][:,0:T]
fig,axes=visualize.multiheatmap(D1["X"], Fplot[0:M1_Z_,:], (1,T), cmap="Blues", **hmkw)
fig.savefig(path.join("mnsfh_sample1_full_v2_spat.png"),bbox_inches='tight')



fig,axes=visualize.multiheatmap(D2["X"], Fplot[M1_Z_:M2_Z_,:], (1,T), cmap="Blues", **hmkw)
fig.savefig(path.join("mnsfh_sample2_full_v2_spat.png"),bbox_inches='tight')
Fplot = inpfh["nonspatial"]["factors"][:,0:(L-T)]


fig,axes=visualize.multiheatmap(D1["X"], Fplot[0:M1_Z_,:], (1,L-T), cmap="Blues", **hmkw)
fig.savefig(path.join("mnsfh_sample1_full_v2_nonspat.png"),bbox_inches='tight')
fig,axes=visualize.multiheatmap(D2["X"], Fplot[M1_Z_:M2_Z_,:], (1,L-T), cmap="Blues", **hmkw)
fig.savefig(path.join("mnsfh_sample2_full_v2_nonspat.png"),bbox_inches='tight')





#########################################################
### assign parameters to each samples' model 
M1_Z_=D1["Z"].shape[0]
M2_Z_=M1_Z_+D2["Z"].shape[0]




# save the loading of each gene into csv filr
loadings=visualize.get_loadings(list_fit__[0])
#>>> loadings.shape
#(900, 7)
# convert array into dataframe

DF = pd.DataFrame(loadings) 
# save the dataframe as a csv file
DF.to_csv(("loadings_NPF_sample1_2_v2_L20__fullData_50iterations_.csv"))



W = inpfh["spatial"]["loadings"]#*inpf["totals"][:,None]
Wdf=pd.DataFrame(W*inpfh["spatial"]["totals1"][:,None], index=ad.var.index, columns=range(1,T+1))
Wdf.to_csv(path.join("loadings_mNSFH_spde_spat.csv"))

W = inpfh["nonspatial"]["loadings"]#*inpf["totals"][:,None]
Wdf=pd.DataFrame(W*inpfh["nonspatial"]["totals1"][:,None], index=ad.var.index, columns=range(1,L-T+1))
Wdf.to_csv(path.join("loadings_mNSFH_spde_nsp.csv"))


Fplot = inpfh["spatial"]["factors"][:,0:T]
DF = pd.DataFrame(Fplot[: M1_Z_,:]) 
DF.to_csv(path.join("mNSFH_spat_s1.csv"))
DF = pd.DataFrame(Fplot[M1_Z_: M2_Z_,:])
DF.to_csv(path.join("mNSFH_spat_s2.csv"))


Fplot = inpfh["nonspatial"]["factors"][:,0:(L-T)]
DF = pd.DataFrame(Fplot[: M1_Z_,:]) 
DF.to_csv(path.join("mNSFH_nsp_s1.csv"))
DF = pd.DataFrame(Fplot[M1_Z_: M2_Z_,:])
DF.to_csv(path.join("mNSFH_nsp_s2.csv"))


##################################################################################################################
############### explore parameter estimates
##################################################################################################################
dpth='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/'
import pickle
import os

os.chdir(dpth)


kkk=0
with open('list_para_'+ str(kkk+1) +'__fullData_fitted_mNSFH_mouse_Sagittal.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)

#save_object(list_fit__, 'fit_threeSampleNSF_list_fit_mNSFH_mouse_Sagittal.pkl')
list_para_tmp["amplitude"]
#array([1.0000632, 1.0000632, 1.0000632, 1.0000632, 1.0000632, 1.0000632,
#       1.0000632, 1.0000632, 1.0000632, 1.0000632, 1.0000632, 1.0000632],
#      dtype=float32)

list_para_tmp["length_scale"]
#array([0.09999048, 0.09999048, 0.09999048, 0.09999048, 0.09999048,
#       0.09999048, 0.09999048, 0.09999048, 0.09999048, 0.09999048,
#       0.09999048, 0.09999048], dtype=float32)





















