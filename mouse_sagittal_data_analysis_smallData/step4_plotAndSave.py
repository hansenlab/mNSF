


##################################################################
import sys

dir_mNSF_functions='/dcs04/hansen/data/ywang/ST/mNSF_package/functions'
sys.path.append(dir_mNSF_functions)
import load_packages # load functions from mNSF and NSF, as well as the dependencies


dpth='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/'


##################################################################


## sample 1
#ad = read_h5ad((dpth,"data_s1.h5ad"))
Y=pd.read_csv(path.join(dpth,'Y_sample1_smallData.csv.csv')
X=pd.read_csv(path.join(dpth,'X_sample1_smallData.csv.csv')
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
Y=pd.read_csv(path.join(dpth,'Y_sample2_smallData.csv.csv')
X=pd.read_csv(path.join(dpth,'X_sample2_smallData.csv.csv')
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





D1["Z"]=D1['X']
D2["Z"]=D2['X']





list_D__=[D1,D2]
#del D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12
psutil.Process().memory_info().rss / (1024 * 1024 * 1024)



list_Ds_train_batches_=[D1_train,D2_train]


## save the result permernantly

nsample=2
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'__fullData_fitted_L8_mouse_Sagittal.pkl')# 50 iterations


##################################################################################################################


fit1=pf_ori_mod.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf_ori_mod.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")


fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)



list_fit__=[fit1,fit2]
nsample=2
for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
            training.assign_paras_from_np_to_tf(list_fit__[kkk],list_para_tmp)
            list_fit__[kkk].Z=list_D__[kkk]["Z"]
            list_fit__[kkk].sample_latent_GP_funcs(list_D__[kkk]['X'],S=100,chol=True)


#########################################################
dpth='/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal/put/'
os.chdir(dpth)
save_object(list_fit__, 'fit_threeSampleNSF_list_fit_L8_mouse_Sagittal.pkl')

######################################################### deviance
nsample=2
for kkk in range(0,nsample):
  fit_tmp=list_fit__[kkk]
  D_tmp=list_D__[kkk]
  Mu_tr,Mu_val = fit_tmp.predict(D_tmp,Dval=None,S=10)
  Mu_tr_df = pd.DataFrame(Mu_tr) 
  #misc.poisson_deviance(D_tmp["Y"],Mu_tr)
  Mu_tr_df.to_csv("Mu_tr_df_twoSampleNSF_sample_"+str(kkk+1)+"_L8.csv")




######################################################### factors
list_X__=[D1["X"],D2["X"],]
M1_Y_=D1["Y"].shape[0]
M2_Y_=M1_Y_+D2["Y"].shape[0]


inpf12 = interpret_npf_v3(list_fit__,list_X__,S=100,lda_mode=False)
Fplot = inpf12["factors"][:,[0,1,2,3,4,5,6,7]]



#cwd = os.getcwd()



fig,axes=visualize.multiheatmap(D12["X"], Fplot[:D12["X"].shape[0],:], (1,3), cmap="Blues", **hmkw)

# save the loading of each gene into csv filr
loadings=visualize.get_loadings(list_fit__[0])
#>>> loadings.shape
#(900, 7)
# convert array into dataframe
import pandas as pd

DF = pd.DataFrame(loadings) 
# save the dataframe as a csv file
DF.to_csv(("loadings_NPF_sample1_2_v2_L8__fullData_50iterations_.csv"))





W = inpf12["loadings"]#*inpf["totals"][:,None]
Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))
Wdf.to_csv(path.join("loadings_NPF_sample1_2_v2_L8__fullData_50iterations_spde_.csv"))



DF = pd.DataFrame(Fplot[: M1_Y_,:]) 
#DF.to_csv("NPF_sample2_v2_edited_L7_v3_fulldata_nsample.csv")
DF.to_csv(path.join("NPF_sample1_L8_v2_fulldata_twoSample__.csv"))
DF = pd.DataFrame(Fplot[M1_Y_: M2_Y_,:]) 
DF.to_csv(path.join("NPF_sample2_L8_v2_fulldata_twoSample__.csv"))











