#
##################################################################
##################################################################
import sys

dir_mNSF_functions='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/mNSF_package/functions'
sys.path.append(dir_mNSF_functions)
import load_packages # load functions from mNSF and NSF, as well as the dependencies



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
################################################################################################################
################################################################################################################

L = 7 # number of spatial factors
M = N 
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


list_D__=[D1,D2,D3]



#################
dir_output_modelFitting='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/mNSF_package/model_fitting/'
sys.path.append(dir_output_modelFitting)
#sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_modZ_nSamples_memorySaving_byBatches')


## load and save mNSF results
nsample=3


for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_fitted_sample1_5_9.pkl')


## format mNSF results
fit1=pf.ProcessFactorization(J,L,D1['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit2=pf.ProcessFactorization(J,L,D2['Z'],psd_kernel=ker,nonneg=True,lik="poi")
fit3=pf.ProcessFactorization(J,L,D3['Z'],psd_kernel=ker,nonneg=True,lik="poi")



fit1.init_loadings(D1["Y"],X=D1['X'],sz=D1["sz"],shrinkage=0.3)
fit2.init_loadings(D2["Y"],X=D2['X'],sz=D2["sz"],shrinkage=0.3)
fit3.init_loadings(D3["Y"],X=D3['X'],sz=D3["sz"],shrinkage=0.3)



list_fit__=[fit1,fit2,fit3]

for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'_1000spotsPerSample_restore.pkl')
            training_multiSample.assign_paras_from_np_to_tf(list_fit__[kkk],list_para_tmp)
            list_fit__[kkk].Z=list_D__[kkk]["Z"]
            list_fit__[kkk].sample_latent_GP_funcs(list_D__[kkk]['X'],S=100,chol=True)

save_object(list_fit__, path.join(dir_output_modelFitting,'fit_mNSF_sample1_5_9.pkl'))





## extract factors from the model fitting output
dir_output='/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/mNSF_package/results/'
list_X__=[D1["X"],D2["X"],D3["X"],D4["X"]]
M1_Y_=D1["Y"].shape[0]
M2_Y_=M1_Y_+D2["Y"].shape[0]
M3_Y_=M2_Y_+D3["Y"].shape[0]


inpf12 = interpret_npf_v3(list_fit__,list_X__,S=100,lda_mode=False)
Fplot = inpf12["factors"][:,[0,1,2,3,4,5,6]]


## save the loading of each gene into csv filr
loadings=visualize.get_loadings(list_fit__[1])
# convert array into dataframe
DF = pd.DataFrame(loadings) 
DF.to_csv(path.join(dir_output,"loadings_v2.csv"))

## save the factors of each sample into csv filr
DF = pd.DataFrame(Fplot[: M1_Y_,:]) 
DF.to_csv(path.join(dir_output,"sample1.csv"))
DF = pd.DataFrame(Fplot[M1_Y_: M2_Y_,:]) 
DF.to_csv(path.join(dir_output,"sample5.csv"))
DF = pd.DataFrame(Fplot[M2_Y_: M3_Y_,:]) 
DF.to_csv(path.join(dir_output,"sample.csv"))


## plot the factors in the 2-dim space
dir_plot=dir_output

#misc.mkdir_p(dir_plot)

fig,axes=visualize.multiheatmap(D1["X"],Fplot[: M1_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(dir_plot,"sample1.png"),bbox_inches='tight')
fig,axes=visualize.multiheatmap(D2["X"], Fplot[M1_Y_:M2_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(dir_plot,"sample2.png"),bbox_inches='tight')
fig,axes=visualize.multiheatmap(D3["X"], Fplot[M2_Y_:M3_Y_,:], (1,7), cmap="Blues", **hmkw)
fig.savefig(path.join(dir_plot,"sample3.png"),bbox_inches='tight')




