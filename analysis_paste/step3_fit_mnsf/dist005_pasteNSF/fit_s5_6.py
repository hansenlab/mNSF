sample_pair = 's5_s6'

######################
import os
import sys 
import pickle
import pandas as pd
from anndata import AnnData
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

hmkw = {"figsize":(7,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}


os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample6/')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_sample2_to_12/nsf-paper-main_sample6/')

######################
import numpy as np
from os import path
#from scanpy import read_h5ad
from tensorflow_probability import math as tm
from models import cf,pf,pfh
from utils import preprocess,misc,training,visualize,postprocess
from scanpy import pp

tfk = tm.psd_kernels

######################
dir_output="/dcs04/hansen/data/ywang/ST/DLPFC/PASTE_out_dist005_scMean/"

sys.path.append(dir_output)
os.chdir(dir_output)
mpth = path.join(dir_output,"models")

######################
def get_D(X,Y):	
	"""
	get the formated data as a directory
	"""
	X = preprocess.rescale_spatial_coords(X)
	X=X.to_numpy()
	ad = AnnData(Y,obsm={"spatial":X})
	ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
	pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
	pp.log1p(ad)
	D,_ = preprocess.anndata_to_train_val(ad, sz="mean", layer="counts", train_frac=1.0,flip_yaxis=False)
	D["Z"]=D['X']
	return D


######################
#%% Data loading
#Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data/Y_kp_'+str(sample_pair)+'_filterDist005.csv')
Y=pd.read_csv('/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_dist005/Y_kp_'+str(sample_pair)+'_filterDist005.csv')
X=pd.read_csv('/dcs04/hansen/data/ywang/ST/DLPFC/processed_Data_dist005/X_adj_paste_kp_'+str(sample_pair)+'_filterDist005.csv')


D=get_D(X,Y)

X = D["X"] 
N = X.shape[0]

J = Y.shape[1]

####################################################




####################################################
# %% Initialize inducing poi_sz-constantnts
L = 3
M = N 

ker = tfk.MaternOneHalf


Dtf = preprocess.prepare_datasets_tf(D,Dval=None,shuffle=False)


Z = D["Z"]

####################################################
#%% NSF mpdel fitting
fit = pf.ProcessFactorization(J,L,Z,psd_kernel=ker,nonneg=True,lik="nb")

fit.init_loadings(D["Y"],X=X,sz=D["sz"],shrinkage=0.3)

pp = fit.generate_pickle_path("constant_s"+sample_pair+"__qsub",base=mpth)

tro = training.ModelTrainer(fit,pickle_path=pp)

del fit
#del D
del Z
del X
del Y


fit_=tro.train_model(*Dtf) 


#tro = training.ModelTrainer.from_pickle(pp)
fit_ = tro.model

####################################################



save_object(fit_, 'fit_'+str(sample_pair)+'_paste.pkl')



#########################################################
#########################################################
#########################################################
with open( 'fit_'+str(sample_pair)+'_paste.pkl', 'rb') as inp:
              fit_ = pickle.load(inp)


fit_.sample_latent_GP_funcs(D['X'],S=100,chol=True)
inpf12 = postprocess.interpret_npf(fit_,D['X'],S=100,lda_mode=False)
Fplot = inpf12["factors"][:,[0,1,2]]


D['X'].shape
D['X'].shape
#(8736, 2)

#fig,axes=visualize.multiheatmap(D["X"], Fplot[:,:], (1,3), cmap="Blues", **hmkw)
#fig.savefig(path.join('ggblocks_s'+str(sample_pair)+'_paste.png'),bbox_inches='tight')

DF = pd.DataFrame(Fplot[:,:]) 


DF.to_csv('NPF_'+str(sample_pair)+'_paste.csv')









