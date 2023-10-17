
from tensorflow_probability import math as tm
from mNSF import pf_multiSample,training_multiSample
from mNSF.NSF import misc,pf,preprocess,postprocess
from anndata import AnnData
from scanpy import pp
import numpy as np
from tensorflow.data import Dataset
import pickle

ker = tm.psd_kernels.MaternThreeHalves


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

def get_D_fromAnnData(ad):	# Same as get_D but starting from AnnData object
	"""
	get the formated data as a directory
	"""
	ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
	pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
	pp.log1p(ad)
	D,_ = preprocess.anndata_to_train_val(ad, sz="mean", layer="counts", train_frac=1.0,flip_yaxis=False)
	D["Z"]=D['X']
	return D


def get_listDtrain(list_D_,nbatch=1):
	"""
	get the training data (here using all data as training data)
	"""
	list_Dtrain=list()
	nsample=len(list_D_)
	for ksample in range(0,nsample):
		D=list_D_[ksample]
		Ntr = D["Y"].shape[0]
		Dtrain = Dataset.from_tensor_slices(D)
		if (nbatch==1): D_train = Dtrain.batch(round(Ntr)+1)
		else:
			Ntr_batch=round(Ntr/nbatch)+1
			D_train = Dtrain.batch(round(Ntr_batch)+1)
		list_Dtrain.append(D_train)
	return list_Dtrain
	

def get_listSampleID(list_D_):
	"""
	get the index of the sampleID for each spot
	"""
	list_sampleID=list()   
	index_=0      
	nsample=len(list_D_)                      
	for ksample in range(0,nsample):
		D=list_D_[ksample]
		Ntr = D["Y"].shape[0]
		list_sampleID.append(np.arange(index_,index_+Ntr))
		index_=index_+Ntr
	return list_sampleID
	
	
		
	
def ini_multiSample(list_D_,L_, lik = 'nb'):
	"""
	do initialization for mNSF
	"""
	list_X=list()
	list_Z=list()
	list_sampleID_=list()
	nsample_=len(list_D_)
	index__=0
	for ksample in range(0,nsample_):
		D=list_D_[ksample]
		list_X.append(D['X'])
		list_Z.append(D['Z'])
		Ntr = D["Z"].shape[0]
		list_sampleID_.append(np.arange(index__,index__+Ntr))
		index__=index__+Ntr                                   
	list_fit_=list()
	J_=list_D_[0]["Y"].shape[1]
	for ksample in range(0,nsample_):
		D=list_D_[ksample]
		fit=pf.ProcessFactorization(J_,L_,D['Z'],psd_kernel=ker,nonneg=True,lik=lik)
		fit.init_loadings(D["Y"],X=D['X'],sz=D["sz"],shrinkage=0.3)
		list_fit_.append(fit)
		if ksample==0:
			X_concatenated=D['X']
			Z_concatenated=D['Z']
			Y_concatenated=D['Y']
			sz_concatenated=D['sz']
		else:
			X_concatenated=np.concatenate((X_concatenated, D['X']), axis=0)
			Z_concatenated=np.concatenate((Z_concatenated, D['Z']), axis=0)
			Y_concatenated=np.concatenate((Y_concatenated, D['Y']), axis=0)
			sz_concatenated=np.concatenate((sz_concatenated, D['sz']), axis=0)
	fit_multiSample=pf_multiSample.ProcessFactorization_multiSample(J_,L_,
  			Z_concatenated,
  			nsample=nsample_,
  			psd_kernel=ker,nonneg=True,lik=lik)
	fit_multiSample.init_loadings(Y_concatenated,
  			list_X=list_X,
  			list_Z=list_Z,
  			sz=sz_concatenated,shrinkage=0.3)
	for ksample in range(0,nsample_):
		indices=list_sampleID_[ksample]
		#print("indices")
		#print(indices)
		indices=indices.astype(int)
		#print(indices)
		#print(fit_multiSample.delta.numpy()[:,indices])
		delta=fit_multiSample.delta.numpy()[:,indices]
		beta0=fit_multiSample.beta0.numpy()[((ksample)*L_):((ksample+1)*L_),:]
		beta=fit_multiSample.beta.numpy()[((ksample)*L_):((ksample+1)*L_),:]
		W=fit_multiSample.W.numpy()
		list_fit_[ksample].delta.assign(delta) 
		list_fit_[ksample].beta0.assign(beta0)
		list_fit_[ksample].beta.assign(beta) 
		list_fit_[ksample].W.assign(W) 
		#list_para_tmp=training_multiSample.store_paras_from_tf_to_np(list_fit_[k])
		save_object(list_fit_[ksample], 'fit_'+str(ksample+1)+'_restore.pkl')
		#save_object(list_fit_[ksample], 'fit_'+str(ksample+1)+'.pkl')
		#save_object(list_para_tmp, 'list_para_'+ str(k+1) +'.pkl')
		#save_object(list_para_tmp, 'list_para_'+ str(k+1) +'_restore.pkl')
	return list_fit_


def save_object(obj, filename):
    """
    save object to disk
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
 

       
        
def interpret_npf_v3(list_fit,list_X,S=10,**kwargs):
  """
  fit: object of type PF with non-negative factors
  X: spatial coordinates to predict on
  returns: interpretable loadings W, factors eF, and total counts vector
  """
  nsample=len(list_fit)
  for ksample in range(0,nsample):
    Fhat_tmp = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_X[ksample],S=S,chol=False)).T #NxL
    if ksample==0:
      Fhat_c=Fhat_tmp
    else:
      Fhat_c=np.concatenate((Fhat_c,Fhat_tmp), axis=0)
  return interpret_nonneg(np.exp(Fhat_c),list_fit[0].W.numpy(),sort=False,**kwargs)




def interpret_nonneg(factors,loadings,lda_mode=False,sort=False):

  """
  Rescale factors and loadings from a nonnegative factorization
  to improve interpretability. Two possible rescalings:

  1. Soft clustering of observations (lda_mode=True):
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "factor_sums"
  factor_sums is the "n" in the multinomial
  (ie the sum of the counts per observations)

  2. Soft clustering of features (lda_mode=False):
  Rows of loadings matrix sum to one, cols of factors matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "feature_sums"
  feature_sums is similar to an intercept term for each feature
  """
  if lda_mode:
    W,eF,eFsum,Wsum = rescale_as_lda(factors,loadings,sort=sort)##!!!!
    return {"factors":eF,"loadings":W,"totalsF":eFsum,"totalsW":Wsum}
  else: #spatialDE mode
    eF,W,Wsum,eFsum = rescale_as_lda(loadings,factors,sort=sort)
    return {"factors":eF,"loadings":W,"totalsW":Wsum,"totalsF":eFsum}



def rescale_as_lda(factors,loadings,sort=False):
  """
  Rescale nonnegative factors and loadings matrices to be
  comparable to LDA:
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  """
  W = postprocess.deepcopy(loadings)
  eF = postprocess.deepcopy(factors)
  W,wsum = postprocess.normalize_cols(W)
  #eF1,eFsum = postprocess.normalize_rows(eF*wsum)##??
  eF,eFsum = postprocess.normalize_rows(eF*wsum)##??
  if sort:
    o = np.argsort(-eF.sum(axis=0))
    return W[:,o],eF[:,o],eFsum
  else:
    return W,eF,eFsum,wsum




