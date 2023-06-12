
from mNSF.NSF import pf
from tensorflow_probability import math as tm
from mNSF import pf_multiSample
ker = tm.psd_kernels.MaternThreeHalves


def get_D(X,Y):	
	X = preprocess.rescale_spatial_coords(X)
	X=X.to_numpy()
	ad = AnnData(Y,obsm={"spatial":X})
	ad.layers = {"counts":ad.X.copy()} #store raw counts before normalization changes ad.X
	pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
	pp.log1p(ad)
	D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,flip_yaxis=False)
	D["Z"]=D['X']
	return D


def get_listDtrain(list_D_):
	list_Dtrain=list()
	for kkk in range(0,nsample):
		D=list_D[kkk]
		Ntr = D["Y"].shape[0]
		Dtrain = Dataset.from_tensor_slices(D)
		D_train = Dtrain.batch(round(Ntr)+1)
		list_Dtrain.append(D_train)
	return list_Dtrain
	

def get_listSampleID(list_D_):
	list_sampleID=list()   
	index_=0                            
	for kkk in range(0,nsample):
		D=list_D[kkk]
		Ntr = D["Y"].shape[0]
		list_sampleID.append([index_:Ntr])
		index_=index_+Ntr
	return list_sampleID
	
	
		
	
def ini_multiSample(list_D_,L_):
	list_X=list()
	list_Z=list()
	list_sampleID_=list()
	nsample_=len(list_D_)
	index__=0
	for kkk in range(0,nsample_):
		D=list_D_[k]
		list_X.append(D['X'])
		list_Z.append(D['Z'])
		list_sampleID.append(D['Z'])
		Ntr = D["Y"].shape[0]
		list_sampleID_.append([index_:Ntr])
		index__=index__+Ntr                                   
	list_fit_=list()
	J_=list_D_[0]["Y"].shape[1]
	for k in range(0,nsample_):
		D=list_D_[k]
		fit=pf.ProcessFactorization(J_,L_,D['Z'],psd_kernel=ker,nonneg=True,lik="poi")
		fit.init_loadings(D["Y"],X=D['X'],sz=D["sz"],shrinkage=0.3)
		list_fit_.append(fit)]
		if k==0:
			X_concatenated=D['X']
			Z_concatenated=D['Z']
			Y_concatenated=D['Y']
			sz_concatenated=D['sz']
		else:
			X_concatenated=np.concatenate((X_concatenated, D['X']), axis=0)
			Z_concatenated=np.concatenate((Z_concatenated, D['Z']), axis=0)
			Y_concatenated=np.concatenate((Y_concatenated, D['Y']), axis=0)
			sz_concatenated=np.concatenate((sz_concatenated, D['sz']), axis=0)
	fit12_=pf_multiSample.ProcessFactorization_fit12(J_,L_,
  			Z_concatenated,
  			nsample=nsample_,
  			psd_kernel=ker,nonneg=True,lik="poi")
	fit12_.init_loadings(Y_concatenated,
  			list_X=list_X,
  			list_Z=list_Z,
  			sz=sz_concatenated), axis=0),shrinkage=0.3)
  	for k in range(0,nsample_):
		indices=list_sampleID_[k]
		delta=fit12_.delta.numpy()[:,indices]
		beta0=fit12_.beta0.numpy()[((k-1)*L_):(k*L_),:]
		beta=fit12_.beta.numpy()[((k-1)*L_):(k*L_),:]
		W=fit12_.W.numpy()
		list_fit_[k].delta.assign(delta) 
		list_fit_[k].beta0.assign(beta0)
		list_fit_[k].beta.assign(beta) 
		list_fit_[k].W.assign(W) 
		list_para_tmp=training_multiSample.store_paras_from_tf_to_np(list_fit[k])
	return list_fit_


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
 

       
        
def interpret_npf_v3(list_fit,list_X,S=10,**kwargs):
  """
  fit: object of type PF with non-negative factors
  X: spatial coordinates to predict on
  returns: interpretable loadings W, factors eF, and total counts vector
  """
  kk=0
  for fit_tmp in list_fit:
    kk=kk+1
  for kkk in range(0,kk):
    Fhat_tmp = misc.t2np(list_fit[kkk].sample_latent_GP_funcs(list_X[kkk],S=S,chol=False)).T #NxL
    if kkk==0:
      Fhat_c=Fhat_tmp
    else:
      Fhat_c=np.concatenate((Fhat_c,Fhat_tmp), axis=0)
  return interpret_nonneg(np.exp(Fhat_c),list_fit[kkk].W.numpy(),sort=False,**kwargs)




def interpret_nonneg(factors,loadings,lda_mode=False,sort=True):
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
    return {"factors":eF,"loadings":W,"totals1":eFsum,"totals2":Wsum}
  else: #spatialDE mode
    eF,W,Wsum,eFsum = rescale_as_lda(loadings,factors,sort=sort)
    return {"factors":eF,"loadings":W,"totals1":Wsum,"totals2":eFsum}



def rescale_as_lda(factors,loadings,sort=True):
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




