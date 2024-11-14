"""
mNSF Utility Functions

This file contains utility functions for the multi-sample Non-negative Spatial Factorization (mNSF) method.
mNSF is a computational approach for analyzing spatial transcriptomics data across multiple samples
without the need for spatial alignment.

Key components:
1. Data Preprocessing: Functions to format and normalize spatial transcriptomics data.
2. Model Initialization: Setup of the mNSF model for multiple samples.
3. Data Handling: Creation of TensorFlow datasets for efficient training.
4. Result Interpretation: Functions to interpret and rescale the factorization results.

The main workflow typically involves:
1. Formatting the input data using get_D() or get_D_fromAnnData().
2. Initializing the mNSF model with ini_multiSample().
3. Preparing the data for training with get_listDtrain().
4. After model fitting (not included in this file), interpreting results with interpret_npf_v3().

This file is part of the larger mNSF package and is designed to work in conjunction
with other components like pf_multiSample and training_multiSample.

@author: Yi Wang based on earlier work by Will Townes for the NSF package. Date: [Current Date]

For more information on the mNSF method, please refer to the accompanying publication

"""

# Import necessary libraries
from tensorflow_probability import math as tm
from mNSF import pf_multiSample,training_multiSample
from mNSF.NSF import misc,pf,preprocess,postprocess
from anndata import AnnData
from scanpy import pp
import numpy as np
from tensorflow.data import Dataset
import pickle

# Define the kernel function for Gaussian Process
# MaternThreeHalves is a specific covariance function used in Gaussian processes
ker = tm.psd_kernels.MaternThreeHalves


def get_D(X,Y):	
	"""
	Format the data as a dictionary for processing.
    
    This function takes spatial coordinates and gene expression data,
    normalizes them, and prepares them for further analysis.
    
    Args:
    X: Spatial coordinates of the spots/cells
    Y: Gene expression data (genes x spots/cells)
    
    Returns:
    D: Formatted data dictionary containing normalized data and metadata
	"""
	# Rescale spatial coordinates to a standard range
	X = preprocess.rescale_spatial_coords(X)
	# Convert to numpy array if not already
	if hasattr(X, 'to_numpy'): X = X.to_numpy()

	# Create an AnnData object, which is a standard format for single-cell data
	ad = AnnData(Y,obsm={"spatial":X})

	# Store raw counts before normalization changes ad.X
	ad.layers = {"counts":ad.X.copy()} 

	# Normalize the total counts per cell
	pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
	
	# Log-transform the data
	pp.log1p(ad)

	# Convert AnnData to a dictionary format suitable for training
	D,_ = preprocess.anndata_to_train_val(ad, sz="mean", layer="counts", train_frac=1.0,flip_yaxis=False)

	# Add spatial coordinates to the dictionary
	D["Z"]=D['X']
	return D

def get_D_fromAnnData(ad):	
	"""
	Format the data as a dictionary starting from an AnnData object.
    
    This function is similar to get_D, but it starts with an AnnData object
    instead of separate X and Y matrices.
    
    Args:
    ad: AnnData object containing gene expression and spatial information
    
    Returns:
    D: Formatted data dictionary containing normalized data and metadata
	"""
	# Store raw counts before normalization changes ad.X
	ad.layers = {"counts":ad.X.copy()} 
	
	# Normalize the total counts per cell
	pp.normalize_total(ad, inplace=True, layers=None, key_added="sizefactor")
	
	# Log-transform the data
	pp.log1p(ad)
	
	# Convert AnnData to a dictionary format suitable for training
	D,_ = preprocess.anndata_to_train_val(ad, sz="mean", layer="counts", train_frac=1.0,flip_yaxis=False)
	
	# Add spatial coordinates to the dictionary
	D["Z"]=D['X']
	return D


def get_listD_chunked(list_D_,list_nchunk=None):
	"""
	Prepare the training data by creating TensorFlow Datasets.
    
    This function converts the data dictionaries into TensorFlow Dataset objects,
    which are efficient for training machine learning models.
    
    Args:
    list_D_: List of data dictionaries, one for each sample
    nbatch: Number of batches to split the data into (default is 1)
    
    Returns:
    list_Dtrain: List of TensorFlow Datasets for training
	"""
	if(list_nchunk is None):list_nchunk=[1]*len(list_D_)
	list_Dtrain=list()
	nsample=len(list_D_)
	# data chunking
	nsample_splitted = sum(list_nchunk)
	list_D_chunk = list()
	for ksample in range(0,nsample):
		D=list_D_[ksample]
		nchunk = list_nchunk[ksample]
		X = D['X']
		Y = D['Y']
		nspot = X.shape[1]
		nspot_perChunk = int(nspot/nchunk)
		for kchunk in range(0,nchunk):
			st = (kchunk)*nspot_perChunk
			end_ = (kchunk+1)*nspot_perChunk
			if (kchunk==nchunk-1):end_=nspot
			X_chunk=X[st:end_,]
			Y_chunk=Y[st:end_,]
			D_chunk = get_D(X,Y)
			list_D_chunk.append(D_chunk)
	return list_D_chunk



def get_listSampleID(list_D_):
	"""
	Get the index of the sampleID for each spot.
    
    This function assigns a unique identifier to each spot/cell across all samples.
    
    Args:
    list_D_: List of data dictionaries, one for each sample
    
    Returns:
    list_sampleID: List of arrays, each containing the sample IDs for a sample
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
	
	
		
	

def save_object(obj, filename):
    """
    ave object to disk using pickle.
    
    This function serializes a Python object and saves it to a file,
    allowing it to be loaded later.
    
    Args:
    obj: Python object to save
    filename: Name of the file to save the object to
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
 
def get_listX_chunked(list_X_, list_nchunk=None):
    """
    Chunk spatial coordinates into smaller pieces.
    
    Args:
    list_X_: List of spatial coordinate arrays
    list_nchunk: List of integers specifying number of chunks for each sample
    
    Returns:
    list_X_chunk: List of chunked coordinate arrays
    """
    if list_nchunk is None:
        list_nchunk = [1] * len(list_X_)
        
    list_X_chunk = []
    
    for ksample, X in enumerate(list_X_):
        nchunk = list_nchunk[ksample]
        nspot = X.shape[0]
        nspot_perChunk = int(nspot / nchunk)
        
        for kchunk in range(nchunk):
            start = kchunk * nspot_perChunk
            end = (kchunk + 1) * nspot_perChunk if kchunk < nchunk - 1 else nspot
            X_chunk = X[start:end]
            list_X_chunk.append(X_chunk)
            
    return list_X_chunk

def get_listDtrain(list_D_, nbatch=1, list_nchunk=None):
    """
    Prepare the training data by creating TensorFlow Datasets from chunked data.
    
    Args:
    list_D_: List of data dictionaries
    nbatch: Number of batches for each chunk
    list_nchunk: List of integers specifying number of chunks for each sample
    
    Returns:
    list_Dtrain: List of TensorFlow Datasets for training
    """
    if list_nchunk is None:
        list_nchunk = [1] * len(list_D_)
        
    # First chunk the data
    list_D_chunk = get_listD_chunked(list_D_, list_nchunk)
    list_Dtrain = []
    
    # Create TensorFlow datasets for each chunk
    for D_chunk in list_D_chunk:
        Ntr = D_chunk["Y"].shape[0]
        
        # Convert dictionary to TensorFlow Dataset
        Dtrain = Dataset.from_tensor_slices(D_chunk)
        
        # Batch the data
        if nbatch == 1:
            D_train = Dtrain.batch(Ntr)
        else:
            Ntr_batch = round(Ntr/nbatch)
            D_train = Dtrain.batch(Ntr_batch)
            
        list_Dtrain.append(D_train)
        
    return list_Dtrain

def ini_multiSample(list_D_, L_, lik='nb', disp="default", chol=True):
    """
    Initialize mNSF model with chunked data support.
    
    Args:
    list_D_: List of data dictionaries
    L_: Number of factors
    lik: Likelihood function type
    disp: Dispersion parameter
    chol: Whether to use Cholesky decomposition
    
    Returns:
    list_fit_: List of initialized ProcessFactorization objects
    """
    # Get total number of spots across all samples for proper indexing
    total_spots = 0
    list_sampleID_ = []
    
    for D in list_D_:
        nspots = D['X'].shape[0]
        list_sampleID_.append(np.arange(total_spots, total_spots + nspots))
        total_spots += nspots
    
    # Initialize concatenated data arrays
    X_concatenated = np.concatenate([D['X'] for D in list_D_], axis=0)
    Z_concatenated = np.concatenate([D['Z'] for D in list_D_], axis=0)
    Y_concatenated = np.concatenate([D['Y'] for D in list_D_], axis=0)
    sz_concatenated = np.concatenate([D['sz'] for D in list_D_], axis=0)
    
    # Initialize individual fits
    list_fit_ = []
    nsample_ = len(list_D_)
    J_ = list_D_[0]["Y"].shape[1]  # Number of genes
    
    for ksample in range(nsample_):
        D = list_D_[ksample]
        fit = pf.ProcessFactorization(
            J_, L_, D['Z'], 
            X=D['X'],
            psd_kernel=ker,
            nonneg=True,
            lik=lik,
            disp=disp,
            chol=chol
        )
        fit.init_loadings(D["Y"], X=D['X'], sz=D["sz"], shrinkage=0.3)
        list_fit_.append(fit)
    
    # Initialize multi-sample fit
    fit_multiSample = pf_multiSample.ProcessFactorization_multiSample(
        J_, L_,
        Z_concatenated,
        nsample=nsample_,
        psd_kernel=ker,
        nonneg=True,
        lik=lik
    )
    
    fit_multiSample.init_loadings(
        Y_concatenated,
        list_X=[D['X'] for D in list_D_],
        list_Z=[D['Z'] for D in list_D_],
        sz=sz_concatenated,
        shrinkage=0.3
    )
    
    # Transfer parameters to individual fits
    for ksample in range(nsample_):
        indices = list_sampleID_[ksample].astype(int)
        delta = fit_multiSample.delta.numpy()[:, indices]
        beta0 = fit_multiSample.beta0.numpy()[ksample*L_:(ksample+1)*L_, :]
        beta = fit_multiSample.beta.numpy()[ksample*L_:(ksample+1)*L_, :]
        W = fit_multiSample.W.numpy()
        
        list_fit_[ksample].delta.assign(delta)
        list_fit_[ksample].beta0.assign(beta0)
        list_fit_[ksample].beta.assign(beta)
        list_fit_[ksample].W.assign(W)
        
        # Save checkpoint
        save_object(list_fit_[ksample], f'fit_{ksample+1}_restore.pkl')
    
    return list_fit_

def interpret_npf_v3(list_fit, list_X, list_nchunk=None, S=10, **kwargs):
    """
    Interpret the factorization results with support for chunked data.
    
    Args:
    list_fit: List of fitted ProcessFactorization objects
    list_X: List of spatial coordinates
    list_nchunk: List specifying number of chunks per sample
    S: Number of samples to draw
    **kwargs: Additional arguments for interpret_nonneg
    
    Returns:
    Dictionary containing interpretable results
    """
    # Get chunked coordinates
    listX_chunked = get_listX_chunked(list_X, list_nchunk)
    
    # Sample latent GPs for each chunk
    Fhat_chunks = []
    for ksample, X_chunk in enumerate(listX_chunked):
        # Get the corresponding fit object (accounting for chunking)
        fit_idx = ksample // (list_nchunk[ksample] if list_nchunk else 1)
        Fhat_tmp = misc.t2np(
            list_fit[fit_idx].sample_latent_GP_funcs(X_chunk, S=S, chol=False)
        ).T
        Fhat_chunks.append(Fhat_tmp)
    
    # Concatenate all chunks
    Fhat_c = np.concatenate(Fhat_chunks, axis=0)
    
    # Interpret the results
    return interpret_nonneg(
        np.exp(Fhat_c),
        list_fit[0].W.numpy(),
        sort=False,
        **kwargs
    )


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
  Rescale nonnegative factors and loadings matrices to be comparable to LDA.
    
    This function normalizes the factors and loadings so that:
    - Rows of the factor matrix sum to one
    - Columns of the loadings matrix sum to one
    
    Args:
    factors: Factor matrix (spots x factors)
    loadings: Loadings matrix (genes x factors)
    sort: If True, sort the factors and loadings by total magnitude
    
    Returns:
    W: Rescaled loadings
    eF: Rescaled factors
    eFsum: Sum of factors before normalization (interpretable as total counts)
    wsum: Sum of loadings before normalization
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



