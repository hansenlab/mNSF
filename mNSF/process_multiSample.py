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
import pandas as pd

# Define the kernel function for Gaussian Process
# MaternThreeHalves is a specific covariance function used in Gaussian processes
ker = tm.psd_kernels.MaternThreeHalves


def get_D(X,Y,rescale_spatial_coords=True):	
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
	if rescale_spatial_coords: X = preprocess.rescale_spatial_coords(X)
	X=X.to_numpy()

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


def get_listDtrain(list_D_,nbatch=1):
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
	list_Dtrain=list()
	nsample=len(list_D_)
	for ksample in range(0,nsample):
		D=list_D_[ksample]
		Ntr = D["Y"].shape[0] # Number of observations in this sample

		# Convert dictionary to TensorFlow Dataset
		Dtrain = Dataset.from_tensor_slices(D)
		
		# Batch the data
		if (nbatch==1): D_train = Dtrain.batch(round(Ntr)+1)
		else:
			Ntr_batch=round(Ntr/nbatch)+1 # Number of observations in this sample
			D_train = Dtrain.batch(round(Ntr_batch)+1)
		list_Dtrain.append(D_train)
	return list_Dtrain
	
def get_vec_batch(nsample, nchunk):
	vec_batch = []
	for ksample in range(0,nsample):
		vec_batch= vec_batch + [False] + [True] * (nchunk-1)
	return vec_batch


def get_chunked_data(X, Y, nchunk, method='random'):
    """
    Split spatial transcriptomics data into chunks using k-means clustering or random assignment.
    
    Args:
    X: Spatial coordinates of the spots/cells
    Y: Gene expression data (genes x spots/cells) 
    nchunk: Number of chunks to split the data into
    method: Chunking method - 'kmeans' or 'random' (default: 'random')
    
    Returns:
    tuple: (list_D_sampleTmp, list_X_sampleTmp) containing chunked data dictionaries and coordinates
    """
    list_D_sampleTmp = []
    list_X_sampleTmp = []
    nspot = X.shape[0]
    D_unchunked = get_D(X, Y)
    
    if method == 'balanced_kmeans':
        from sklearn.cluster import KMeans
        coords = X.iloc[:, :2].values
        
        # Initialize with regular k-means
        kmeans = KMeans(n_clusters=nchunk, random_state=42)
        labels = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_
        
        # Target size for each cluster
        target_size = nspot // nchunk
        
        # Iteratively balance clusters
        max_iter = 20
        for _ in range(max_iter):
            # Calculate distances to all centers
            distances = np.zeros((nspot, nchunk))
            for i in range(nchunk):
                distances[:, i] = np.sum((coords - centers[i])**2, axis=1)
            
            # Sort points by distance to assigned cluster
            cluster_sizes = np.bincount(labels, minlength=nchunk)
            new_labels = np.zeros(nspot, dtype=int)
            
            # For each cluster
            unassigned = np.ones(nspot, dtype=bool)
            
            # First, handle clusters that are too small
            for i in range(nchunk):
                if cluster_sizes[i] < target_size:
                    # Get closest unassigned points
                    dist_to_cluster = distances[unassigned, i]
                    n_needed = target_size - cluster_sizes[i]
                    closest = np.argsort(dist_to_cluster)[:n_needed]
                    unassigned_indices = np.where(unassigned)[0]
                    points_to_assign = unassigned_indices[closest]
                    new_labels[points_to_assign] = i
                    unassigned[points_to_assign] = False
            
            # Then assign remaining points to closest clusters that aren't full
            remaining_points = np.where(unassigned)[0]
            for point in remaining_points:
                # Find closest cluster that isn't full
                dist_to_clusters = distances[point]
                for closest in np.argsort(dist_to_clusters):
                    if np.sum(new_labels == closest) < target_size:
                        new_labels[point] = closest
                        break
            
            # Update centers
            old_centers = centers.copy()
            for i in range(nchunk):
                if np.sum(new_labels == i) > 0:
                    centers[i] = coords[new_labels == i].mean(axis=0)
            
            # Check for convergence
            if np.allclose(old_centers, centers):
                break
            
            labels = new_labels
        
        # Create chunks based on final labels
        for k in range(nchunk):
            mask = labels == k
            Y_chunk = D_unchunked['Y'][mask, :]
            X_chunk = X.iloc[mask]
            Y_chunk = pd.DataFrame(Y_chunk)
            D = get_D(X_chunk, Y_chunk, rescale_spatial_coords=False)
            list_D_sampleTmp.append(D)
            list_X_sampleTmp.append(X_chunk)
    elif method == 'random':
        indices = np.random.permutation(nspot)
        nspot_perChunk = int(nspot/nchunk)
        
        for k in range(0,nchunk):                                                                             
            st = nspot_perChunk*k
            end_ = nspot_perChunk*(k+1)                                   
            if(k==nchunk-1): end_ = nspot
            Y_chunk = D_unchunked['Y'][st:end_,:]
            X_chunk = D_unchunked['X'][st:end_,:]
            Y_chunk = pd.DataFrame(Y_chunk)
            X_chunk = pd.DataFrame(X_chunk)
            D = get_D(X_chunk,Y_chunk,rescale_spatial_coords=False)
            list_D_sampleTmp.append(D)
            list_X_sampleTmp.append(X_chunk)  
    else:
        raise ValueError("method must be one of: 'kmeans' or 'random'")
        
    return list_D_sampleTmp, list_X_sampleTmp



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
	
	
		
	
def ini_multiSample(list_D_,L_, lik = 'nb', disp = "default",chol=True):
	"""
	Initialize mNSF (multi-sample Non-negative Spatial Factorization).
    
    This function sets up the initial state for the mNSF model, including
    creating ProcessFactorization objects for each sample and initializing
    their parameters.
    
    Args:
    list_D_: List of data dictionaries, one for each sample
    L_: Number of factors to use in the factorization
    lik: Likelihood function ('nb' for negative binomial)
    disp: Dispersion parameter for the negative binomial distribution
    
    Returns:
    list_fit_: List of initialized ProcessFactorization objects
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
		fit=pf.ProcessFactorization(J_,L_,D['Z'],X=list_X[ksample],psd_kernel=ker,nonneg=True,lik=lik,disp = disp, chol = chol)
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
    ave object to disk using pickle.
    
    This function serializes a Python object and saves it to a file,
    allowing it to be loaded later.
    
    Args:
    obj: Python object to save
    filename: Name of the file to save the object to
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
 

       
        
def interpret_npf_v3(list_fit,list_X,S=10,**kwargs):
  """
  	Interpret the non-negative process factorization results.
    
    This function samples from the learned Gaussian processes to generate
    interpretable factors and loadings.
    
    Args:
    list_fit: List of fitted ProcessFactorization objects
    list_X: List of spatial coordinates for each sample
    S: Number of samples to draw from the Gaussian processes
    **kwargs: Additional keyword arguments to pass to interpret_nonneg
    
    Returns:
    Dictionary containing interpretable loadings W, factors eF, and total counts vector
  """
  nsample=len(list_fit)
  for ksample in range(0,nsample):
    Fhat_tmp = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_X[ksample],S=S,chol=False)).T #NxL
    if ksample==0:
      Fhat_c=Fhat_tmp
    else:
      Fhat_c=np.concatenate((Fhat_c,Fhat_tmp), axis=0)
  return interpret_nonneg(np.exp(Fhat_c),list_fit[0].W.numpy(),sort=False,**kwargs)


def reorder_spatial_factors(factors, list_D, list_X_original):
    """
    Reorder factors to match the original spatial coordinates order.
    
    Args:
        factors: numpy array of factors from mNSF analysis
        list_D: list of chunked data dictionaries
        list_X_original: list of original spatial coordinate dataframes
    
    Returns:
        numpy array of reordered factors
    """
    # Calculate chunks per sample based on the total number of chunks divided by number of samples
    chunks_per_sample = len(list_D) // len(list_X_original)
    reordered_factors = np.zeros_like(factors)
    current_pos = 0
    
    for sample_idx, X_orig in enumerate(list_X_original):
        # Get start and end indices for this sample's chunks
        start_chunk = sample_idx * chunks_per_sample
        end_chunk = start_chunk + chunks_per_sample if sample_idx < len(list_X_original)-1 else len(list_D)
        
        # Collect all coordinates and their corresponding factors for this sample
        sample_coords = []
        sample_factors = []
        
        # Get coordinates and factors from each chunk of this sample
        for chunk_idx in range(start_chunk, end_chunk):
            chunk_start = sum(len(list_D[i]['X']) for i in range(chunk_idx))
            chunk_end = chunk_start + len(list_D[chunk_idx]['X'])
            
            sample_coords.extend(list_D[chunk_idx]['X'].values)
            sample_factors.extend(factors[chunk_start:chunk_end])
        
        sample_coords = np.array(sample_coords)
        sample_factors = np.array(sample_factors)
        
        # For each spot in the original data, find its position in the chunked data
        for orig_idx, orig_spot in X_orig.iterrows():
            # Find matching position in chunked data
            matches = np.where(
                (np.abs(sample_coords[:, 0] - orig_spot.iloc[0]) < 1e-10) & 
                (np.abs(sample_coords[:, 1] - orig_spot.iloc[1]) < 1e-10)
            )[0]
            
            if len(matches) > 0:
                reordered_factors[current_pos] = sample_factors[matches[0]]
            current_pos += 1
    
    return reordered_factors

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



