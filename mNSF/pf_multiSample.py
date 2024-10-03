#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(spatial/temporal) Process Factorization base class

This class implements a multi-sample non-negative spatial factorization model.
It's designed to analyze spatial transcriptomics data from multiple samples
without the need for spatial alignment. The model uses Gaussian Processes (GPs)
to capture spatial dependencies and variational inference for tractable computation.

Key features:
- Handles multiple samples simultaneously
- Supports non-negative factorization
- Uses inducing points for scalable GP inference
- Allows for different likelihoods (Poisson, Gaussian)

@author: Yi Wang based on earlier work by Will Townes for the NSF package. 
"""

# Import necessary libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from os import path
from math import ceil
from tensorflow import linalg as tfl
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from mNSF.NSF import likelihoods, misc, nnfu

# Set up TensorFlow Probability aliases for convenience
tfd = tfp.distributions
tfb = tfp.bijectors
tv = tfp.util.TransformedVariable
tfk = tfp.math.psd_kernels

# Set default data type and random number generator
dtp = "float32"  # Using 32-bit floats for efficiency
rng = np.random.default_rng()  # NumPy's new random number generator

class ProcessFactorization_multiSample(tf.Module):
    def __init__(self, J, L, Z, nsample, lik="poi", psd_kernel=tfk.MaternThreeHalves,
                 nugget=1e-5, length_scale=0.1, disp="default",
                 nonneg=False, isotropic=True, feature_means=None, **kwargs):
        """
        Non-negative process factorization

        This model factorizes spatial gene expression data into latent factors
        and spatial processes. It can handle multiple samples simultaneously.

        Parameters:
        - J: Number of features in the multivariate outcome (e.g., number of genes)
        - L: Number of desired latent Gaussian processes (factors)
        - Z: Coordinates of inducing point locations in input space
             Shape: (M, D) where M is number of inducing points, D is spatial dimensions
        - nsample: Number of samples in the dataset
        - lik: Likelihood model ("poi" for Poisson, "gau" for Gaussian)
        - psd_kernel: Positive semidefinite kernel function for GP
                      Default is Matérn 3/2, which balances smoothness and flexibility
        - nugget: Small value added to diagonal of kernel matrix for numerical stability
        - length_scale: Initial length scale for the kernel, controls smoothness of GP
        - disp: Overdispersion parameter initialization (for negative binomial likelihood)
        - nonneg: If True, use non-negative factorization (constrain factors to be positive)
        - isotropic: If True, use isotropic kernel (same length scale in all dimensions)
        - feature_means: Feature means for centered input data (used only for Gaussian likelihood)
        """
        super().__init__(**kwargs)
        
        # Store model parameters
        self.lik = lik
        self.isotropic = isotropic
        M, D = Z.shape  # M: number of inducing points, D: dimensionality of input space
        self.Z = tf.Variable(Z, trainable=False, dtype=dtp, name="inducing_points")
        self.nonneg = tf.Variable(nonneg, trainable=False, name="is_non_negative")
        
        # Initialize variational parameters
        with tf.name_scope("variational"):
            # Delta represents the mean of the variational distribution q(u)
            # Shape: (L, M) - one mean vector for each latent GP at each inducing point
            self.delta = tf.Variable(rng.normal(size=(L,M)), dtype=dtp, name="mean")
        
        # Initialize GP hyperparameters
        with tf.name_scope("gp_mean"):
            if self.nonneg:
                # For non-negative factorization, use log-normal approximation to Dirichlet
                # This ensures that the factors remain positive
                prior_mu, prior_sigma = misc.lnormal_approx_dirichlet(max(L,1.1))
                self.beta0 = tf.Variable(prior_mu*tf.ones((L*nsample,1)), dtype=dtp,
                                         name="intercepts")
            else:
                # For unconstrained factorization, initialize intercepts to zero
                self.beta0 = tf.Variable(tf.zeros((L*nsample,1)), dtype=dtp, name="intercepts")
            # Initialize slopes (coefficients) to zero
            # Shape: (L*nsample, D) - allows for sample-specific spatial trends
            self.beta = tf.Variable(tf.zeros((L*nsample,D)), dtype=dtp, name="slopes")
        
        # Initialize GP kernel parameters
        with tf.name_scope("gp_kernel"):
            # Nugget term for numerical stability
            self.nugget = tf.Variable(nugget, dtype=dtp, trainable=False, name="nugget")
            # Amplitude is constrained to be positive using a softplus transformation
            self.amplitude = tv(np.tile(1.0,[L]), tfb.Softplus(), dtype=dtp, name="amplitude")
            self._ls0 = length_scale
            if self.isotropic:
                # For isotropic kernel, use a single length scale per latent dimension
                self.length_scale = tv(np.tile(self._ls0,[L]), tfb.Softplus(),
                                       dtype=dtp, name="length_scale")
            else:
                # For anisotropic kernel, use separate length scales for each input dimension
                self.scale_diag = tv(np.tile(np.sqrt(self._ls0),[L,D]),
                                     tfb.Softplus(), dtype=dtp, name="scale_diag")
        
        # Initialize loadings weights (W)
        if self.nonneg:
            # For non-negative factorization, initialize W with positive values
            # Shape: (J, L) - each column represents a latent factor
            self.W = tf.Variable(rng.exponential(size=(J,L)), dtype=dtp,
                                 constraint=misc.make_nonneg, name="loadings")
        else:
            # For unconstrained factorization, initialize W with normal distribution
            self.W = tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings")
        
        self.psd_kernel = psd_kernel
        self._disp0 = disp
        self._init_misc()
        
        # Store feature means for Gaussian likelihood with centered data
        if self.lik=="gau" and not self.nonneg:
            self.feature_means = feature_means
        else:
            self.feature_means = None

    def _init_misc(self):
        """
        Initialize miscellaneous parameters and set up caching for kernel computations
        This method is called at the end of __init__ to finalize the setup
        """
        J = self.W.shape[0]
        # Initialize likelihood-specific parameters (e.g., dispersion for negative binomial)
        self.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)
        
        # Set up caching for kernel and non-kernel variables
        # This separation allows for efficient updates during optimization
        self.trvars_kernel = tuple(i for i in self.trainable_variables if i.name[:10]=="gp_kernel/")
        self.trvars_nonkernel = tuple(i for i in self.trainable_variables if i.name[:10]!="gp_kernel/")
        
        # Initialize the kernel object
        if self.isotropic:
            self.kernel = self.psd_kernel(amplitude=self.amplitude, length_scale=self.length_scale)
        else:
            self.kernel = tfk.FeatureScaled(self.psd_kernel(amplitude=self.amplitude), self.scale_diag)

    def get_dims(self):
        """Return the number of latent dimensions (factors)"""
        return self.W.shape[1]

    def get_loadings(self):
        """Return the loadings matrix W as a numpy array"""
        return self.W.numpy()

    def set_loadings(self, Wnew):
        """Set the loadings matrix W to a new value"""
        self.W.assign(Wnew, read_value=False)

    def init_loadings(self, Y, list_X, list_Z, X=None, sz=1, **kwargs):
        """
        Initialize the loadings matrix from data Y
        Uses either PCA or NMF depending on whether non-negative constraint is applied
        """
        if self.nonneg:
            init_npf_with_nmf(self, Y, list_X, list_Z, X=X, sz=sz, **kwargs)
        else:  # real-valued factors
            if self.lik in ("poi", "nb"):
                pass  # TODO: implement GLM-PCA initialization
            elif self.lik == "gau":
                L = self.W.shape[1]
                fit = TruncatedSVD(L).fit(Y)
                self.set_loadings(fit.components_.T)
            else:
                raise likelihoods.InvalidLikelihoodError

    # ... (other methods would follow here)

# Helper functions
def smooth_spatial_factors(F, Z, list_X, list_Z, X=None):
    """
    Smooth spatial factors using linear regression and K-nearest neighbors
    
    This function is used to initialize the spatial factors in a way that
    respects the spatial structure of the data.
    
    Parameters:
    - F: Real-valued factors (on the log scale for non-negative factorization)
    - Z: Inducing point locations
    - list_X: List of spatial coordinates for each sample
    - list_Z: List of inducing point locations for each sample
    - X: Combined spatial coordinates (optional)
    
    Returns:
    - U: Smoothed factors at inducing points
    - beta0: Intercepts from linear regression
    - beta: Slopes from linear regression
    """
    # ... (implementation details)

def init_npf_with_nmf(fit, Y, list_X, list_Z, X=None, sz=1, pseudocount=1e-2, factors=None,
                      loadings=None, shrinkage=0.2):
    """
    Initialize non-negative process factorization with non-negative matrix factorization
    
    This function is used to provide a good starting point for the optimization
    of the non-negative spatial factorization model.
    
    Parameters:
    - fit: ProcessFactorization_multiSample object
    - Y: Data matrix (genes x spots)
    - list_X: List of spatial coordinates for each sample
    - list_Z: List of inducing point locations for each sample
    - X: Combined spatial coordinates (optional)
    - sz: Size factors (e.g., total counts per spot)
    - pseudocount: Small value added to avoid log(0)
    - factors: Initial factors (optional)
    - loadings: Initial loadings (optional)
    - shrinkage: Shrinkage parameter for regularization
    """
    # ... (implementation details)
