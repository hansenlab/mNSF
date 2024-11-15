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

# Set up TensorFlow Probability aliases for convenience
from tensorflow import linalg as tfl
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from mNSF.NSF import likelihoods, misc, nnfu
tfd = tfp.distributions
tfb = tfp.bijectors
tv = tfp.util.TransformedVariable
tfk = tfp.math.psd_kernels

# Set default data type and random number generator
dtp = "float32"
rng = np.random.default_rng()

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
    self.isotropic=isotropic
    M,D = Z.shape # M: number of inducing points, D: dimensionality of input space
    self.Z = tf.Variable(Z, trainable=False, dtype=dtp, name="inducing_points")
    self.nonneg = tf.Variable(nonneg, trainable=False, name="is_non_negative")
    # Initialize variational parameters
    with tf.name_scope("variational"):
      # Delta represents the mean of the variational distribution q(u)
      # Shape: (L, M) - one mean vector for each latent GP at each inducing point
      self.delta = tf.Variable(rng.normal(size=(L,M)), dtype=dtp, name="mean") #LxM
      #_Omega_tril = self._init_Omega_tril(L,M,nugget=nugget)
      # _Omega_tril = .01*tf.eye(M,batch_shape=[L],dtype=dtp)
      #self.Omega_tril=tv(_Omega_tril, tfb.FillScaleTriL(), dtype=dtp, name="covar_tril") #LxMxM
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
        self.beta0 = tf.Variable(tf.zeros((L*nsample,1)), dtype=dtp, name="intercepts") #L
      # Initialize slopes (coefficients) to zero
      # Shape: (L*nsample, D) - allows for sample-specific spatial trends
      self.beta = tf.Variable(tf.zeros((L*nsample,D)), dtype=dtp, name="slopes") #LxD
    
    # Initialize GP kernel parameters
    with tf.name_scope("gp_kernel"):
      # Nugget term for numerical stability
      self.nugget = tf.Variable(nugget, dtype=dtp, trainable=False, name="nugget")
      # Amplitude is constrained to be positive using a softplus transformation
      self.amplitude = tv(np.tile(1.0,[L]), tfb.Softplus(), dtype=dtp, name="amplitude")
      self._ls0 = length_scale #store for .reset() method
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
                           constraint=misc.make_nonneg, name="loadings") #JxL
    else:
      # For unconstrained factorization, initialize W with normal distribution
      self.W = tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings") #JxL
    self.psd_kernel = psd_kernel #this is a class, not yet an object
    #likelihood parameters, set defaults
    self._disp0 = disp
    self._init_misc()
    # Store feature means for Gaussian likelihood with centered data
    if self.lik=="gau" and not self.nonneg:
      self.feature_means = feature_means
    else:
      self.feature_means = None

  @staticmethod
  def _init_Omega_tril(L, M, nugget=None):
    """
    convenience function for initializing the batch of lower triangular
    cholesky factors of the variational covariance matrices.
    L: number of latent dimensions (factors)
    M: number of inducing points
    """
    #note most of these operations are in float64 by default
    #the 0.01 below is to make the elbo more numerically stable at initialization
    Omega_sqt = 0.01*rng.normal(size=(L,M,M)) #LxMxM
    Omega = [Omega_sqt[l,:,:]@ Omega_sqt[l,:,:].T for l in range(L)] #list len L, elements MxM
    # Omega += nugget*np.eye(M)
    res = np.stack([np.linalg.cholesky(Omega[l]) for l in range(L)], axis=0)
    return res.astype(dtp)

  def _init_misc(self):
    """
      Initialize miscellaneous parameters and set up caching for kernel computations
      This method is called at the end of __init__ to finalize the setup
    """
    J = self.W.shape[0]
    self.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)
    #stuff to facilitate caching
    self.trvars_kernel = tuple(i for i in self.trainable_variables if i.name[:10]=="gp_kernel/")
    self.trvars_nonkernel = tuple(i for i in self.trainable_variables if i.name[:10]!="gp_kernel/")
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

  def set_loadings(self,Wnew):
    """Set the loadings matrix W to a new value"""
    self.W.assign(Wnew,read_value=False)

  def init_loadings(self,Y,list_X,list_Z,X=None,sz=1,**kwargs):
    """
      Initialize the loadings matrix from data Y
      Uses either PCA or NMF depending on whether non-negative constraint is applied
    """
    if self.nonneg:
      init_npf_with_nmf(self,Y,list_X,list_Z,X=X,sz=sz,**kwargs)
    else: #real-valued factors
      if self.lik in ("poi","nb"):
        pass #use GLM-PCA?
      elif self.lik=="gau":
        L = self.W.shape[1]
        fit = TruncatedSVD(L).fit(Y)
        self.set_loadings(fit.components_.T)
      else:
        raise likelihoods.InvalidLikelihoodError

  def generate_pickle_path(self,sz,base=None):
    """
    sz : str
      Indicate what type of size factors are used (eg 'none' or 'scanpy').
    base : str, optional
      Parent directory for saving pickle files. The default is cwd.
    """
    pars = {"L":self.W.shape[1], "lik":self.lik, "sz":sz,
            "model":"NPF" if self.nonneg else "RPF",
            "kernel":self.psd_kernel.__name__,
            "M":self.Z.shape[0]
            }
    pth = misc.params2key(pars)
    if base: pth = path.join(base,pth)
    return pth

  def get_kernel(self):
    return self.kernel

  def eval_Kuu_chol(self, kernel=None):
    if kernel is None:
      kernel = self.get_kernel()
    M,D = self.Z.shape
    Kuu = kernel.matrix(self.Z, self.Z) + self.nugget*tf.eye(M)
    #Kuu_chol is LxMxM and lower triangular in last two dims
    return tfl.cholesky(Kuu)

  def get_Kuu_chol(self, k_, kernel=None, from_cache=False):
    if not from_cache:
      Kuu_chol = self.eval_Kuu_chol(kernel=kernel)
      self.list_Kuu_chol[k_].assign(Kuu_chol) #update cache
      return Kuu_chol
    else: #use cached variable, gradients cannot propagate to kernel hps
      return self.list_Kuu_chol[k_]

  def get_mu_z(self):
    return self.beta0+tfl.matmul(self.beta, self.Z, transpose_b=True) #LxM

  def sample_latent_GP_funcs(self, X, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
    """
    Draw random samples of the latent variational GP function values "F"
    based on spatial coordinates X.
    The sampling comes from the variational approximation to the posterior.
    This function is needed to compute the expected log-likelihood term of the
    ELBO.
    X is a numpy array with shape NxD
    N=number of observations
    D=number of spatial dimensions
    The first dimension can be observations from a minibatch.
    S is the number of random samples to draw from latent GPs
    logscale: if False the function vals are exponentiated before returning
    i.e. they are positive valued random functions.
    If logscale=True (default), the functions are real-valued
    """
    if kernel is None:
      kernel = self.get_kernel()
    if mu_z is None:
      mu_z = self.get_mu_z()
    if Kuu_chol is None:
      Kuu_chol = self.get_Kuu_chol(kernel=kernel, from_cache=(not chol))
    if (not chol):
      N = X.shape[0]
      L = self.W.shape[1]
      mu_x = self.beta0+tfl.matmul(self.beta, X, transpose_b=True) #LxN
      mu_tilde = mu_x + tfl.matvec(self.alpha_x, self.delta-mu_z, transpose_a=True) #LxN
      #a_t_Kchol = self.a_t_Kchol
      #aKa = tf.reduce_sum(tf.square(a_t_Kchol), axis=2) #LxN
      Sigma_tilde = self.Sigma_tilde #LxN
    if chol:
      alpha_x = tfl.cholesky_solve(Kuu_chol, Kuf) #LxMxN
      N = X.shape[0]
      L = self.W.shape[1]
      mu_x = self.beta0+tfl.matmul(self.beta, X, transpose_b=True) #LxN
      Kuf = kernel.matrix(self.Z, X) #LxMxN
      Kff_diag = kernel.apply(X, X, example_ndims=1)+self.nugget #LxN
    
      mu_tilde = mu_x + tfl.matvec(alpha_x, self.delta-mu_z, transpose_a=True) #LxN
      #compute the alpha(x_i)'(K_uu-Omega)alpha(x_i) term
      a_t_Kchol = tfl.matmul(alpha_x, Kuu_chol, transpose_a=True) #LxNxM
      aKa = tf.reduce_sum(tf.square(a_t_Kchol), axis=2) #LxN
      a_t_Omega_tril = tfl.matmul(alpha_x, self.Omega_tril, transpose_a=True) #LxNxM
      aOmega_a = tf.reduce_sum(tf.square(a_t_Omega_tril), axis=2) #LxN
      Sigma_tilde = Kff_diag - aKa + aOmega_a #LxN
    #print(S)
    #print(L)
    #print(N)
    eps = tf.random.normal((S,L,N)) #note this is not the same random generator as self.rng!
    return mu_tilde + tf.math.sqrt(Sigma_tilde)*eps #"F", dims: SxLxN

  def sample_predictive_mean(self, X, sz=1, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
    """
    See sample_latent_variational_GP_funcs for X,S definitions
    sz is a tensor of shape (N,1) of size factors.
    Typically sz would be the rowSums or rowMeans of the outcome matrix Y.
    """
    F = self.sample_latent_GP_funcs(X, S=S, kernel=kernel, mu_z=mu_z,
                                    Kuu_chol=Kuu_chol, chol=chol) #SxLxN
    if self.nonneg:
      Lam = tfl.matrix_transpose(tfl.matmul(self.W, tf.exp(F))) #SxNxJ
      if self.lik=="gau":
        return Lam
      else:
        return sz*Lam
    else:
      Lam = tfl.matrix_transpose(tfl.matmul(self.W, F))
      if self.lik=="gau": #identity link
        return Lam
      else: #log link (poi, nb)
        return tf.exp(tf.math.log(sz)+Lam)

  def eval_kl_term(self, mu_z, Kuu_chol):
    """
    KL divergence from the prior distribution to the variational distribution.
    This is one component of the ELBO:
    ELBO=expected log-likelihood - sum(kl_terms)
    qpars: a tuple containing (mu_z,Kuu_chol)
    qpars can be obtained by calling self.get_variational_params()
    mu_z is the GP mean function at all inducing points (dimension: LxM)
    Kuu_chol is the cholesky lower triangular of the kernel matrix of all inducing points.
    Its dimension is LxMxM
    where L = number of latent dimensions and M = number of inducing points.
    """
    qu = tfd.MultivariateNormalTriL(loc=self.delta, scale_tril=self.Omega_tril)
    pu = tfd.MultivariateNormalTriL(loc=mu_z, scale_tril=Kuu_chol)
    return qu.kl_divergence(pu) #L-vector

  # @tf.function
  def elbo_avg(self, X, Y, sz=1, S=1, Ntot=None, chol=True):
    """
    Parameters
    ----------
    X : numpy array of spatial coordinates (NxD)
        **OR** a tuple of spatial coordinates, multivariate outcomes,
        and size factors (convenient for minibatches from tensor slices)
    Y : numpy array of multivariate outcomes (NxJ)
        If Y is None then X must be a tuple of length three
    sz : size factors, optional
        vector of length N, typically the rowSums or rowMeans of Y.
        If X is a tuple then this is ignored as sz is expected in the third
        element of the X tuple.
    S : integer, optional
        Number of random GP function evaluations to use. The default is 1.
        Larger S=more accurate approximation to true ELBO but slower
    Ntot : total number of observations in full dataset
        This is needed when X,Y,sz are a minibatch from the full data
        If Ntot is None, we assume X,Y,sz provided are the full data NOT a minibatch.

    Returns
    -------
    The numeric evidence lower bound value, divided by Ntot.
    """
    batch_size, J = Y.shape
    if Ntot is None: Ntot = batch_size #no minibatch, all observations provided
    #print(11)
    ker = self.get_kernel()
    mu_z = self.get_mu_z()
    #print(111)
    Kuu_chol = self.get_Kuu_chol(kernel=ker,from_cache=(not chol))
    #kl_terms is not affected by minibatching so use reduce_sum
    #print(1111)
    kl_term = tf.reduce_sum(self.eval_kl_term(mu_z, Kuu_chol))
    Mu = self.sample_predictive_mean(X, sz=sz, S=S, kernel=ker, mu_z=mu_z, Kuu_chol=Kuu_chol, chol = chol)
    eloglik = likelihoods.lik_to_distr(self.lik, Mu, self.disp).log_prob(Y)
    return J*tf.reduce_mean(eloglik) - kl_term/Ntot

  def elbo_avg_self1(self, X, Y, sz=1, S=1, Ntot=None, chol=True):
    """
    Parameters
    ----------
    X : numpy array of spatial coordinates (NxD)
        **OR** a tuple of spatial coordinates, multivariate outcomes,
        and size factors (convenient for minibatches from tensor slices)
    Y : numpy array of multivariate outcomes (NxJ)
        If Y is None then X must be a tuple of length three
    sz : size factors, optional
        vector of length N, typically the rowSums or rowMeans of Y.
        If X is a tuple then this is ignored as sz is expected in the third
        element of the X tuple.
    S : integer, optional
        Number of random GP function evaluations to use. The default is 1.
        Larger S=more accurate approximation to true ELBO but slower
    Ntot : total number of observations in full dataset
        This is needed when X,Y,sz are a minibatch from the full data
        If Ntot is None, we assume X,Y,sz provided are the full data NOT a minibatch.

    Returns
    -------
    The numeric evidence lower bound value, divided by Ntot.
    """
    batch_size, J = Y.shape
    if Ntot is None: Ntot = batch_size #no minibatch, all observations provided
    ker = self.get_kernel()#not trainable
    mu_z = self.get_mu_z_1()#trainable_
    Kuu_chol = self.get_Kuu_chol(kernel=ker,from_cache=(not chol))#not trainable
    #kl_terms is not affected by minibatching so use reduce_sum
    print("Kuu_chol")
    print(Kuu_chol)
    kl_term = tf.reduce_sum(self.eval_kl_term(mu_z, Kuu_chol))#trainable_
    Mu = self.sample_predictive_mean_1(X, sz=sz, S=S, kernel=ker, mu_z=mu_z, Kuu_chol=Kuu_chol)#trainable_
    eloglik = likelihoods.lik_to_distr(self.lik, Mu, self.disp).log_prob(Y)#not trainable
    return J*tf.reduce_mean(eloglik) - kl_term/Ntot#not trainable




  def train_step(self, list_self, list_D, optimizer, optimizer_k, S=1, Ntot=None, chol=True):
    """
    Executes one training step and returns the loss.
    D is training data: a tensorflow dataset (from slices) of (X,Y,sz)
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    kk=0
    for self_tmp in list_self:
      kk=kk+1
    list_D=[None]*kk
    list_Y=[None]*kk
    with tf.GradientTape(persistent=True) as tape:
      loss = -self.elbo_avg(D["X"], D["Y"], sz=D["sz"], S=S, Ntot=Ntot, chol=chol)
    try:
      gradients = tape.gradient(loss, self.trvars_nonkernel)
      if chol:
        gradients_k = tape.gradient(loss, self.trvars_kernel)
        optimizer_k.apply_gradients(zip(gradients_k, self.trvars_kernel))
      optimizer.apply_gradients(zip(gradients, self.trvars_nonkernel))
    finally:
      del tape
    return loss

  def validation_step(self, D, S=1, chol=False):
    """
    Compute the validation loss on held-out data D
    D is a tensorflow dataset (from slices) of (X,Y,sz)
    """
    return -self.elbo_avg(D["X"], D["Y"], sz=D["sz"], S=S, chol=chol)

  def predict(self, Dtr, Dval=None, S=10):
    """
    Here Dtr,Dval should be raw counts (not normalized or log-transformed)

    returns the predicted training data mean and validation data mean
    on the original count scale
    """
    Mu_tr = misc.t2np(self.sample_predictive_mean(Dtr["X"], sz=Dtr["sz"], S=S))
    if self.lik=="gau":
      sz_tr = Dtr["Y"].sum(axis=1)
      #note self.feature_means is None if self.nonneg=True
      misc.reverse_normalization(Mu_tr, feature_means=self.feature_means,
                            transform=np.expm1, sz=sz_tr, inplace=True)
    if Dval:
      Mu_val = misc.t2np(self.sample_predictive_mean(Dval["X"], sz=Dval["sz"], S=S))
      if self.lik=="gau":
        sz_val = Dval["Y"].sum(axis=1)
        misc.reverse_normalization(Mu_val, feature_means=self.feature_means,
                              transform=np.expm1, sz=sz_val, inplace=True)
    else:
      Mu_val = None
    return Mu_tr,Mu_val

def smooth_spatial_factors(F,Z,list_X,list_Z,X=None):
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
  M=0
  for Z_tmp in list_Z:
    M=M+Z_tmp.shape[0]
  #M = Z.shape[0]
  #M1=Z1.shape[0]
  #MM1=X1.shape[0]
  if list_X is None: #no spatial coordinates, just use the mean
    beta0 = F.mean(axis=0)
    U = np.tile(beta0,[M,1])
    beta = None
  else: #spatial coordinates
    kk=0
    for X_tmp in list_X:
      kk=kk+1 #kk: number of samples
      print(kk)
    MM_=0
    M_=0
    beta_tmp=[]
    beta0_tmp=[]
    U=[]
    beta0_=[]

    print(kk)

    kkk=0
    print(kkk)
    Z_tmp=list_Z[kkk]
    X_tmp=list_X[kkk]
    MM_next=MM_+X_tmp.shape[0]
    lr_tmp = LinearRegression().fit(X_tmp,F[MM_:MM_next,:])
    beta0_tmp = lr_tmp.intercept_
    beta_tmp = lr_tmp.coef_
    nn = max(2, ceil(X_tmp.shape[0]/Z_tmp.shape[0]/2))
    knn_tmp = KNeighborsRegressor(n_neighbors=nn).fit(X_tmp,F[MM_:MM_next,:])
    M_next = M_+Z_tmp.shape[0]
    U_tmp = knn_tmp.predict(Z_tmp[:,:])
    beta0_= beta0_tmp
    beta_= beta_tmp
    U_ = U_tmp
    MM_ = MM_next
    M_ = M_next

    for kkk in range(1,kk):
      print(kkk)
      Z_tmp=list_Z[kkk]
      X_tmp=list_X[kkk]
      MM_next=MM_+X_tmp.shape[0]
      lr_tmp = LinearRegression().fit(X_tmp,F[MM_:MM_next,:])
      beta0_tmp = lr_tmp.intercept_
      beta_tmp = lr_tmp.coef_
      nn = max(2, ceil(X_tmp.shape[0]/Z_tmp.shape[0]/2))
      knn_tmp = KNeighborsRegressor(n_neighbors=nn).fit(X_tmp,F[MM_:MM_next,:])
      M_next = M_+Z_tmp.shape[0]
      U_tmp = knn_tmp.predict(Z_tmp[:,:])
      beta0_= np.concatenate((beta0_, beta0_tmp), axis=0)
      beta_= np.concatenate((beta_, beta_tmp), axis=0)
      U_ = np.concatenate((U_, U_tmp), axis=0)
      MM_ = MM_next
      M_ = M_next
      print(beta0_.shape)
    #lr1 = LinearRegression().fit(X[0:MM1,:],F[0:MM1,:])
    #lr2 = LinearRegression().fit(X[MM1:,:],F[MM1:,:])
    #beta01 = lr1.intercept_
    #beta02 = lr2.intercept_
    #beta1 = lr1.coef_
    #beta2 = lr2.coef_
    #nn = max(2, ceil(X.shape[0]/M/2))
    #nn = max(2, ceil(500/M))
    #beta0 = lr.intercept_
    #beta = lr.coef_
    #nn = max(2, ceil(X.shape[0]/M))
    #knn1 = KNeighborsRegressor(n_neighbors=nn).fit(X[0:MM1,:],F[0:MM1,:])
    #knn2 = KNeighborsRegressor(n_neighbors=nn).fit(X[MM1:,:],F[MM1:,:])
    #U1 = knn1.predict(Z[0:M1,:])
    #U2 = knn2.predict(Z[M1:,:])
    #U = np.concatenate((U1, U2), axis=0)
    #beta0= np.concatenate((beta01, beta02), axis=0)
    #beta= np.concatenate((beta1, beta2), axis=0)
  return U_,beta0_,beta_

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
  L = fit.W.shape[1]
  # M = fit.Z.shape[0] #number of inducing points
  kw = likelihoods.choose_nmf_pars(fit.lik)
  F,W = nnfu.regularized_nmf(Y, L, sz=sz, pseudocount=pseudocount,
                             factors=factors, loadings=loadings,
                             shrinkage=shrinkage, **kw)
  # eF = factors
  # W = loadings
  # if eF is None or W is None:
  #   kw = likelihoods.choose_nmf_pars(fit.lik)
  #   nmf = NMF(L,**kw)
  #   eF = nmf.fit_transform(Y)#/sz
  #   W = nmf.components_.T
  # W = postprocess.shrink_loadings(W, shrinkage=shrinkage)
  # wsum = W.sum(axis=0)
  # eF = postprocess.shrink_factors(eF*wsum, shrinkage=shrinkage)
  # F = np.log(pseudocount+eF)-np.log(sz)
  # beta0 = fit.beta0.numpy().flatten()
  # wt_to_W = F.mean(axis=0)- beta0
  # F-= wt_to_W
  # W,wsum = normalize_cols(W)
  # eF *= wsum
  # eFm2 = eF.mean()/2
  # eF/=eFm2
  # W*=eFm2
  # F = np.log(pseudocount+eF)
  fit.set_loadings(W)
  print('fit_set_loadings')
  U,beta0,beta = smooth_spatial_factors(F,fit.Z.numpy(), list_X, list_Z,X=X)###---
  # if X is None: #no spatial coordinates, just use the mean
  #   beta0 = F.mean(axis=0)
  #   U = np.tile(beta0,[M,1])
  # else: #spatial coordinates
  #   lr = LinearRegression().fit(X,F)
  #   beta0 = lr.intercept_
  #   fit.beta.assign(lr.coef_,read_value=False)
  #   nn = max(2, ceil(X.shape[0]/M))
  #   knn = KNeighborsRegressor(n_neighbors=nn).fit(X,F)
  #   U = knn.predict(fit.Z.numpy())
  fit.beta0.assign(beta0[:,None],read_value=False)
  fit.delta.assign(U.T,read_value=False)
  if beta is not None: fit.beta.assign(beta,read_value=False)


#fit1=pf.ProcessFactorization(J,L,Z,psd_kernel=ker,nonneg=True,lik="poi")
#fit2=fit1
#self1_0=fit1
#self2_0=fit2

