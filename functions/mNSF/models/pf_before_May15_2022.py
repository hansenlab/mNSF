#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(spatial/temporal) Process Factorization base class

@author: townesf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from os import path
from math import ceil
from tensorflow import linalg as tfl
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from models import likelihoods
from utils import misc, nnfu
tfd = tfp.distributions
tfb = tfp.bijectors
tv = tfp.util.TransformedVariable
tfk = tfp.math.psd_kernels

dtp = "float32"
rng = np.random.default_rng()

class ProcessFactorization(tf.Module):
  def __init__(self, J, L, Z, lik="poi", psd_kernel=tfk.MaternThreeHalves,
               nugget=1e-5, length_scale=0.1, disp="default",
               nonneg=False, isotropic=True, feature_means=None, **kwargs):
    """
    Non-negative process factorization

    Parameters
    ----------
    J : integer scalar
        Number of features in the multivariate outcome.
    L : integer scalar
        Number of desired latent Gaussian processes.
    T : integer scalar
        Number of desired non-spatial latent factors
    Z : 2D numpy array
        Coordinates of inducing point locations in input space.
        First dimension: 'M' is number of inducing points.
        Second dimension: 'D' is dimensionality of input to GP.
        More inducing points= slower but more accurate inference.
    lik: likelihood (Poisson or Gaussian)
    disp: overdispersion parameter initialization.
    --for Gaussian likelihood, the scale (stdev) parameter
    --for negative binomial likelihood, the parameter phi such that
      var=mean+phi*mean^2, i.e. phi->0 corresponds to Poisson, phi=1 to geometric
    --for Poisson likelihood, this parameter is ignored and set to None
    psd_kernel : an object of class PositiveSemidefiniteKernel, must accept a
        length_scale parameter.
    feature_means : if input data is centered (for lik='gau', nonneg=False)
    """
    super().__init__(**kwargs)
    # self.is_spatial=True
    self.lik = lik
    self.isotropic=isotropic
    M,D = Z.shape
    self.Z = tf.Variable(Z, trainable=False, dtype=dtp, name="inducing_points")
    self.nonneg = tf.Variable(nonneg, trainable=False, name="is_non_negative")
    #variational parameters
    with tf.name_scope("variational"):
      self.delta = tf.Variable(rng.normal(size=(L,M)), dtype=dtp, name="mean") #LxM
      _Omega_tril = self._init_Omega_tril(L,M,nugget=nugget)
      # _Omega_tril = .01*tf.eye(M,batch_shape=[L],dtype=dtp)
      self.Omega_tril=tv(_Omega_tril, tfb.FillScaleTriL(), dtype=dtp, name="covar_tril") #LxMxM
    #GP hyperparameters
    with tf.name_scope("gp_mean"):
      if self.nonneg:
        prior_mu, prior_sigma = misc.lnormal_approx_dirichlet(max(L,1.1))
        self.beta0 = tf.Variable(prior_mu*tf.ones((L,1)), dtype=dtp,
                                 name="intercepts")
      else:
        self.beta0 = tf.Variable(tf.zeros((L,1)), dtype=dtp, name="intercepts") #L
      self.beta = tf.Variable(tf.zeros((L,D)), dtype=dtp, name="slopes") #LxD
    with tf.name_scope("gp_kernel"):
      self.nugget = tf.Variable(nugget, dtype=dtp, trainable=False, name="nugget")
      self.amplitude = tv(np.tile(1.0,[L]), tfb.Softplus(), dtype=dtp, name="amplitude")
      self._ls0 = length_scale #store for .reset() method
      if self.isotropic:
        self.length_scale = tv(np.tile(self._ls0,[L]), tfb.Softplus(),
                               dtype=dtp, name="length_scale")
      else:
        self.scale_diag = tv(np.tile(np.sqrt(self._ls0),[L,D]),
                             tfb.Softplus(), dtype=dtp, name="scale_diag")
    #Loadings weights
    if self.nonneg:
      self.W = tf.Variable(rng.exponential(size=(J,L)), dtype=dtp,
                           constraint=misc.make_nonneg, name="loadings") #JxL
    else:
      self.W = tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings") #JxL
    self.psd_kernel = psd_kernel #this is a class, not yet an object
    #likelihood parameters, set defaults
    self._disp0 = disp
    self._init_misc()
    self.Kuu_chol = tf.Variable(self.eval_Kuu_chol(self.get_kernel()), dtype=dtp, trainable=False)
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
    misc initialization shared between __init__ and reset
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
    return self.W.shape[1]

  def get_loadings(self):
    return self.W.numpy()

  def set_loadings(self,Wnew):
    self.W.assign(Wnew,read_value=False)

  def init_loadings(self,Y,X=None,sz=1,**kwargs):
    """
    Use either PCA or NMF to initialize the loadings matrix from data Y
    """
    if self.nonneg:
      init_npf_with_nmf(self,Y,X=X,sz=sz,**kwargs)
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

  def get_Kuu_chol(self, kernel=None, from_cache=False):
    if not from_cache:
      Kuu_chol = self.eval_Kuu_chol(kernel=kernel)
      self.Kuu_chol.assign(Kuu_chol) #update cache
      return Kuu_chol
    else: #use cached variable, gradients cannot propagate to kernel hps
      return self.Kuu_chol

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
    N = X.shape[0]
    L = self.W.shape[1]
    mu_x = self.beta0+tfl.matmul(self.beta, X, transpose_b=True) #LxN
    Kuf = kernel.matrix(self.Z, X) #LxMxN
    Kff_diag = kernel.apply(X, X, example_ndims=1)+self.nugget #LxN
    alpha_x = tfl.cholesky_solve(Kuu_chol, Kuf) #LxMxN
    mu_tilde = mu_x + tfl.matvec(alpha_x, self.delta-mu_z, transpose_a=True) #LxN
    #compute the alpha(x_i)'(K_uu-Omega)alpha(x_i) term
    a_t_Kchol = tfl.matmul(alpha_x, Kuu_chol, transpose_a=True) #LxNxM
    aKa = tf.reduce_sum(tf.square(a_t_Kchol), axis=2) #LxN
    a_t_Omega_tril = tfl.matmul(alpha_x, self.Omega_tril, transpose_a=True) #LxNxM
    aOmega_a = tf.reduce_sum(tf.square(a_t_Omega_tril), axis=2) #LxN
    Sigma_tilde = Kff_diag - aKa + aOmega_a #LxN
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
    Mu = self.sample_predictive_mean(X, sz=sz, S=S, kernel=ker, mu_z=mu_z, Kuu_chol=Kuu_chol)
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
    kl_term = tf.reduce_sum(self.eval_kl_term(mu_z, Kuu_chol))#trainable_
    Mu = self.sample_predictive_mean_1(X, sz=sz, S=S, kernel=ker, mu_z=mu_z, Kuu_chol=Kuu_chol)#trainable_
    eloglik = likelihoods.lik_to_distr(self.lik, Mu, self.disp).log_prob(Y)#not trainable
    return J*tf.reduce_mean(eloglik) - kl_term/Ntot#not trainable




  def train_step(self, D, optimizer, optimizer_k, S=1, Ntot=None, chol=True):
    """
    Executes one training step and returns the loss.
    D is training data: a tensorflow dataset (from slices) of (X,Y,sz)
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
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

def smooth_spatial_factors(F,Z,X=None):
  """
  F: real-valued factors (ie on the log scale for NPF)
  Z: inducing point locations
  X: spatial coordinates
  """
  M = Z.shape[0]
  if X is None: #no spatial coordinates, just use the mean
    beta0 = F.mean(axis=0)
    U = np.tile(beta0,[M,1])
    beta = None
  else: #spatial coordinates
    lr = LinearRegression().fit(X,F)
    beta0 = lr.intercept_
    beta = lr.coef_
    nn = max(2, ceil(X.shape[0]/M))
    knn = KNeighborsRegressor(n_neighbors=nn).fit(X,F)
    U = knn.predict(Z)
  return U,beta0,beta

def init_npf_with_nmf(fit, Y, X=None, sz=1, pseudocount=1e-2, factors=None,
                      loadings=None, shrinkage=0.2):
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
  # W*= np.exp(wt_to_W-np.log(wsum))
  # W,wsum = normalize_cols(W)
  # eF *= wsum
  # eFm2 = eF.mean()/2
  # eF/=eFm2
  # W*=eFm2
  # F = np.log(pseudocount+eF)
  fit.set_loadings(W)
  U,beta0,beta = smooth_spatial_factors(F,fit.Z.numpy(),X=X)
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



class ProcessFactorization_self12(tf.Module):
  def __init__(self, self1_0, self2_0, J, L, Z1, Z2, lik="poi", psd_kernel=tfk.MaternThreeHalves, 
               nugget=1e-5, length_scale=0.1, disp="default",
               nonneg=False, isotropic=True, feature_means=None, **kwargs):
    """
    Non-negative process factorization

    Parameters
    ----------
    J : integer scalar
        Number of features in the multivariate outcome.
    L : integer scalar
        Number of desired latent Gaussian processes.
    T : integer scalar
        Number of desired non-spatial latent factors
    Z : 2D numpy array
        Coordinates of inducing point locations in input space.
        First dimension: 'M' is number of inducing points.
        Second dimension: 'D' is dimensionality of input to GP.
        More inducing points= slower but more accurate inference.
    lik: likelihood (Poisson or Gaussian)
    disp: overdispersion parameter initialization.
    --for Gaussian likelihood, the scale (stdev) parameter
    --for negative binomial likelihood, the parameter phi such that
      var=mean+phi*mean^2, i.e. phi->0 corresponds to Poisson, phi=1 to geometric
    --for Poisson likelihood, this parameter is ignored and set to None
    psd_kernel : an object of class PositiveSemidefiniteKernel, must accept a
        length_scale parameter.
    feature_means : if input data is centered (for lik='gau', nonneg=False)
    """
    ## The __init__ function is called every time an object is created from a class
    super().__init__(**kwargs)
    # self.is_spatial=True
    self.lik = lik #shared by rep1 and rep2
    self.isotropic=isotropic #shared by rep1 and rep2
    #self.self1_0=self1_0 ###????
    #self.self2_0=self2_0 ###????

    ##### data
    ## data - rep1 
    M1,D1 = Z1.shape
    #print(M1)
    self.Z1 = tf.Variable(Z1, trainable=False, dtype=dtp, name="inducing_points_1")
    self.nonneg = tf.Variable(nonneg, trainable=False, name="is_non_negative_1") #shared by rep1 and rep2 
    ## data - rep12
    M2,D2 = Z2.shape#(144, 2)
    self.Z2 = tf.Variable(Z2, trainable=False, dtype=dtp, name="inducing_points_2")
    self.nonneg = tf.Variable(nonneg, trainable=False, name="is_non_negative_2") #shared by rep1 and rep2 
    ##### variational parameters
    with tf.name_scope("variational"):
          # variational parameters -rep1
      self.delta1 = tf.Variable(rng.normal(size=(L,M1)), dtype=dtp, name="mean_1") #LxM
      _Omega_tril1 = self._init_Omega_tril(L=L,M=M1)
      #_Omega_tril1 = self._init_Omega_tril(L,M1,nugget=nugget)
      # _Omega_tril = .01*tf.eye(M,batch_shape=[L],dtype=dtp)
      self.Omega_tril1=tv(_Omega_tril1, tfb.FillScaleTriL(), dtype=dtp, name="covar_tril_1") #LxMxM
    #variational parameters-rep2
    #with tf.name_scope("variational"):
      self.delta2 = tf.Variable(rng.normal(size=(L,M2)), dtype=dtp, name="mean_2") #LxM
      _Omega_tril2 = self._init_Omega_tril(L,M2)
      #_Omega_tril2 = self._init_Omega_tril(L,M2,nugget=nugget)
      # _Omega_tril = .01*tf.eye(M,batch_shape=[L],dtype=dtp)
      self.Omega_tril2=tv(_Omega_tril2, tfb.FillScaleTriL(), dtype=dtp, name="covar_tril_2") #LxMxM
    ###GP hyperparameters
    with tf.name_scope("gp_mean"):
      if self.nonneg:
        prior_mu, prior_sigma = misc.lnormal_approx_dirichlet(max(L,1.1))
         #rep1
        self.beta0_1 = tf.Variable(prior_mu*tf.ones((L,1)), dtype=dtp,
                                 name="intercepts_1")
         #rep2
        self.beta0_2 = tf.Variable(prior_mu*tf.ones((L,1)), dtype=dtp,
                                 name="intercepts_2")
      else:
        #rep1
        self.beta0_1 = tf.Variable(tf.zeros((L,1)), dtype=dtp, name="intercepts_1") #L
        #rep2
        self.beta0_2 = tf.Variable(tf.zeros((L,1)), dtype=dtp, name="intercepts_2") #L
      #rep1
      self.beta_1 = tf.Variable(tf.zeros((L,D1)), dtype=dtp, name="slopes_1") #LxD
      #rep2
      self.beta_2 = tf.Variable(tf.zeros((L,D2)), dtype=dtp, name="slopes_2") #LxD
    # gp_kernel
    with tf.name_scope("gp_kernel"):
      self.nugget = tf.Variable(nugget, dtype=dtp, trainable=False, name="nugget")
      self.amplitude1 = tv(np.tile(1.0,[L]), tfb.Softplus(), dtype=dtp, name="amplitude_1")
      self.amplitude2 = tv(np.tile(1.0,[L]), tfb.Softplus(), dtype=dtp, name="amplitude_2")
      self._ls0 = length_scale #store for .reset() method
      if self.isotropic:
        self.length_scale1 = tv(np.tile(self._ls0,[L]), tfb.Softplus(),
                               dtype=dtp, name="length_scale_1")
        self.length_scale2 = tv(np.tile(self._ls0,[L]), tfb.Softplus(),
                               dtype=dtp, name="length_scale_2")
      else:
        #rep1
        self.scale_diag1 = tv(np.tile(np.sqrt(self._ls0),[L,D1]),
                             tfb.Softplus(), dtype=dtp, name="scale_diag_1")
        #rep2
        self.scale_diag2 = tv(np.tile(np.sqrt(self._ls0),[L,D2]),
                             tfb.Softplus(), dtype=dtp, name="scale_diag_2")
    #Loadings weights - shared by rep1 and rep2
    if self.nonneg:
      self.W = tf.Variable(rng.exponential(size=(J,L)), dtype=dtp,
                           constraint=misc.make_nonneg, name="loadings") #JxL
    else:
      self.W = tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings") #JxL
    self.psd_kernel = psd_kernel #this is a class, not yet an object
    #likelihood parameters, set defaults
    self._disp0 = disp
    self._init_misc()###>>>
    #rep1
    self.Kuu_chol1 = tf.Variable(self1_0.eval_Kuu_chol(self1_0.get_kernel()), dtype=dtp, trainable=False)
    #rep2
    self.Kuu_chol2 = tf.Variable(self2_0.eval_Kuu_chol(self2_0.get_kernel()), dtype=dtp, trainable=False)
    if self.lik=="gau" and not self.nonneg:
      self.feature_means = feature_means
    else:
      self.feature_means = None

  @staticmethod
  def _init_Omega_tril(L, M):
  #def _init_Omega_tril(L, M):
  #def _init_Omega_tril(L, M, nugget=None):
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
    misc initialization shared between __init__ and reset
    """
    J = self.W.shape[0]
    self.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)
    #stuff to facilitate caching
    self.trvars_kernel = tuple(i for i in self.trainable_variables if i.name[:10]=="gp_kernel/")
    self.trvars_nonkernel = tuple(i for i in self.trainable_variables if i.name[:10]!="gp_kernel/")
    if self.isotropic:
      self.kernel1 = self.psd_kernel(amplitude=self.amplitude1, length_scale=self.length_scale1)
      self.kernel2 = self.psd_kernel(amplitude=self.amplitude2, length_scale=self.length_scale2)
    #else:
    #  self.kernel = tfk.FeatureScaled(self.psd_kernel(amplitude1=self.amplitude1, amplitude2=self.amplitude2), self.scale_diag1, self.scale_diag2)

  def get_dims(self):
    return self.W.shape[1]

  def get_loadings(self):
    return self.W.numpy()

  def set_loadings(self,Wnew):
    self.W.assign(Wnew,read_value=False)

  def init_loadings(self,Y,X=None,sz=1,**kwargs):
    """
    Use either PCA or NMF to initialize the loadings matrix from data Y
    """
    if self.nonneg:
      init_npf_with_nmf(self,Y,X=X,sz=sz,**kwargs)
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
      eval_Kuu_chol what type of size factors are used (eg 'none' or 'scanpy').
    base : str, optional
      Parent directory for saving pickle files. The default is cwd.
    """
    pars = {"L":self.W.shape[1], "lik":self.lik, "sz":sz,
            "model":"NPF" if self.nonneval_Kuu_choleg else "RPF",
            "kernel":self.psd_kernel.__name__,
            "M":self.Z.shape[0]
            }
    pth = misc.params2key(pars)
    if base: pth = path.join(base,pth)
    return pth

  def get_kernel(self):
    return self.kernel

  def get_kernel_1(self):
    return self.kernel1

  def get_kernel_2(self):
    return self.kernel2

  def eval_Kuu_chol(self, kernel=None):
    if kernel is None:
      kernel = self.get_kernel()
    M,D = self.Z.shape
    Kuu = kernel.matrix(self.Z, self.Z) + self.nugget*tf.eye(M)
    #Kuu_chol is LxMxM and lower triangular in last two dims
    return tfl.cholesky(Kuu)

  def eval_Kuu_chol_rep1(self, kernel=None):
    if kernel is None:
      kernel = self.get_kernel_1()
    M1,D1 = self.Z1.shape
    Kuu1 = kernel.matrix(self.Z1, self.Z1) + self.nugget*tf.eye(M1)
    #Kuu_chol is LxMxM and lower triangular in last two dims
    return tfl.cholesky(Kuu1)

  def eval_Kuu_chol_rep2(self, kernel=None):
    if kernel is None:
      kernel = self.get_kernel_2()
    M2,D2 = self.Z2.shape
    Kuu2 = kernel.matrix(self.Z2, self.Z2) + self.nugget*tf.eye(M2)
    #Kuu_chol is LxMxM and lower triangular in last two dims
    return tfl.cholesky(Kuu2)

  def get_Kuu_chol(self, kernel=None, from_cache=False):
    if not from_cache:
      Kuu_chol = self.eval_Kuu_chol(kernel=kernel)
      self.Kuu_chol.assign(Kuu_chol) #update cache
      return Kuu_chol
    else: #use cached variable, gradients cannot propagate to kernel hps
      return self.Kuu_chol

  def get_Kuu_chol_1(self, kernel=None, from_cache=False):
    #if not from_cache:
    Kuu_chol = self.eval_Kuu_chol_rep1(kernel=kernel)
    self.Kuu_chol1.assign(Kuu_chol) #update cache
    return Kuu_chol
    #else: #use cached variable, gradients cannot propagate to kernel hps
    #  return self.Kuu_chol

  def get_Kuu_chol_2(self, kernel=None, from_cache=False):
    #if not from_cache:
    Kuu_chol = self.eval_Kuu_chol_rep2(kernel=kernel)
    self.Kuu_chol2.assign(Kuu_chol) #update cache
    return Kuu_chol
    #else: #use cached variable, gradients cannot propagate to kernel hps
    #  return self.Kuu_chol

  def get_mu_z(self):
    return self.beta0+tfl.matmul(self.beta, self.Z, transpose_b=True) #LxM


  def get_mu_z_1(self):
    #print("get_mu_z_1-start")
    #print(self.beta_1[1])# the actual beta_1 is not saved into self
    #print(self.Z_1[1,1])
    #print(self.beta0_1)
    #print("get_mu_z_1-end")
    return self.beta0_1+tfl.matmul(self.beta_1, self.Z_1, transpose_b=True) #LxM

  def get_mu_z_2(self):
    return self.beta0_2+tfl.matmul(self.beta_2, self.Z_2, transpose_b=True) #LxM


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
    N = X.shape[0]
    L = self.W.shape[1]
    mu_x = self.beta0+tfl.matmul(self.beta, X, transpose_b=True) #LxN
    Kuf = kernel.matrix(self.Z, X) #LxMxN
    Kff_diag = kernel.apply(X, X, example_ndims=1)+self.nugget #LxN
    alpha_x = tfl.cholesky_solve(Kuu_chol, Kuf) #LxMxN
    mu_tilde = mu_x + tfl.matvec(alpha_x, self.delta-mu_z, transpose_a=True) #LxN
    #compute the alpha(x_i)'(K_uu-Omega)alpha(x_i) term
    a_t_Kchol = tfl.matmul(alpha_x, Kuu_chol, transpose_a=True) #LxNxM
    aKa = tf.reduce_sum(tf.square(a_t_Kchol), axis=2) #LxN
    a_t_Omega_tril = tfl.matmul(alpha_x, self.Omega_tril, transpose_a=True) #LxNxM
    aOmega_a = tf.reduce_sum(tf.square(a_t_Omega_tril), axis=2) #LxN
    Sigma_tilde = Kff_diag - aKa + aOmega_a #LxN
    eps = tf.random.normal((S,L,N)) #note this is not the same random generator as self.rng!
    return mu_tilde + tf.math.sqrt(Sigma_tilde)*eps #"F", dims: SxLxN


  def sample_latent_GP_funcs_1(self, X, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
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
      kernel = self.get_kernel_1()
    if mu_z is None:
      mu_z = self.get_mu_z_1()
    if Kuu_chol is None:
      Kuu_chol = self.get_Kuu_chol_1(kernel=kernel, from_cache=(not chol))
    N = X.shape[0]
    L = self.W.shape[1]
    mu_x = self.beta0_1+tfl.matmul(self.beta_1, X, transpose_b=True) #LxN
    Kuf = kernel.matrix(self.Z1, X) #LxMxN
    Kff_diag = kernel.apply(X, X, example_ndims=1)+self.nugget #LxN
    alpha_x = tfl.cholesky_solve(Kuu_chol, Kuf) #LxMxN
    mu_tilde = mu_x + tfl.matvec(alpha_x, self.delta1-mu_z, transpose_a=True) #LxN
    #compute the alpha(x_i)'(K_uu-Omega)alpha(x_i) term
    a_t_Kchol = tfl.matmul(alpha_x, Kuu_chol, transpose_a=True) #LxNxM
    aKa = tf.reduce_sum(tf.square(a_t_Kchol), axis=2) #LxN
    a_t_Omega_tril = tfl.matmul(alpha_x, self.Omega_tril1, transpose_a=True) #LxNxM
    aOmega_a = tf.reduce_sum(tf.square(a_t_Omega_tril), axis=2) #LxN
    Sigma_tilde = Kff_diag - aKa + aOmega_a #LxN
    eps = tf.random.normal((S,L,N)) #note this is not the same random generator as self.rng!
    return mu_tilde + tf.math.sqrt(Sigma_tilde)*eps #"F", dims: SxLxN


  def sample_latent_GP_funcs_2(self, X, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
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
      kernel = self.get_kernel_2()
    if mu_z is None:
      mu_z = self.get_mu_z_2()
    if Kuu_chol is None:
      Kuu_chol = self.get_Kuu_chol_2(kernel=kernel, from_cache=(not chol))
    N = X.shape[0]
    L = self.W.shape[1]
    mu_x = self.beta0_2+tfl.matmul(self.beta_2, X, transpose_b=True) #LxN
    Kuf = kernel.matrix(self.Z2, X) #LxMxN
    Kff_diag = kernel.apply(X, X, example_ndims=1)+self.nugget #LxN
    alpha_x = tfl.cholesky_solve(Kuu_chol, Kuf) #LxMxN
    mu_tilde = mu_x + tfl.matvec(alpha_x, self.delta2-mu_z, transpose_a=True) #LxN
    #compute the alpha(x_i)'(K_uu-Omega)alpha(x_i) term
    a_t_Kchol = tfl.matmul(alpha_x, Kuu_chol, transpose_a=True) #LxNxM
    aKa = tf.reduce_sum(tf.square(a_t_Kchol), axis=2) #LxN
    a_t_Omega_tril = tfl.matmul(alpha_x, self.Omega_tril2, transpose_a=True) #LxNxM
    aOmega_a = tf.reduce_sum(tf.square(a_t_Omega_tril), axis=2) #LxN
    Sigma_tilde = Kff_diag - aKa + aOmega_a #LxN
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

  def sample_predictive_mean_1(self, X, sz=1, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
    """
    See sample_latent_variational_GP_funcs for X,S definitions
    sz is a tensor of shape (N,1) of size factors.
    Typically sz would be the rowSums or rowMeans of the outcome matrix Y.
    """
    F = self.sample_latent_GP_funcs_1(X, S=S, kernel=kernel, mu_z=mu_z,
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

  def sample_predictive_mean_2(self, X, sz=1, S=1, kernel=None, mu_z=None, Kuu_chol=None, chol=True):
    """
    See sample_latent_variational_GP_funcs for X,S definitions
    sz is a tensor of shape (N,1) of size factors.
    Typically sz would be the rowSums or rowMeans of the outcome matrix Y.
    """
    F = self.sample_latent_GP_funcs_2(X, S=S, kernel=kernel, mu_z=mu_z,
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

  def eval_kl_term_1(self, mu_z, Kuu_chol):
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
    qu = tfd.MultivariateNormalTriL(loc=self.delta1, scale_tril=self.Omega_tril1)
    pu = tfd.MultivariateNormalTriL(loc=mu_z, scale_tril=Kuu_chol)
    return qu.kl_divergence(pu) #L-vector

  def eval_kl_term_2(self, mu_z, Kuu_chol):
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
    qu = tfd.MultivariateNormalTriL(loc=self.delta2, scale_tril=self.Omega_tril2)
    pu = tfd.MultivariateNormalTriL(loc=mu_z, scale_tril=Kuu_chol)
    return qu.kl_divergence(pu) #L-vector



  # @tf.function
  def elbo_avg(self, X1, X2, Y1, Y2, sz1=1, sz2=1, S=1, Ntot1=None, Ntot2=None, chol=True):#Ntot1!!,Ntot2!!
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
    #batch_size
    #sz1=1
    #sz2=1
    J1 = Y1.shape[1]
    J2 = Y2.shape[1]
    #if Ntot is None: Ntot = batch_size #no minibatch, all observations provided
    ker1 = self.get_kernel_1()
    ker2 = self.get_kernel_2()
    mu_z1 = self.get_mu_z_1()
    mu_z2 = self.get_mu_z_2()
    Kuu_chol1 = self.get_Kuu_chol_1(kernel=ker1,from_cache=(not chol))
    Kuu_chol2 = self.get_Kuu_chol_2(kernel=ker2,from_cache=(not chol))
    #print(1111)
    #kl_terms is not affected by minibatching so use reduce_sum
    kl_term1 = tf.reduce_sum(self.eval_kl_term_1(mu_z1, Kuu_chol1)) 
    kl_term2 = tf.reduce_sum(self.eval_kl_term_2(mu_z2, Kuu_chol2))
    Mu1 = self.sample_predictive_mean_1(X1, sz=sz1, S=S, kernel=ker1, mu_z=mu_z1, Kuu_chol=Kuu_chol1)
    Mu2 = self.sample_predictive_mean_2(X2, sz=sz2, S=S, kernel=ker2, mu_z=mu_z2, Kuu_chol=Kuu_chol2)
    #print(1111)
    #print(mu_z1)
    eloglik1 = likelihoods.lik_to_distr(self.lik, Mu1, 0).log_prob(Y1)##!!!!let self.disp=0
    eloglik2 = likelihoods.lik_to_distr(self.lik, Mu2, 0).log_prob(Y2)##!!!!let self.disp=0
    #eloglik2 = likelihoods.lik_to_distr(self.lik, Mu2, self.disp).log_prob(Y2)
    #print(1111)
    #print(11111)
    #print(kl_term1)
    #print(Ntot1)##!!!!
    #print(eloglik2[1])
    #print(tf.reduce_mean(eloglik2))##!!!!
    #return tf.reduce_mean(eloglik2)
    #Ntot1=Ntot
    #Ntot2=Ntot1
    #return J1*tf.reduce_mean(eloglik1) + J2*tf.reduce_mean(eloglik2) - kl_term1/Ntot1 -kl_term2/Ntot2 #!!!!
    #return J1*tf.reduce_mean(eloglik1) + J2*tf.reduce_mean(eloglik2) - kl_term1/144 -kl_term2/144
    loss_=J1*tf.reduce_mean(eloglik1) - kl_term1/900 + J2*tf.reduce_mean(eloglik2) -kl_term2/900
    return loss_

  #def elbo_avg_self12(self,X1, X2, Y1, Y2, sz1, sz2, self1_0, self2_0, S=1, Ntot=None, chol=True):
  #  self1,self2=self.split_self12(self1_0, self2_0,L=4)
  #  return self1.elbo_avg(X1, Y1, sz=sz1, S=S, Ntot=Ntot, chol=chol) + self2.elbo_avg(X2, Y2, sz=sz2, S=S, Ntot=Ntot, chol=chol)



  def train_step(self, D1, D2, optimizer, optimizer_k, S=1, Ntot=None, chol=True):
    """
    Executes one training step and returns the loss.
    D is training data: a tensorflow dataset (from slices) of (X,Y,sz)
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    #self1,self2=self.split_self12(self1=self.self1_0,self2=self.self2_0,L=4)
    with tf.GradientTape(persistent=True) as tape:
      #loss1 = -self1.elbo_avg(D1["X"], D1["Y"], sz=D1["sz"], S=S, Ntot=Ntot, chol=chol)
      #loss2 = -self2.elbo_avg(D2["X"], D2["Y"], sz=D2["sz"], S=S, Ntot=Ntot, chol=chol)
      #def elbo_avg(self, X1, X2, Y1, Y2, sz1=1, sz2=1, S=1, Ntot1=None, Ntot2=None, chol=True):
      #loss12= self.elbo_avg(D1["X"],D2["X"], D1["Y"], D2["Y"], sz1=D1["sz"], sz2=D2["sz"], S=S, Ntot1=Ntot, Ntot2=Ntot, chol=chol)
      loss12 = -self.elbo_avg(D1["X"],D2["X"], D1["Y"], D2["Y"], sz1=D1["sz"], sz2=D2["sz"], S=S, Ntot1=Ntot, Ntot2=Ntot, chol=chol)#!!!!
    try:#!!!!!
      print(0)
      gradients = tape.gradient(loss12, self.trvars_nonkernel)
      if chol:
        gradients_k = tape.gradient(loss12, self.trvars_kernel)
        optimizer_k.apply_gradients(zip(gradients_k, self.trvars_kernel))
        optimizer.apply_gradients(zip(gradients, self.trvars_nonkernel))
    finally:
      del tape
    print(99999)
    print(loss12)
    print(99999)
    return loss12

#for i in self.trvars_nonkernel:
#  print(i.name)


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


def split_self12(self,self1,self2,L):
  ##### variational parameters
  self_1=self1
  self_2=self2
  #with tf.name_scope("variational"):
  #  # variational parameters -rep1
  #  self_1.delta.assign(self.delta1.numpy)#=tf.Variable(self.delta1,name="mean")
  #  #self_1.delta=tf.Variable(self.delta1,name="mean")
  #  #self_1.Omega_tril.assign(self.Omega_tril1,read_value=False)
  #  # variational parameters -rep2
  #  self_2.delta.assign(self.delta2.numpy)#=tf.Variable(self.delta1,name="mean")
  #  #self_2.delta=tf.Variable(self.delta2,name="mean")
  #  #self_2.Omega_tril=self.Omega_tril2
  #  #self_2.Omega_tril.assign(self.Omega_tril2,read_value=False)

  ###GP hyperparameters
  with tf.name_scope("gp_mean"):
    if self.nonneg:
      prior_mu, prior_sigma = misc.lnormal_approx_dirichlet(max(L,1.1))
      #rep1
      self_1.beta0 = tf.Variable(self.beta0_1,name="intercepts")
      #self_1.beta0.assign(self.beta0_1.numpy)
      #tf.Variable(rng.normal(size=(J,L)), dtype=dtp, name="loadings") 
      #rep2
      self_2.beta0 = tf.Variable(self.beta0_2,name="intercepts")
      #self_2.beta0.assign(self.beta0_2.numpy)
    else:
      #rep1
      self_1.beta0.assign(self.beta0_1.numpy)
      #rep2
      self_2.beta0.assign(self.beta0_2.numpy)
    #rep1
    self_1.beta=tf.Variable(self.beta_1.numpy,name="slopes")
    #self_1.beta.assign(self.beta_1.numpy)
    #rep2
    self_2.beta=tf.Variable(self.beta_2.numpy,name="slopes")
    #self_2.beta.assign(self.beta_2.numpy)
  # gp_kernel
  with tf.name_scope("gp_kernel"):
    #self.nugget = tf.Variable(nugget, dtype=dtp, trainable=False, name="nugget")
    self_1.amplitude = tv(self.amplitude1, tfb.Softplus(), dtype=dtp, name="amplitude")
    self_2.amplitude = tv(self.amplitude2, tfb.Softplus(), dtype=dtp, name="amplitude")
    #self_1.amplitude.assign(self.amplitude1.numpy)
    #self_2.amplitude.assign(self.amplitude2.numpy)
    #self._ls0 = length_scale #store for .reset() method
    if self.isotropic:
      #self_1.length_scale.assign(self.amplitude1.numpy)
      #self_1.length_scale.assign(self.amplitude2.numpy)
      self1.length_scale = tv(nself.length_scale1.numpy, name="length_scale")
      self_2.length_scale = tv(nself.length_scale2.numpy, name="length_scale")
    else:
      self_1.length_scale.assign(self.amplitude1.numpy)
      self_2.length_scale.assign(self.amplitude2.numpy)
  #Loadings weights - shared by rep1 and rep2
  if self.nonneg:
    #rep1
    #self_1.W.assign(self.W.numpy ) #JxL
    self_1.W = tf.Variable(self.W, name="loadings")
    #rep2
    self_2.W = tf.Variable(self.W, name="loadings")
  else:
    #rep1
    self_1.W.assign(self.W.numpy ) #JxL
    #rep2
    self_2.W.assign(self.W.numpy ) #JxL
  #self.psd_kernel = psd_kernel #this is a class, not yet an object
  #likelihood parameters, set defaults
  #self._disp0 = disp
  #self._init_misc()
  #rep1
  #self1.Kuu_chol = self.Kuu_chol1
  #rep2
  #self2.Kuu_chol = self.Kuu_chol2
  #if self.lik=="gau" and not self.nonneg:
  #  self.feature_means = feature_means
  #else:
  #  self.feature_means = None
  #J = self.W.shape[0]
  #self1.disp = likelihoods.init_lik(self.lik, J, disp=self._disp0, dtp=dtp)#?
  #stuff to facilitate caching
  #self1.trvars_kernel = tuple(i for i in self.trainable_variables if i.name[:10]=="gp_kernel/")
  #self.trvars_nonkernel = tuple(i for i in self.trainable_variables if i.name[:10]!="gp_kernel/")
  self_1.trvars_kernel = tuple(i for i in self_1.trainable_variables if i.name[:10]=="gp_kernel/")
  self_2.trvars_nonkernel = tuple(i for i in self_2.trainable_variables if i.name[:10]!="gp_kernel/")
  if self.isotropic:
    self_1.kernel = self.psd_kernel(amplitude=self_1.amplitude, length_scale=self_1.length_scale)
    self_2.kernel = self.psd_kernel(amplitude=self_2.amplitude, length_scale=self_2.length_scale)
  return self_1,self_2

  #def elbo_avg_self12(self,X1, X2, Y1, Y2, sz1, sz2, self1_0, self2_0, S=1, Ntot=None, chol=True):
  #  self1,self2=self.split_self12(self1_0, self2_0,L=4)
  #  return self1.elbo_avg(X1, Y1, sz=sz1, S=S, Ntot=Ntot, chol=chol) + self2.elbo_avg(X2, Y2, sz=sz2, S=S, Ntot=Ntot, chol=chol)
