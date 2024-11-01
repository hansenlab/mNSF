
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for training and saving multi-sample Non-negative Spatial Factorization (mNSF) models

This module implements the core functionality for training mNSF models, including:
- Model initialization
- Training loop management
- Convergence checking
- Checkpointing and model saving
- Error handling and learning rate adjustment

@author: Yi Wang based on earlier work by Will Townes for the NSF package. 
"""
# Import necessary libraries
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from time import time,process_time
from contextlib import suppress
from tempfile import TemporaryDirectory
from os import path
import gc
import matplotlib.pyplot as plt
from mNSF.NSF.misc import mkdir_p, pickle_to_file, unpickle_from_file, rpad
from mNSF.NSF import training,visualize

# Import TensorFlow Probability for transformed variables and bijectors
# These are used for constrained optimization and parameter transformations
import tensorflow_probability as tfp
tv = tfp.util.TransformedVariable
tfb = tfp.bijectors

# Set default float type for consistency across the module
dtp = "float32"


# Function to save objects using pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Main function to train mNSF model
def train_model_mNSF(list_fit_,pickle_path_,
            list_Dtrain_,list_D_,legacy=False,test_cvdNorm=False, maxtry=10, lr=0.01, **kwargs):
  """
    Run model training for mNSF across multiple samples.
    
    Args:
        list_fit_: List of initial model fits
        pickle_path_: Path to save pickled models
        list_Dtrain_: List of training datasets
        list_D_: List of full datasets
        legacy: Boolean to use legacy TensorFlow optimizers
        test_cvdNorm: Boolean to test normalized convergence
        maxtry: Maximum number of training attempts
        lr: Initial learning rate
        **kwargs: Additional arguments passed to train_model
    
    Returns:
        list_fit_: List of trained model fits
  """
  # Initialize ModelTrainer object for the first sample
  tro_ = ModelTrainer(list_fit_[0],pickle_path=pickle_path_,legacy=legacy, lr=lr)
  list_tro=list()
  nsample=len(list_D_)     
  # Create ModelTrainer objects for each sample
  for k in range(0,nsample):
    tro_tmp=training.ModelTrainer(list_fit_[k],pickle_path=pickle_path_,legacy=legacy, lr=lr)
    list_tro.append(tro_tmp)
  # Train the model using the main ModelTrainer object
  tro_.train_model(list_tro,
            list_Dtrain_,list_D_, test_cvdNorm=test_cvdNorm,maxtry=maxtry, **kwargs)
  
  # automatically create and save loss plot
  # included here to avoid changing analysis scripts, might want to change later
  tr = np.array(tro_.loss["train"])
  plt.figure()
  plt.plot(tr,c="blue",label="train")
  plt.xlabel("epoch")
  plt.ylabel("ELBO loss")
  plt.savefig("loss.png")
  plt.show()
  plt.clf()

  return list_fit_        


# Custom error for numerical divergence during training
class NumericalDivergenceError(ValueError):
  pass

# Function to truncate loss history to the current epoch
def truncate_history(loss_history, epoch):
  """
    Truncate the loss history to the current epoch.
    
    Args:
        loss_history: Dictionary containing loss histories
        epoch: Current epoch number
    
    Returns:
        Truncated loss history
  """
  with suppress(AttributeError):
    epoch = epoch.numpy()
  cutoff = epoch+1
  for i in loss_history:
    loss_history[i] = loss_history[i][:cutoff]
  return loss_history

# Class to check convergence of the model
class ConvergenceChecker(object):
  def __init__(self,span,dtp="float64"):
    """
        Initialize the ConvergenceChecker with polynomial basis functions.
        
        Args:
            span: Number of recent observations to consider
            dtp: Data type for computations
    """
    x = np.arange(span,dtype=dtp)
    x-= x.mean()
    X = np.column_stack((np.ones(shape=x.shape),x,x**2,x**3))
    self.U = np.linalg.svd(X,full_matrices=False)[0]

  def smooth(self,y):
    """
        Apply smoothing to the input vector using the precomputed basis.
        
        Args:
            y: Input vector to smooth
        
        Returns:
            Smoothed version of y
    """
    return self.U@(self.U.T@y)

  def subset(self,y,idx=-1):
    """
        Extract a subset of the input vector.
        
        Args:
            y: Input vector
            idx: Index to end the subset (default: -1, meaning the end of the vector)
        
        Returns:
            Subset of y
    """
    span = self.U.shape[0]
    lo = idx-span+1
    if idx==-1:
      return y[lo:]
    else:
      return y[lo:(idx+1)]

  def relative_change(self,y,idx=-1,smooth=True):
    """
        Calculate the relative change in the input vector.
        
        Args:
            y: Input vector
            idx: Index to calculate change at (default: -1)
            smooth: Whether to apply smoothing (default: True)
        
        Returns:
            Relative change in y
    """
    y = self.subset(y,idx=idx)
    if smooth:
      y = self.smooth(y)
    prev=y[-2]
    return (y[-1]-prev)/(0.1+abs(prev))
    
    

    
  def converged(self,y,tol=1e-4,**kwargs):
    """
        Check if the relative change is below the tolerance.
        
        Args:
            y: Input vector
            tol: Convergence tolerance (default: 1e-4)
            **kwargs: Additional arguments passed to relative_change
        
        Returns:
            Boolean indicating whether convergence criterion is met
    """
    return abs(self.relative_change(y,**kwargs)) < tol

  
  def relative_chg_normalized(self, y, idx_current = -1, len_trace=50, smooth=True):
    """
        Calculate the normalized relative change over a trace of values.
        
        Args:
            y: Input vector
            idx_current: Current index (default: -1)
            len_trace: Length of the trace to consider (default: 50)
            smooth: Whether to apply smoothing (default: True)
        
        Returns:
            Normalized relative change
    """
    cc = self.relative_change_trace( y, idx_current = idx_current, len_trace=len_trace, smooth=smooth)
    print(cc)
    mean_cc = np.nanmean(cc) 
    sd_cc = np.std(cc) 
    return mean_cc/sd_cc
    

    
  def relative_change_trace(self, y, idx_current = -1, len_trace=50, smooth=True):
    """
        Calculate the trace of relative changes.
        
        Args:
            y: Input vector
            idx_current: Current index (default: -1)
            len_trace: Length of the trace to consider (default: 50)
            smooth: Whether to apply smoothing (default: True)
        
        Returns:
            Array of relative changes
    """
    #n = len(y)
    #span = self.U.shape[0]
    #span = self.U.shape[0]
    cc=self.relative_change_all(y,smooth=smooth)
    return cc[(idx_current-len_trace):idx_current]
    
	
  def relative_change_all(self,y,smooth=True):
    """
        Calculate relative changes for all possible subsets of y.
        
        Args:
            y: Input vector
            smooth: Whether to apply smoothing (default: True)
        
        Returns:
            Array of relative changes for all subsets
    """
    n = len(y)
    span = self.U.shape[0]
    cc = np.tile([np.nan],n)
    for i in range(span,n):
      cc[i] = self.relative_change(y,idx=i,smooth=smooth)
    return cc

  def converged_all(self,y,tol=1e-4,smooth=True):
    """
        Check convergence for all possible subsets of y.
        
        Args:
            y: Input vector
            tol: Convergence tolerance (default: 1e-4)
            smooth: Whether to apply smoothing (default: True)
        
        Returns:
            Boolean array indicating convergence for each subset
    """
    cc = self.relative_change_all(y,smooth=smooth)
    return np.abs(cc)<tol

class ModelTrainer(object): #goal to change this to tf.module?
  """
  Time keeping policy:
    * when object is first created, elapsed wtime and ptime are set to 0.0
    * when train_model is called a wtic, ptic baseline is set
    * whenever a checkpoint is saved or the object is pickled,
    the elapsed time is added to wtime and ptime and a new tic baseline is set
    * wtic and ptic are not stored, only the elapsed time is stored
    * checkpointing and pickling require the user to specify the additional
    elapsed time. If no time has elapsed the user can supply 0.0 values.
  """
  #model: output from pf.ProcessFactorization
  def __init__(self, model, lr=0.01, pickle_path=None, max_to_keep=3, legacy=False, **kwargs):
    #ckpt_path="/tmp/tf_ckpts", #use temporary directory instead
    """
    Initialize the ModelTrainer.
        
    Args:
        model: The mNSF model to be trained
        lr: Initial learning rate
        pickle_path: Path to save pickled models
        max_to_keep: Maximum number of checkpoints to keep
        legacy: Whether to use legacy TensorFlow optimizers
        **kwargs: Additional arguments for the optimizer
    """
    self.loss = {"train":np.array([np.nan]), "val":np.array([np.nan])}
    self.model = model
    self.legacy = legacy
    # Initialize optimizers (main and kernel hyperparameter optimizer)
    if self.legacy: #need to use legacy optimizer with tensorflow v2.12.0 +
      self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr, **kwargs)
      self.optimizer_k = tf.keras.optimizers.legacy.Adam(learning_rate=0.01*lr, **kwargs)
    else:
      #optimizer for all variables except kernel hyperparams
      self.optimizer = tf.optimizers.Adam(learning_rate=lr, **kwargs)
      #optimizer for kernel hyperparams, does nothing for nonspatial models
      self.optimizer_k = tf.optimizers.Adam(learning_rate=0.01*lr, **kwargs)

    # Initialize training variables
    self.epoch = tf.Variable(0,name="epoch")
    # self.tries = tf.Variable(0,name="number of tries")
    self.ptime = tf.Variable(0.0,name="elapsed process time")
    self.wtime = tf.Variable(0.0,name="elapsed wall time")
    #Set up checkpointing
    self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                    optimizer_k=self.optimizer_k,
                                    epoch=self.epoch, ptime=self.ptime,
                                    wtime=self.wtime)#,counter=self.ckpt_counter)
    self.set_pickle_path(pickle_path)
    self.converged=False
  # def reset(self,lr_reduce=0.5):
  #   """
  #   Reset everything and decrease the learning rate
  #   """
  #   self.loss = {"train":[np.nan],"val":[np.nan]}
  #   self.model.reset()
  #   cfg = self.optimizer.get_config()
  #   cfg["learning_rate"]*=lr_reduce
  #   self.optimizer = tf.optimizers.Adam.from_config(cfg)
  #   cfg_k = self.optimizer_k.get_config()
  #   cfg_k["learning_rate"]*=lr_reduce
  #   self.optimizer_k = tf.optimizers.Adam.from_config(cfg_k)
  #   self.epoch.assign(0)
  #   self.ptime.assign(0.0)
  #   self.wtime.assign(0.0)
  #   self.converged=False

  def multiply_lr(self,factor):
    lr = self.optimizer.learning_rate
    lr_old = lr.numpy()
    lr_new = lr_old*factor
    lr.assign(lr_new)
    lr_k = self.optimizer_k.learning_rate
    lr_k.assign(lr_k.numpy()*factor)
    return lr_old,lr_new

  def set_pickle_path(self,pickle_path):
    if pickle_path is not None:
      mkdir_p(pickle_path)
    self.pickle_path=pickle_path

  def update_times(self,pchg,wchg):
    self.ptime.assign_add(pchg)
    self.wtime.assign_add(wchg)
    return process_time(),time() #for resetting the counter

  def checkpoint(self,mgr,*args):
    """
    *args passed to update_times method
    """
    p,w = self.update_times(*args)
    mgr.save(checkpoint_number=self.epoch)
    return p,w

  def restore(self,ckpt_id):
    #implicitly resets self.wtime,self.ptime to what was in the checkpoint
    self.ckpt.restore(ckpt_id)#self.manager.latest_checkpoint)
    return process_time(),time()

  def pickle(self,*args):
    """
    *args passed to update_times method
    """
    p,w = self.update_times(*args)
    if self.converged:
      fname = "converged.pickle"
    else:
      fname = "epoch{}.pickle".format(self.epoch.numpy())
    pickle_to_file(self, path.join(self.pickle_path,fname))
    return p,w

  @staticmethod
  def from_pickle(pth,epoch=None):
    if epoch:
      fname = "epoch{}.pickle".format(epoch)
    else:
      fname = "converged.pickle"
    return unpickle_from_file(path.join(pth,fname))

  def _train_model_fixed_lr(self, list_tro,list_Dtrain, list_D__, ckpt_mgr,Dval=None, #Ntr=None,
                            S=3,
                           verbose=True,num_epochs=500,
                           ptic = process_time(), wtic = time(), ckpt_freq=50, test_cvdNorm=False,
                           kernel_hp_update_freq=10, status_freq=10, chol=True,
                           span=100, tol=1e-4, tol_norm = 0.4, pickle_freq=None, check_convergence: bool = True, vec_batch = None):
    """train_step
    Dtrain, Dval : tensorflow Datasets produced by prepare_datasets_tf func
    ckpt_mgr must store at least 2 checkpoints (max_to_keep)
    Ntr: total number of training observations, needed to adjust KL term in ELBO
    S: number of samples to approximate the ELBO
    verbose: should status updates be printed
    num_epochs: maximum passes through the data after which optimization will be stopped
    ptic,wtic: process and wall time baselines
    kernel_hp_update_freq: how often to update the kernel hyperparameters (eg every 10 epochs)
      updating less than once per epoch improves speed but reduces numerical stability
    status_freq: how often to check for convergence and print updates
    ckpt_freq: how often to save tensorflow checkpoints to disk
    span: when checking for convergence, how many recent observations to consider
    tol: numerical (relative) change below which convergence is declared
    pickle_freq: how often to save the entire object to disk as a pickle file
    """
    ptic,wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
    self.loss["train"] = rpad(self.loss["train"],num_epochs+1)
    if pickle_freq is None: #only pickle at the end
      pickle_freq = num_epochs
    msg = '{:04d} train: {:.3e}'
    if Dval:
      msg += ', val: {:.3e}'
      self.loss["val"] = rpad(self.loss["val"],num_epochs+1)
    msg2 = "" #modified later to include rel_chg
    cvg = 0 #increment each time we think it has converged
    cvg_normalized=0
    cc = ConvergenceChecker(span)
    #epoch=0
    while (not self.converged) and (self.epoch < num_epochs):
      epoch_loss = tf.keras.metrics.Mean()
      #epoch=epoch+1
      #epoch=self.epoch 
      #chol=(self.epoch % kernel_hp_update_freq==0)
      trl=0.0
      nsample=len(list_Dtrain)
      if vec_batch is None:
      	for ksample in range(0,nsample):
        	list_tro[ksample].model.Z=list_D__[ksample]["Z"]
        	Dtrain_ksample = list_Dtrain[ksample]
        	for D in Dtrain_ksample: #iterate through each of the batches 
        		epoch_loss.update_state(list_tro[ksample].model.train_step( D, list_tro[ksample].optimizer, list_tro[ksample].optimizer_k,
                                   Ntot=list_tro[ksample].model.delta.shape[1], chol=chol))
        	trl = trl + epoch_loss.result().numpy()
      else:
      	for ksample in range(0,nsample):
        	list_tro[ksample].model.Z=list_D__[ksample]["Z"]
        	Dtrain_ksample = list_Dtrain[ksample]
        	if vec_batch[ksample]:
        		list_tro[ksample].model.beta0.assign(list_tro[ksample-1].model.beta0)
        		list_tro[ksample].model.beta.assign(list_tro[ksample-1].model.beta)
        		list_tro[ksample].model.W.assign(list_tro[ksample-1].model.W.numpy())
        		list_tro[ksample].model.amplitude=(list_tro[ksample-1].model.amplitude())
        		list_tro[ksample].model.length_scale=(list_tro[ksample-1].model.length_scale())
    #Loadings weights
        	for D in Dtrain_ksample: #iterate through each of the batches 
        		epoch_loss.update_state(list_tro[ksample].model.train_step( D, list_tro[ksample].optimizer, list_tro[ksample].optimizer_k,
                                   Ntot=list_tro[ksample].model.delta.shape[1], chol=chol))
        	trl = trl + epoch_loss.result().numpy()
      W_updated=list_tro[ksample].model.W-list_tro[ksample].model.W
      #print(trl)
      for ksample in range(0,nsample):
      	W_updated = W_updated+ (list_tro[ksample].model.W / nsample)
      self.epoch.assign_add(1)
      i = self.epoch.numpy()
      self.loss["train"][i] = trl

      ## check for nan in any sample loadings
      for fit_i in list_tro:
        if np.isnan(fit_i.model.W).any():
          print('NaN in sample ' + str(list_tro.index(fit_i) + 1))

      if not np.isfinite(trl): ### modified
        print("training loss calculated at the point of divergence: ")
        print(trl)
        raise NumericalDivergenceError###!!!NumericalDivergenceError
      #if not np.isfinite(trl) or trl>self.loss["train"][1]: ### modified
      #  raise NumericalDivergenceError###!!!NumericalDivergenceError
      if i%status_freq==0 or i==num_epochs:
        if Dval:
          val_loss = self.model.validation_step(Dval, S=S, chol=False).numpy()
          self.loss["val"][i] = val_loss
        if i>span and check_convergence: #checking for convergence
          rel_chg = cc.relative_change(self.loss["train"],idx=i)
          print("rel_chg")
          print(rel_chg)
          msg2 = ", chg: {:.2e}".format(-rel_chg)
          if abs(rel_chg)<tol: cvg+=1
          else: cvg=0   
          if test_cvdNorm:
          	rel_chg_normalized=cc.relative_chg_normalized(self.loss["train"],idx_current=i) 
          	print("rel_chg_normalized")
          	print(rel_chg_normalized)
          	if(-(rel_chg_normalized)<tol_norm): cvg_normalized+=1 # positive values of rel_chg_normalized indicates increase of loss throughout the past 10 iterations
          if cvg>=2 or cvg_normalized>=2: #i.e. either convergence or normalized convergence has been detected twice in a row
            self.converged=True
            pickle_freq = i #ensures final pickling will happen
            self.loss = truncate_history(self.loss, i)
        if verbose:
          if Dval: print(msg.format(i,trl,val_loss)+msg2)
          else: print(msg.format(i,trl)+msg2)
      if i%ckpt_freq==0:
        ptic,wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
      if self.pickle_path and i%pickle_freq==0:
        ptic,wtic = self.pickle(process_time()-ptic, time()-wtic)
      # except tf.errors.InvalidArgumentError as err: #cholesky failed
      #   j = i.numpy() #save the current epoch value for printing
      #   ptic,wtic = self.restore()
      #   # self.ckpt.restore(self.manager.latest_checkpoint) #resets i to last checkpt
      #   if ng < 1.0: ng *= 10.0
      #   else: raise err #nugget has gotten too big so just give up
      #   try: self.model.set_nugget(ng) #spatial or integrated model
      #   except AttributeError: raise err #nonspatial model
      #   if verbose:
      #     print("Epoch: {:04d}, numerical error, reverting to epoch {:04d}, \
      #           increase nugget to {:.3E}".format(j, i.numpy(), ng))
      #   self.loss = truncate_history(self.loss,i)
      #   continue

  def find_checkpoint(self, ckpt_freq, back=1, epoch0=0):
    """
    If checkpoints are saved every [ckpt_freq] epochs, and we want to go back
    at least [back*ckpt_freq] epochs from the current epoch,
    returns the epoch number where the files can be found
    For example, if we are at epoch 201, ckpt_freq=50, and back=1, we want to
    go back to the checkpoint saved at epoch 150 (NOT 200).
    If back=2, we would go back to epoch 100.
    If this takes it below zero, we truncate at zero, since there should always
    be a checkpoint at epoch 0. Note this assumption may be violated for
    models loaded by from_pickle. For example if pickling happened at epoch
    201, and further model fitting proceeded to hit a numerical divergence at
    epoch 209, this function would try to go back to epoch 150 but the checkpoint
    would not exist because the temporary directory would have been cleaned up.
    This is considered acceptable since objects loaded from pickle are assumed
    to be either already numerically converged or to have at least run a large
    number of epochs without numerical problems.
    """
    return max(ckpt_freq*(self.epoch.numpy()//ckpt_freq - back), epoch0)

  def train_model(self, list_tro, list_Dtrain, list_D__, lr_reduce=0.5, maxtry=10, verbose=True,#*args,
                  ckpt_freq=50, test_cvdNorm=False, **kwargs):
    """
    See train_model_fixed_lr for args and kwargs. This method is a wrapper
    that automatically tries to adjust the learning rate
    """
    # assert self.tries<=maxtry
    ptic=process_time()
    wtic=time()
    tries=0
    #set the earliest epoch that could be returned to if numerical divergence
    epoch0=self.epoch.numpy() #usually zero, unless loaded from a pickle file
    with TemporaryDirectory() as ckpth:
      if verbose: print("Temporary checkpoint directory: {}".format(ckpth))
      mgr = tf.train.CheckpointManager(self.ckpt, directory=ckpth, max_to_keep=maxtry)
      while tries < maxtry:
        try:
          self._train_model_fixed_lr(list_tro,list_Dtrain, list_D__, mgr, #*args,
                                      #ptic=ptic, wtic=wtic, 
                                     #verbose=verbose, 
                                     ckpt_freq=ckpt_freq,
                                     **kwargs)
          if self.epoch>=len(self.loss["train"])-1: break #finished training
        except (tf.errors.InvalidArgumentError,NumericalDivergenceError) as err: #cholesky failure
          # plot loss when diverges          
          tr = np.array(self.loss["train"])
          #color_list = list(matplotlib.colors.TABLEAU_COLORS.values())
          plt.figure()
          plt.plot(tr,c='blue',label="train")
          plt.xlabel("epoch")
          plt.ylabel("ELBO loss")
          plt.title("Try " + str(tries))
          plt.savefig("loss" + str(tries) + ".png")
          plt.show()
          plt.clf()
          #
          tries+=1
          print("tries")
          print(tries)
          if tries==maxtry: raise err
          #else: #not yet reached the maximum number of tries
          if verbose:
            msg = "{:04d} numerical instability (try {:d})"
            print(msg.format(self.epoch.numpy(),tries))
          #new_epoch = self.find_checkpoint(ckpt_freq, back=1, epoch0=epoch0) #int  #modofied
          new_epoch=0
          self.epoch.assign(0)#modofied
          #new_epoch=0
          #self.epoch=0
          ckpt = path.join(ckpth,"ckpt-{}".format(new_epoch))
          # self.reset(lr_reduce=lr_reduce)
          ptic,wtic = self.restore(ckpt)
          nsample=len(list_Dtrain)
          #print("nsample")
          #print(nsample)
          for ksample in range(0,nsample):
            with open('fit_'+ str(ksample+1) +'_restore.pkl', 'rb') as inp:
              list_tro[ksample].model = pickle.load(inp)  
            #save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
          lr_old,lr_new=self.multiply_lr(lr_reduce)
          if verbose:
            msg = "{:04d} learning rate: {:.2e}"
            print(msg.format(new_epoch,lr_new))
    if verbose:
      msg = "{:04d} training complete".format(self.epoch.numpy())
      if self.converged:
        print(msg+", converged.")
      else:
        print(msg+", reached maximum epochs.")


# def check_convergence(x,epoch,tol=1e-4):
#   """
#   x: a vector of loss function values
#   epoch: index of x with most recent loss
#   """
#   # with suppress(AttributeError):
#   #   epoch = epoch.numpy()
#   prev = x[epoch-1]
#   return abs(x[epoch]-prev)/(0.1+abs(prev)) < tol

# def check_convergence_linear(y,pval=.05):
#   n = len(y)
#   x = np.arange(n,dtype=y.dtype)
#   x -= x.mean()
#   yctr = y-y.mean()
#   return linregress(x,y).pvalue>pval

# def standardize(x,dtp="float64"):
#   xm = x.astype(dtp)-x.mean(dtype=dtp)
#   return xm/norm(xm)

# def check_convergence_linear(y,z=None,tol=0.1):
#   if z is None:
#     z = standardize(np.arange(len(y)))
#   # return abs(pearsonr(x,idx)[0])<tol
#   return np.abs(np.dot(standardize(y),z))<tol

# def check_convergence_posthoc(x,tol,method="linear",span=50):
#   if method=="simple":
#     start = 2
#     f = lambda i: check_convergence(x,i,tol)
#   elif method=="linear":
#     start = span
#     z=standardize(np.arange(span))
#     f = lambda i: check_convergence_linear(x[(i-span+1):(i+1)],z=z,tol=tol)
#   else:
#     raise ValueError("method must be linear or simple")
#   cc = np.tile([False],len(x))
#   for i in range(start,len(x)):
#     cc[i] = f(i)
#   return cc

# def update_time(t=None,chg=None):
#   try:
#     return t+chg
#   except TypeError: #either t or chg is None
#     return chg #note this may not be None if t is None but chg is not
