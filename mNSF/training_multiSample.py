#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for training and saving models

@author: Yi Wang based on earlier work by Will Townes for the NSF package. 
"""
import pickle
import pandas as pd

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


import psutil
import numpy as np
import tensorflow as tf
from time import time,process_time
from contextlib import suppress
from tempfile import TemporaryDirectory
from os import path
dtp = "float32"

### check tv objects memory usage
import tensorflow_probability as tfp
tv = tfp.util.TransformedVariable
tfb = tfp.bijectors

from mNSF.NSF.misc import mkdir_p, pickle_to_file, unpickle_from_file, rpad



def assign_paras_from_np_to_tf(fit_, list_para):
  fit_.beta0.assign(list_para["beta0"])
  fit_.beta.assign(list_para["beta"])
  fit_.W.assign(list_para["W"])
  fit_.nugget.assign(list_para["nugget"])
  fit_.amplitude.assign(list_para["amplitude"])
  fit_.length_scale.assign(list_para["length_scale"])
  fit_._ls0=list_para["_ls0"]
  #fit_.scale_diag.assign(list_para["scale_diag"])
  del fit_.Omega_tril
  del fit_.delta
  with tf.name_scope("variational"):
      #self.delta = tf.Variable(rng.normal(size=(L,M)), dtype=dtp, name="mean") #LxM
      #tf.assign(fit_.delta, list_para["delta"], validate_shape=False)
      #tf.reshape(fit12.delta, fit12.delta[1:10].shape)
      #fit12.delta = tf.Variable(fit12.delta,shape=tf.TensorShape(None), dtype=dtp, name="mean") #LxM
      #fit12.delta = tf.Variable(fit12.delta, dtype=dtp, name="mean") #LxM
      #fit12.delta.assign(fit12.delta[1:10])
      fit_.delta = tf.Variable(list_para["delta"], dtype=dtp, name="mean") #LxM
      #fit12.Omega_tril=tv(fit12.delta.numpy()[1:2,1:2], tfb.FillScaleTriL(),shape=tf.TensorShape(None), dtype=dtp, name="covar_tril")
      #fit12.Omega_tril.assign(fit12.delta.numpy()[1:3,1:3])
      #fit12.Omega_tril=tv(fit12.Omega_tril, tfb.FillScaleTriL(),shape=(3,3), dtype=dtp, name="covar_tril")
      #fit12.Omega_tril.numpy()  
      #tf.reshape(fit12.Omega_tril,[9])
      fit_.Omega_tril=tv(list_para["Omega_tril"], tfb.FillScaleTriL(), dtype=dtp, name="covar_tril") #LxMxM #don't need this in the initialization
      #print(fit_.delta)
      #print(fit_.Omega_tril)
  del fit_.Kuu_chol
  print("assign_paras")
  print(list_para["Kuu_chol"].shape)
  #fit_.Kuu_chol = tf.Variable(list_para["Kuu_chol"], dtype=dtp, name="Variable:0", trainable=False)#don't need this in the initialization
  fit_.Kuu_chol = tf.Variable(fit_.eval_Kuu_chol(fit_.get_kernel()), dtype=dtp, trainable=False)#don't need this in the initialization
  #stuff to facilitate caching
  for i in fit_.trainable_variables:
    print(i.name)
  del fit_.trvars_kernel
  del fit_.trvars_nonkernel
  fit_.trvars_kernel = tuple(i for i in fit_.trainable_variables if i.name[:10]=="gp_kernel/")
  fit_.trvars_nonkernel = tuple(i for i in fit_.trainable_variables if i.name[:10]!="gp_kernel/")
  del fit_.kernel
  if fit_.isotropic:
    fit_.kernel = fit_.psd_kernel(amplitude=fit_.amplitude, length_scale=fit_.length_scale)
  else:
    fit_.kernel = tfk.FeatureScaled(fit_.psd_kernel(amplitude=fit_.amplitude), fit_.scale_diag)


def store_paras_from_tf_to_np(fit_):
      list_para={}
      list_para["beta0"]=fit_.beta0.numpy()
      list_para["beta"]=fit_.beta.numpy()
      list_para["W"]=fit_.W.numpy()
      list_para["nugget"]=fit_.nugget.numpy()
      list_para["amplitude"]=fit_.amplitude.numpy()
      list_para["length_scale"]=fit_.length_scale.numpy()
      list_para["_ls0"]=fit_._ls0
      list_para["Kuu_chol"]=fit_.Kuu_chol.numpy()
      #list_para["scale_diag"]=fit_.scale_diag.numpy()
      #self.Kuu_chol = tf.Variable(self.eval_Kuu_chol(self.get_kernel()), dtype=dtp, trainable=False)#don't need this in the initialization
      list_para["Omega_tril"]=fit_.Omega_tril.numpy() #LxMxM #don't need this in the initialization
      list_para["delta"]=fit_.delta.numpy()  #LxM
      return list_para

class NumericalDivergenceError(ValueError):
  pass

def truncate_history(loss_history, epoch):
  with suppress(AttributeError):
    epoch = epoch.numpy()
  cutoff = epoch+1
  for i in loss_history:
    loss_history[i] = loss_history[i][:cutoff]
  return loss_history

class ConvergenceChecker(object):
  def __init__(self,span,dtp="float64"):
    x = np.arange(span,dtype=dtp)
    x-= x.mean()
    X = np.column_stack((np.ones(shape=x.shape),x,x**2,x**3))
    self.U = np.linalg.svd(X,full_matrices=False)[0]

  def smooth(self,y):
    return self.U@(self.U.T@y)

  def subset(self,y,idx=-1):
    span = self.U.shape[0]
    lo = idx-span+1
    if idx==-1:
      return y[lo:]
    else:
      return y[lo:(idx+1)]

  def relative_change(self,y,idx=-1,smooth=True):
    y = self.subset(y,idx=idx)
    if smooth:
      y = self.smooth(y)
    prev=y[-2]
    return (y[-1]-prev)/(0.1+abs(prev))

  def converged(self,y,tol=1e-4,**kwargs):
    return abs(self.relative_change(y,**kwargs)) < tol

  def relative_change_all(self,y,smooth=True):
    n = len(y)
    span = self.U.shape[0]
    cc = np.tile([np.nan],n)
    for i in range(span,n):
      cc[i] = self.relative_change(y,idx=i,smooth=smooth)
    return cc

  def converged_all(self,y,tol=1e-4,smooth=True):
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
  def __init__(self, model, lr=0.01, pickle_path=None, max_to_keep=3, **kwargs):
    #ckpt_path="/tmp/tf_ckpts", #use temporary directory instead
    """
    **kwargs are passed to tf.optimizers.[Optimizer] constructor
    """
    self.loss = {"train":np.array([np.nan]), "val":np.array([np.nan])}
    self.model = model
    #optimizer for all variables except kernel hyperparams
    self.optimizer = tf.optimizers.Adam(learning_rate=lr, **kwargs)
    #optimizer for kernel hyperparams, does nothing for nonspatial models
    self.optimizer_k = tf.optimizers.Adam(learning_rate=0.01*lr, **kwargs)
    self.epoch = tf.Variable(0,name="epoch")
    # self.tries = tf.Variable(0,name="number of tries")
    self.ptime = tf.Variable(0.0,name="elapsed process time")
    self.wtime = tf.Variable(0.0,name="elapsed wall time")
    # self.ckpt_counter = tf.Variable(0,name="checkpoint counter")
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

  def _train_model_fixed_lr(self, list_self,list_D, list_D__, ckpt_mgr,Dval=None, #Ntr=None,  
                            S=3,
                           verbose=True,num_epochs=5000,
                           ptic = process_time(), wtic = time(), ckpt_freq=50,
                           kernel_hp_update_freq=10, status_freq=10,
                           span=100, tol=(5e-5)*2, pickle_freq=None):
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
    #print("self.model.beta0_1")
    #print(self.model.beta0_1)# it looks fins here
    #print("self.model.beta0_1")
    num_epochs=1####modified
    #num_epochs=5000####modified
    ptic,wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
    print("num_epochs")
    print(num_epochs)
    print(type(num_epochs))
    self.loss["train"] = rpad(self.loss["train"],num_epochs+1)
    if pickle_freq is None: #only pickle at the end
      pickle_freq = num_epochs
    msg = '{:04d} train: {:.3e}'
    if Dval:
      msg += ', val: {:.3e}'
      self.loss["val"] = rpad(self.loss["val"],num_epochs+1)
    msg2 = "" #modified later to include rel_chg
    #model, optimizer, S, Ntr do not change during optimization
    # @tf.function
    # def train_step_chol(D):
    #   return self.model.train_step(D, self.optimizer, self.optimizer_k,
    #                                S=S, Ntot=Ntr, chol=True)
    # @tf.function
    # def train_step_nochol(D):
    #   return self.model.train_step(D, self.optimizer, self.optimizer_k,
    #                                S=S, Ntot=Ntr, chol=False)
    print(12300)
    #print(self.model.beta0_1[1])
    #print(self.model.W[1,1])
    print(12300)
    #@tf.function
    #def train_step(list_self, list_D, chol=True):
    #  # if chol: return train_step_chol(D)
    #  # else: return train_step_nochol(D)
    #  return self.model.train_step(list_self,list_D, self.optimizer, self.optimizer_k,
    #                               S=S, Ntot=Ntr, chol=chol)
    #def train_step(self, D1, D2, optimizer, optimizer_k, S=1, Ntot=None, chol=True):
    cvg = 0 #increment each time we think it has converged
    cc = ConvergenceChecker(span)
    while (not self.converged) and (self.epoch < num_epochs):
      epoch_loss = tf.keras.metrics.Mean()
      chol=(self.epoch % kernel_hp_update_freq==0)
      trl=0.0
      nsample=len(list_D)
      #for kkk in range(0,nsample):
      #  with open('tro_'+ str(kkk+1) +'.pkl', 'rb') as inp:
      #    list_self[kkk].model = pickle.load(inp)
      for kkk in range(0,nsample):
        #list_self[kkk]
        print(kkk)
        with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
          list_para_tmp = pickle.load(inp)
        list_self[0].model.Z=list_D__[kkk]["Z"]
        print("self_epoch_numpy")
        #print(list_self[0].model.Z.shape)
        print(self.epoch.numpy())
        #if self.epoch.numpy()==0 and kkk==0:
        #  W_new=list_para_tmp["W"]
        #if self.epoch.numpy()>0 OR kkk>0:
        #list_para_tmp["W"]=W_new
        for D in list_D[kkk]: #iterate through each of the batches ####???????!!!!!!!
          #epoch_loss.update_state(train_step(D,chol=chol)) # perform one training step for each bath, #assume all batches shares the same parameters (? mean?)   
          print(psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
          print(list_para_tmp["Kuu_chol"].shape)
          assign_paras_from_np_to_tf(list_self[0].model,list_para_tmp)
          print("training_Kuu_chol_dim")#(7, 4384, 4384)
          print(list_self[0].model.Kuu_chol.shape)
          del list_para_tmp
          #del list_self[0].model.Z
          #print("Zshape")
          #print(D["X"].shape)#(4226, 2)
          #print(list_self[0].model.Z.shape)
          aa=list_self[0].model.train_step( D, list_self[0].optimizer, list_self[0].optimizer_k,
                                   Ntot=list_self[0].model.delta.shape[1], chol=True)
          print(psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
          epoch_loss.update_state(aa)
          trl = trl + epoch_loss.result().numpy()
          print("trl")
          print(trl)
          list_para_tmp=store_paras_from_tf_to_np(list_self[0].model)
          W_new=list_para_tmp["W"]
          save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
      ## calculate the updated W by getting the average of W across all n samples
      #kkk=0
      W_new=list_para_tmp["W"] - list_para_tmp["W"]
      #W_new=W_new
      for kkk in range(0,nsample):
        with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
          list_para_tmp = pickle.load(inp)
        W_new=W_new+list_para_tmp['W'] / nsample
      # assign the updated W to each of the model trainer object for each of the sample
      for kkk in range(0,nsample):
        with open('list_para_'+ str(kkk+1) +'.pkl', 'rb') as inp:
          list_para_tmp = pickle.load(inp)
          list_para_tmp["W"]=W_new
          save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
      #a=train_step(self, D1, D2, chol=chol)
      #epoch_loss.update_state(train_step(self, D1, D2, chol=chol))### this step has problems
      #for D in Dtrain: #iterate through each of the batches
      #  epoch_loss.update_state(train_step(D1, D2, chol=chol))
      print(12312312300)
      print(12312312300)
      self.epoch.assign_add(1)
      i = self.epoch.numpy()
      print("i")
      print(i)
      self.loss["train"][i] = trl
      if not np.isfinite(trl): ### modified
        raise NumericalDivergenceError###!!!NumericalDivergenceError
      #if not np.isfinite(trl) or trl>self.loss["train"][1]: ### modified
      #  raise NumericalDivergenceError###!!!NumericalDivergenceError
      if i%status_freq==0 or i==num_epochs:
        if Dval:
          val_loss = self.model.validation_step(Dval, S=S, chol=False).numpy()
          self.loss["val"][i] = val_loss
        if i>span: #checking for convergence
          rel_chg = cc.relative_change(self.loss["train"],idx=i)
          print("rel_chg")
          print(rel_chg)
          msg2 = ", chg: {:.2e}".format(-rel_chg)
          if abs(rel_chg)<tol: cvg+=1
          else: cvg=0
          if cvg>=2: #ie convergence has been detected twice in a row---!!!!!!!
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

  def train_model(self, list_self, list_D, list_D__, lr_reduce=0.5, maxtry=10, verbose=True,#*args,
                  ckpt_freq=50, **kwargs):
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
          self._train_model_fixed_lr(list_self,list_D, list_D__, mgr, #*args,
                                      #ptic=ptic, wtic=wtic, 
                                     #verbose=verbose, 
                                     ckpt_freq=ckpt_freq,
                                     **kwargs)
          if self.epoch>=len(self.loss["train"])-1: break #finished training
        except (tf.errors.InvalidArgumentError,NumericalDivergenceError) as err: #cholesky failure
          tries+=1
          print("tries")
          print(tries)
          if tries==maxtry: raise err
          #else: #not yet reached the maximum number of tries
          if verbose:
            msg = "{:04d} numerical instability (try {:d})"
            print(msg.format(self.epoch.numpy(),tries))
          new_epoch = self.find_checkpoint(ckpt_freq, back=1, epoch0=epoch0) #int
          ckpt = path.join(ckpth,"ckpt-{}".format(new_epoch))
          # self.reset(lr_reduce=lr_reduce)
          ptic,wtic = self.restore(ckpt)
          nsample=len(list_D__)
          print("nsample")
          print(nsample)
          for kkk in range(0,nsample):
            with open('list_para_'+ str(kkk+1) +'_restore.pkl', 'rb') as inp:
              list_para_tmp = pickle.load(inp)
            save_object(list_para_tmp, 'list_para_'+ str(kkk+1) +'.pkl')
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
