#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelTrainer Class for Multi-sample Non-negative Spatial Factorization (mNSF)

This file contains the ModelTrainer class, which is responsible for training and managing
the mNSF model. It includes methods for:

1. Learning rate adjustment
2. Checkpoint management
3. Model serialization (pickling)
4. Training loop implementation
5. Convergence checking

The ModelTrainer class handles both single-epoch training with a fixed learning rate
(_train_model_fixed_lr) and multi-epoch training with automatic learning rate adjustment
(train_model). It also provides utilities for saving and loading model states, updating
training times, and managing the training process across multiple samples.

Key features:
- Supports multiple samples and datasets
- Implements adaptive learning rate
- Provides convergence checking mechanisms
- Handles model checkpointing and serialization
- Supports both CPU and GPU training

This class is a core component of the mNSF package, designed for analyzing spatial
transcriptomics data across multiple samples without requiring alignment.

Author: Yi Wang (based on earlier work by Will Townes for the NSF package)
"""

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

# Check memory usage of TensorFlow Variable objects
import tensorflow_probability as tfp
tv = tfp.util.TransformedVariable
tfb = tfp.bijectors

dtp = "float32"  # Default data type for floating-point operations

# Function to save Python objects to a file using pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Function to train the mNSF model
def train_model_mNSF(list_fit_, pickle_path_, list_Dtrain_, list_D_, legacy=False, test_cvdNorm=False, maxtry=10, lr=0.01, **kwargs):
    """
    Run model training for mNSF
    Returns a list of fitted models, where each item in the list is for one sample
    """
    # Initialize the model trainer for the first sample
    tro_ = ModelTrainer(list_fit_[0], pickle_path=pickle_path_, legacy=legacy, lr=lr)
    list_tro = []
    nsample = len(list_D_)     
    
    # Create ModelTrainer objects for each sample
    for k in range(0, nsample):
        tro_tmp = training.ModelTrainer(list_fit_[k], pickle_path=pickle_path_, legacy=legacy, lr=lr)
        list_tro.append(tro_tmp)
    
    # Train the model
    tro_.train_model(list_tro, list_Dtrain_, list_D_, test_cvdNorm=test_cvdNorm, maxtry=maxtry, **kwargs)
  
    # Automatically create and save loss plot
    tr = np.array(tro_.loss["train"])
    plt.figure()
    plt.plot(tr, c="blue", label="train")
    plt.xlabel("epoch")
    plt.ylabel("ELBO loss")
    plt.savefig("loss.png")
    plt.show()
    plt.clf()

    return list_fit_        

# Custom error for numerical divergence
class NumericalDivergenceError(ValueError):
    pass

# Function to truncate loss history
def truncate_history(loss_history, epoch):
    with suppress(AttributeError):
        epoch = epoch.numpy()
    cutoff = epoch + 1
    for i in loss_history:
        loss_history[i] = loss_history[i][:cutoff]
    return loss_history

# Class to check convergence of the model
class ConvergenceChecker(object):
    def __init__(self, span, dtp="float64"):
        # Initialize with a polynomial basis for smoothing
        x = np.arange(span, dtype=dtp)
        x -= x.mean()
        X = np.column_stack((np.ones(shape=x.shape), x, x**2, x**3))
        self.U = np.linalg.svd(X, full_matrices=False)[0]

    def smooth(self, y):
        # Apply smoothing to the input data
        return self.U @ (self.U.T @ y)

    def subset(self, y, idx=-1):
        # Extract a subset of the data
        span = self.U.shape[0]
        lo = idx - span + 1
        if idx == -1:
            return y[lo:]
        else:
            return y[lo:(idx+1)]

    def relative_change(self, y, idx=-1, smooth=True):
        # Calculate relative change in the data
        y = self.subset(y, idx=idx)
        if smooth:
            y = self.smooth(y)
        prev = y[-2]
        return (y[-1] - prev) / (0.1 + abs(prev))
    
    def converged(self, y, tol=1e-4, **kwargs):
        # Check if the data has converged
        return abs(self.relative_change(y, **kwargs)) < tol
  
    def relative_chg_normalized(self, y, idx_current=-1, len_trace=50, smooth=True):
        # Calculate normalized relative change
        cc = self.relative_change_trace(y, idx_current=idx_current, len_trace=len_trace, smooth=smooth)
        print(cc)
        mean_cc = np.nanmean(cc) 
        sd_cc = np.std(cc) 
        return mean_cc / sd_cc
    
    def relative_change_trace(self, y, idx_current=-1, len_trace=50, smooth=True):
        # Calculate relative change over a trace of data points
        cc = self.relative_change_all(y, smooth=smooth)
        return cc[(idx_current-len_trace):idx_current]
	
    def relative_change_all(self, y, smooth=True):
        # Calculate relative change for all data points
        n = len(y)
        span = self.U.shape[0]
        cc = np.tile([np.nan], n)
        for i in range(span, n):
            cc[i] = self.relative_change(y, idx=i, smooth=smooth)
        return cc

    def converged_all(self, y, tol=1e-4, smooth=True):
        # Check convergence for all data points
        cc = self.relative_change_all(y, smooth=smooth)
        return np.abs(cc) < tol

# Main class for training the model
class ModelTrainer(object):
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
    def __init__(self, model, lr=0.01, pickle_path=None, max_to_keep=3, legacy=False, **kwargs):
        # Initialize the ModelTrainer object
        self.loss = {"train": np.array([np.nan]), "val": np.array([np.nan])}
        self.model = model
        self.legacy = legacy
        
        # Set up optimizers
        if self.legacy:
            self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr, **kwargs)
            self.optimizer_k = tf.keras.optimizers.legacy.Adam(learning_rate=0.01*lr, **kwargs)
        else:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr, **kwargs)
            self.optimizer_k = tf.optimizers.Adam(learning_rate=0.01*lr, **kwargs)

        # Initialize training variables
        self.epoch = tf.Variable(0, name="epoch")
        self.ptime = tf.Variable(0.0, name="elapsed process time")
        self.wtime = tf.Variable(0.0, name="elapsed wall time")
        
        # Set up checkpoint
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                        optimizer_k=self.optimizer_k,
                                        epoch=self.epoch, ptime=self.ptime,
                                        wtime=self.wtime)
        self.set_pickle_path(pickle_path)
        self.converged = False

  def multiply_lr(self, factor):
    """
    Multiply the learning rate of both optimizers by a given factor.
    
    Args:
        factor (float): The multiplier for the learning rate.
    
    Returns:
        tuple: The old and new learning rates.
    """
    lr = self.optimizer.learning_rate
    lr_old = lr.numpy()
    lr_new = lr_old * factor
    lr.assign(lr_new)  # Update main optimizer learning rate
    
    lr_k = self.optimizer_k.learning_rate
    lr_k.assign(lr_k.numpy() * factor)  # Update kernel optimizer learning rate
    
    return lr_old, lr_new

def set_pickle_path(self, pickle_path):
    """
    Set the path for saving pickle files and create the directory if it doesn't exist.
    
    Args:
        pickle_path (str): The path where pickle files will be saved.
    """
    if pickle_path is not None:
        mkdir_p(pickle_path)  # Create directory if it doesn't exist
    self.pickle_path = pickle_path

def update_times(self, pchg, wchg):
    """
    Update the process and wall times.
    
    Args:
        pchg (float): Change in process time.
        wchg (float): Change in wall time.
    
    Returns:
        tuple: Current process time and wall time.
    """
    self.ptime.assign_add(pchg)  # Add change to process time
    self.wtime.assign_add(wchg)  # Add change to wall time
    return process_time(), time()  # Return current times for resetting the counter

def checkpoint(self, mgr, *args):
    """
    Save a checkpoint and update times.
    
    Args:
        mgr (tf.train.CheckpointManager): Checkpoint manager.
        *args: Arguments passed to update_times method.
    
    Returns:
        tuple: Updated process time and wall time.
    """
    p, w = self.update_times(*args)
    mgr.save(checkpoint_number=self.epoch)  # Save checkpoint
    return p, w

def restore(self, ckpt_id):
    """
    Restore the model from a checkpoint.
    
    Args:
        ckpt_id (str): Checkpoint identifier.
    
    Returns:
        tuple: Current process time and wall time.
    """
    self.ckpt.restore(ckpt_id)  # Restore checkpoint (resets times implicitly)
    return process_time(), time()

def pickle(self, *args):
    """
    Save the model state to a pickle file.
    
    Args:
        *args: Arguments passed to update_times method.
    
    Returns:
        tuple: Updated process time and wall time.
    """
    p, w = self.update_times(*args)
    if self.converged:
        fname = "converged.pickle"
    else:
        fname = f"epoch{self.epoch.numpy()}.pickle"
    pickle_to_file(self, path.join(self.pickle_path, fname))
    return p, w

@staticmethod
def from_pickle(pth, epoch=None):
    """
    Load a model from a pickle file.
    
    Args:
        pth (str): Path to the pickle file directory.
        epoch (int, optional): Specific epoch to load. If None, loads the converged state.
    
    Returns:
        ModelTrainer: Loaded model trainer object.
    """
    if epoch:
        fname = f"epoch{epoch}.pickle"
    else:
        fname = "converged.pickle"
    return unpickle_from_file(path.join(pth, fname))

def _train_model_fixed_lr(self, list_tro, list_Dtrain, list_D__, ckpt_mgr, Dval=None, S=3,
                          verbose=True, num_epochs=500, ptic=process_time(), wtic=time(), 
                          ckpt_freq=50, test_cvdNorm=False, kernel_hp_update_freq=10, 
                          status_freq=10, span=100, tol=1e-4, tol_norm=0.4, 
                          pickle_freq=None, check_convergence: bool = True):
    """
    Train the model with a fixed learning rate.
    
    Args:
        ... [args description as in the original docstring] ...
    """
    # Save initial checkpoint and update time
    ptic, wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
    
    # Pad the loss array to accommodate all epochs
    self.loss["train"] = rpad(self.loss["train"], num_epochs+1)
    
    # Set pickle frequency to num_epochs if not specified
    if pickle_freq is None:
        pickle_freq = num_epochs
    
    # Prepare status message format
    msg = '{:04d} train: {:.3e}'
    if Dval:
        msg += ', val: {:.3e}'
        self.loss["val"] = rpad(self.loss["val"], num_epochs+1)
    msg2 = ""  # Will be used to include relative change info
    
    # Initialize convergence counters
    cvg = 0  # Regular convergence counter
    cvg_normalized = 0  # Normalized convergence counter
    
    # Create ConvergenceChecker object
    cc = ConvergenceChecker(span)
    
    # Main training loop
    while (not self.converged) and (self.epoch < num_epochs):
        epoch_loss = tf.keras.metrics.Mean()
        chol = (self.epoch % kernel_hp_update_freq == 0)  # Check if kernel hyperparameters should be updated
        trl = 0.0  # Total training loss for this epoch
        nsample = len(list_Dtrain)
        
        # Iterate over all samples
        for ksample in range(0, nsample):
            list_tro[ksample].model.Z = list_D__[ksample]["Z"]  # Update Z for each sample
            Dtrain_ksample = list_Dtrain[ksample]
            
            # Train on each batch in the sample
            for D in Dtrain_ksample:
                epoch_loss.update_state(list_tro[ksample].model.train_step(
                    D, list_tro[ksample].optimizer, list_tro[ksample].optimizer_k,
                    Ntot=list_tro[ksample].model.delta.shape[1], chol=True
                ))
                trl = trl + epoch_loss.result().numpy()
        
        # Update W (not clear what this does without more context)
        W_updated = list_tro[ksample].model.W - list_tro[ksample].model.W
        for ksample in range(0, nsample):
            W_updated = W_updated + (list_tro[ksample].model.W / nsample)
        
        # Increment epoch and record loss
        self.epoch.assign_add(1)
        i = self.epoch.numpy()
        self.loss["train"][i] = trl

        # Check for NaN in sample loadings
        for fit_i in list_tro:
            if np.isnan(fit_i.model.W).any():
                print('NaN in sample ' + str(list_tro.index(fit_i) + 1))

        # Check for loss divergence
        if not np.isfinite(trl):
            print("training loss calculated at the point of divergence: ")
            print(trl)
            raise NumericalDivergenceError

        # Periodic status check and convergence test
        if i % status_freq == 0 or i == num_epochs:
            # Compute validation loss if validation data is provided
            if Dval:
                val_loss = self.model.validation_step(Dval, S=S, chol=False).numpy()
                self.loss["val"][i] = val_loss
            
            # Check for convergence
            if i > span and check_convergence:
                rel_chg = cc.relative_change(self.loss["train"], idx=i)
                print("rel_chg")
                print(rel_chg)
                msg2 = ", chg: {:.2e}".format(-rel_chg)
                
                # Update convergence counter
                if abs(rel_chg) < tol:
                    cvg += 1
                else:
                    cvg = 0
                
                # Check normalized convergence if requested
                if test_cvdNorm:
                    rel_chg_normalized = cc.relative_chg_normalized(self.loss["train"], idx_current=i)
                    print("rel_chg_normalized")
                    print(rel_chg_normalized)
                    if -(rel_chg_normalized) < tol_norm:
                        cvg_normalized += 1
                
                # Check if converged
                if cvg >= 2 or cvg_normalized >= 2:
                    self.converged = True
                    pickle_freq = i  # Ensure final pickling
                    self.loss = truncate_history(self.loss, i)
            
            # Print status if verbose
            if verbose:
                if Dval:
                    print(msg.format(i, trl, val_loss) + msg2)
                else:
                    print(msg.format(i, trl) + msg2)
        
        # Periodic checkpoint saving
        if i % ckpt_freq == 0:
            ptic, wtic = self.checkpoint(ckpt_mgr, process_time()-ptic, time()-wtic)
        
        # Periodic pickle saving
        if self.pickle_path and i % pickle_freq == 0:
            ptic, wtic = self.pickle(process_time()-ptic, time()-wtic)


def train_model(self, list_tro, list_Dtrain, list_D__, lr_reduce=0.5, maxtry=10, 
                verbose=True, ckpt_freq=50, test_cvdNorm=False, **kwargs):
    """
    Train the model with automatic learning rate adjustment.
    
    Args:
        list_tro (list): List of model trainers for each sample.
        list_Dtrain (list): List of training datasets.
        list_D__ (list): List of additional data for each sample.
        lr_reduce (float): Factor to reduce learning rate by on failure.
        maxtry (int): Maximum number of training attempts.
        verbose (bool): Whether to print status updates.
        ckpt_freq (int): Frequency of saving checkpoints.
        test_cvdNorm (bool): Whether to test normalized convergence.
        **kwargs: Additional arguments passed to _train_model_fixed_lr.
    """
    # Initialize timers
    ptic = process_time()
    wtic = time()
    tries = 0
    
    # Get initial epoch (usually zero, unless loaded from a pickle file)
    epoch0 = self.epoch.numpy()
    
    # Create a temporary directory for checkpoints
    with TemporaryDirectory() as ckpth:
        if verbose:
            print(f"Temporary checkpoint directory: {ckpth}")
        
        # Create a checkpoint manager
        mgr = tf.train.CheckpointManager(self.ckpt, directory=ckpth, max_to_keep=maxtry)
        
        # Main training loop
        while tries < maxtry:
            try:
                # Attempt to train the model with fixed learning rate
                self._train_model_fixed_lr(list_tro, list_Dtrain, list_D__, mgr, 
                                           ckpt_freq=ckpt_freq,
                                           **kwargs)
                
                # Check if training is complete
                if self.epoch >= len(self.loss["train"]) - 1:
                    break  # Finished training
            
            except (tf.errors.InvalidArgumentError, NumericalDivergenceError) as err:
                # Handle numerical instability or divergence
                
                # Plot and save the loss curve
                tr = np.array(self.loss["train"])
                plt.figure()
                plt.plot(tr, c='blue', label="train")
                plt.xlabel("epoch")
                plt.ylabel("ELBO loss")
                plt.title(f"Try {tries}")
                plt.savefig(f"loss{tries}.png")
                plt.show()
                plt.clf()
                
                # Increment try counter
                tries += 1
                print(f"tries: {tries}")
                
                # If maximum tries reached, raise the error
                if tries == maxtry:
                    raise err
                
                # Print verbose message about numerical instability
                if verbose:
                    msg = "{:04d} numerical instability (try {:d})"
                    print(msg.format(self.epoch.numpy(), tries))
                
                # Reset epoch and restore from checkpoint
                new_epoch = 0
                self.epoch.assign(0)
                ckpt = path.join(ckpth, f"ckpt-{new_epoch}")
                ptic, wtic = self.restore(ckpt)
                
                # Restore model for each sample
                nsample = len(list_Dtrain)
                for ksample in range(0, nsample):
                    with open(f'fit_{ksample+1}_restore.pkl', 'rb') as inp:
                        list_tro[ksample].model = pickle.load(inp)
                
                # Reduce learning rate
                lr_old, lr_new = self.multiply_lr(lr_reduce)
                if verbose:
                    msg = "{:04d} learning rate: {:.2e}"
                    print(msg.format(new_epoch, lr_new))
    
    # Print final training status
    if verbose:
        msg = "{:04d} training complete".format(self.epoch.numpy())
        if self.converged:
            print(msg + ", converged.")
        else:
            print(msg + ", reached maximum epochs.")
