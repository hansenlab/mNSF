

import psutil

################################################################################################################################
# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1)
# Squeeze to take the shape from [50, 1] to [50].
observed_values = f(observed_index_points)

# Define a kernel with trainable parameters.
kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
    length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'))

gp = tfd.GaussianProcess(kernel, observed_index_points)

optimizer = tf.optimizers.Adam()

#@tf.function
def optimize():
  with tf.GradientTape() as tape:
    loss = -gp.log_prob(observed_values)
  grads = tape.gradient(loss, gp.trainable_variables)
  optimizer.apply_gradients(zip(grads, gp.trainable_variables))
  return loss

for i in range(1000):
  neg_log_likelihood = optimize()
  if i % 100 == 0:
    print("Step {}: NLL = {}".format(i, neg_log_likelihood))
print("Final NLL = {}".format(neg_log_likelihood))



def sinusoid(x):
  return np.sin(3 * np.pi * x[..., 0])

def generate_1d_data(num_training_points, observation_noise_variance):
  """Generate noisy sinusoidal observations at a random set of points.

  Returns:
     observation_index_points, observations
  """
  index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
  index_points_ = index_points_.astype(np.float64)
  # y = f(x) + noise
  observations_ = (sinusoid(index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  return index_points_, observations_

def generate_2d_data(num_training_points=100, observation_noise_variance=0.1):
  """Generate noisy sinusoidal observations at a random set of points.

  Returns:
     observation_index_points, observations
  """
  index_points_ = np.random.uniform(-1., 1., (num_training_points, 2))
  index_points_ = index_points_.astype(np.float64)
  # y = f(x) + noise
  observations_ = (sinusoid(index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  list_out=[None]*2
  list_out[0]=index_points_
  list_out[1]=observations_
  return list_out

# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
NUM_TRAINING_POINTS = 100
list_out_= generate_2d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.1)

observation_index_points_=list_out_[0]
observations_=list_out_[1]

def build_gp(amplitude, length_scale, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""
  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})

x = gp_joint_model.sample()
lp = gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var]]

def target_log_prob(amplitude, length_scale, observation_noise_variance):
  return gp_joint_model.log_prob({
      'amplitude': amplitude,
      'length_scale': length_scale,
      'observation_noise_variance': observation_noise_variance,
      'observations': observations_
})

# Now we optimize the model parameters.
num_iters = 10
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Use `tf.function` to trace the loss for more efficient evaluation.
#@tf.function(autograph=False, jit_compile=False)
def train_model():
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var, length_scale_var,
                            observation_noise_variance_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  return loss

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  loss = train_model()
  print(i)
  lls_[i] = loss


print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
#1.0452804565429688



# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
#predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
list_out_pred= generate_2d_data(
    num_training_points=100,
    observation_noise_variance=observation_noise_variance_var._value().numpy())

predictive_index_points=list_out_pred[0]
predictive_index_points.shape

predictive_index_points=observation_index_points_

# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
#predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = 10
samples = gprm.sample(num_samples)









################################
# addional parameters in NSF
# loading, L times of the GP parameters

# additional variables in NSF
# factors

################################################################
################################################################
## spatial Real-valued NSF
################################################################
################################################################

##########
########## step1: intialization using nmf
##########
## 
#%%
L=3
#Y=observations_.to_numpy()
import pandas as pd
Y = pd.read_csv('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/Jan3_2022_LukasData_mgcv_correctedModel/data/counts_sample_tmp_HVG_t_sample5_shared.csv')
Y=Y.to_numpy()
Y=Y[0:600,0:200]

from sklearn.decomposition import NMF
fit = NMF(L,beta_loss="kullback-leibler",solver="mu",init="nndsvda")
Fplot = fit.fit_transform(Y)
#Fplot = fit.fit_transform(Y)
#hmkw = {"figsize":(4,.9),"bgcol":"white","subplot_space":0.1,"marker":"s","s":10}
#fig,axes=visualize.multiheatmap(X, Fplot, (1,4), cmap="Blues", **hmkw)
ker = tfk.MaternThreeHalves
fit=pf.ProcessFactorization(J,L,D1['X'],psd_kernel=ker,nonneg=True,lik="poi")
fit.init_loadings(D["Y"],X=D['X'],sz=D["sz"],shrinkage=0.3)
loadings_ini=visualize.get_loadings(fit)

########################################################################
########################################################################
########################################################################

####################
#################### step2: format the loss function
####################
#######step 2-1: add additional training parameters
loadings_var = tfp.util.TransformedVariable(
    initial_value=loadings_ini,
    bijector=constrain_positive,
    name='loadings_var',
    dtype=np.float64)

amplitude_var_factor1 = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude_factor1',
    dtype=np.float64)

length_scale_var_factor1 = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale_factor1',
    dtype=np.float64)

observation_noise_variance_var_factor1 = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var_factor1',
    dtype=np.float64)

amplitude_var_factor2 = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude_factor2',
    dtype=np.float64)

length_scale_var_factor2 = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale_factor2',
    dtype=np.float64)

observation_noise_variance_var_factor2 = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var_factor2',
    dtype=np.float64)


trainable_variables_factor1 = [v.trainable_variables[0] for v in 
                       [amplitude_var_factor1,
                       length_scale_var_factor1,
                       observation_noise_variance_var_factor1]]

trainable_variables_factor2 = [v.trainable_variables[0] for v in 
                       [amplitude_var_factor1,
                       length_scale_var_factor1,
                       observation_noise_variance_var_factor1]]

trainable_variables_sampleSpecific=trainable_variables

####### step 2-2 create loss function: sum of all gp likelihood of all factors


def train_model():
  with tf.GradientTape() as tape:
    #### loss function for the GP part
    loss1_gp = -target_log_prob(amplitude_var_factor1, length_scale_var_factor1,
                            observation_noise_variance_var_factor1)
    loss2_gp = -target_log_prob(amplitude_var_factor2, length_scale_var_factor2,
                            observation_noise_variance_var_factor2)
    #### loss function for the poisson part
    ##factor1
    optimized_kernel1 = tfk.ExponentiatedQuadratic(amplitude_var_factor1, length_scale_var_factor1)
    loss1_poisson = gp.log_prob(Y,)
    #gprm1 = tfd.GaussianProcessRegressionModel(
    #  kernel=optimized_kernel1,
    #  index_points=X,
    #  observation_index_points=X,
    #  observations=F1,
    #  observation_noise_variance=observation_noise_variance_var_factor1,
    #  predictive_noise_variance=0.)
    ##factor2
    optimized_kernel2 = tfk.ExponentiatedQuadratic(amplitude_var_factor2, length_scale_var_factor2)
    #gprm2 = tfd.GaussianProcessRegressionModel( #posterior distribution of the factor at each location
    #  kernel=optimized_kernel2,
    #  index_points=X,
    #  observation_index_points=X,
    #  observations=F2,
    #  observation_noise_variance=observation_noise_variance_var_factor2,
    #  predictive_noise_variance=0.)
    num_samples_ = 1
    samples_factor1 = gprm.sample(num_samples_) # draw only one sample here, should draw more
    predicted_Y_factor1=loading*
    #poissonP = tfd.Poisson(rate=loadingp[]+samples[1])
    #loss_factor1=poissonP()
    #loss1_poisson=
    loss = -gp.log_prob(observed_values)

  grads1 = tape.gradient(loss1, trainable_variables_factor1)
  grads2 = tape.gradient(loss2, trainable_variables_factor2)
  optimizer.apply_gradients(zip(grads1, trainable_variables_factor1))
  optimizer.apply_gradients(zip(grads2, trainable_variables_factor2))
  #optimizer.apply_gradients(zip(grads1+grads2, loadings_var))
  optimizer.apply_gradients(zip(grads1, loadings_var))
  optimizer.apply_gradients(zip(grads2, loadings_var))
  return loss1+loss2

####### step 2-3: train the model
#Store the likelihood values during training, so we can plot the progress
num_iters=1
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  loss = train_model()
  print(i)
  lls_[i] = loss





######################################################################
## step 2-X get predicted gene expression, lambda
## step 2-X get poisson likelihood of the observed data





################################
# Will's model fitting method：
# it uses EM algorithm which theoretically makes sure (?) the algorithm converges to the MLE


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#%% Data loading
from scanpy import read_h5ad
from os import path


#%% Data loading
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_factor2_to_12/nsf-paper-main_factor1/')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/April8_2022_NSF/nsf-paper-main_8_factor2_to_12/nsf-paper-main_factor1/')
#from models import cf,pf,pfh
#from models import cf,pf,pfh
rng = np.random.default_rng()
dtp = "float32"
pth = "simulations/ggblocks_lr"

dpth = path.join(pth,"data")
mpth = path.join(pth,"models")
plt_pth = path.join(pth,"results/plots")
misc.mkdir_p(plt_pth)

ad = read_h5ad(path.join(dpth,"ggblocks_lr_sub500_topleft_layer2to5_500genes.h5ad"))
J = ad.shape[1]
D,_ = preprocess.anndata_to_train_val(ad, layer="counts", train_frac=1.0,
                                      flip_yaxis=False)
D_n,_ = preprocess.anndata_to_train_val(ad,train_frac=1.0,flip_yaxis=False)
fmeans,D_c,_ = preprocess.center_data(D_n)
X = D["X"] #note this should be identical to Dtr_n["X"]
N = X.shape[0]
Dtf = preprocess.prepare_datasets_tf(D,Dval=None,shuffle=False)
Dtf_n = preprocess.prepare_datasets_tf(D_n,Dval=None,shuffle=False)
Dtf_c = preprocess.prepare_datasets_tf(D_c,Dval=None,shuffle=False)


#%% functions loading
import os
import sys
os.chdir('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_byBatches')
sys.path.append('/dcs04/legacy-dcs01-hansen/hansen_lab1/ywang/ST/May6_2022_sNMF_2samples/nsf-paper-main_2samples_rotate_regularSizeData_SSlayer_byBatches')
from utils import preprocess,misc,training,visualize,postprocess


from utils import preprocess,misc,training,visualize,postprocess

from models import cf,pf,pfh


import numpy as np

from os import path
from scanpy import read_h5ad
from tensorflow_probability import math as tm
tfk = tm.psd_kernels

from models import cf,pf,pfh
from utils import preprocess,misc,training,visualize,postprocess
from models import pf_fit12

from models import pf_fit12_500spots_700spots_perSample_weighed_likelihood




