import gc
from pathlib import Path

import pandas as pd

from mNSF import process_multiSample, training_multiSample

########################################################################
########################################################################
L = 12
dir_output = Path(".")
pth = Path("/dcs04/hansen/data/ywang/ST/data_10X_ST/mouse_Sagittal_spaceRanger1_1_0/out/")

## nsample = 2:
# worked on L12

##nsample = 4:
# not work on L=12 or 15: error shows in the 2nd iteration in gradient cal
# put the par_storage into a function, so that the mem could be relived?

nsample = 4
(pth / "models").mkdir(parents=True, exist_ok=True)
(pth / "models" / "pp").mkdir(parents=True, exist_ok=True)
# list_fit[0].generate_pickle_path("constant",base=mpth)

########################################################################3
################### step 0  Data loading
########################################################################

list_D = list()
list_X = list()
for ksample in range(1, nsample + 1):
    list_D.append(
        process_multiSample.get_D(
            X := pd.read_csv(pth / "X_sample" / f"{ksample}.csv"),
            Y := pd.read_csv(pth / "Y_sample" / f"{ksample}.csv"),
        )
    )
    list_X.append(X)


list_Dtrain = process_multiSample.get_listDtrain(list_D)
list_sampleID = process_multiSample.get_listSampleID(list_D)


########################################################################3
################### step 1 initialize model
########################################################################

list_fit = process_multiSample.ini_multiSample(list_D, L)


########################################################################
################### step 2 fit model
########################################################################

gc.collect()

training_multiSample.train_model_mNSF(list_fit, pp, list_Dtrain, list_D)

gc.collect()

# during training:
# print(tf.config.experimental.get_memory_info('GPU:0'))
# {'current': 4684887040, 'peak': 29394932480}

# save the fitted model
process_multiSample.save_object(list_fit, dir_output / "list_fit.pkl")

gc.collect()


########################################################################3
################### step 3 save and plot results
########################################################################
## save the loadings
# loadings=visualize.get_loadings(list_fit[0])
# DF = pd.DataFrame(loadings)
# DF.to_csv(("loadings.csv"))
inpf12 = process_multiSample.interpret_npf_v3(list_fit, list_X, S=100, lda_mode=False)

W = inpf12["loadings"]
# Wdf=pd.DataFrame(W*inpf12["totals1"][:,None], index=ad.var.index, columns=range(1,L+1))
Wdf = pd.DataFrame(W * inpf12["totals1"][:, None], columns=range(1, L + 1))
Wdf.to_csv(dir_output / "loadings_spde.csv")


## save the factors
# inpf12 = process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
Factors = inpf12["factors"][:, 0:L]

for k in range(nsample):
    indices = list_sampleID[k].astype(int)
    Factors_df = pd.DataFrame(Factors[indices, :])
    Factors_df.to_csv(dir_output / f"factors_sample{k + 1:02d}.csv")
