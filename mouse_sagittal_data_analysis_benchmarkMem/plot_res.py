# To plot memory benchmarking results

#%%
import pandas as pd
import matplotlib.pyplot as plt


#%%
dataFile = "results/memoryUsage_mouseSagittal_n3_iterL_runSeperately_SOB_062323.csv"
X = "L"
Y = "peak_after_training"
savefig = True
savename = "mouseSagittal_n3_iterL_SOB"

#%%
data = pd.read_csv(dataFile)
plt.plot(data[X], data[Y])
plt.title(savename)

if savefig:
    plt.savefig("plots/" + savename+".png")
# %%
