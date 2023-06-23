# To plot memory benchmarking results

#%%
import pandas as pd
import matplotlib.pyplot as plt


#%%
dataFile = "results/memoryUsage_mouseSagittal_n3_iterL_runSeperately_SOB_062323.csv"
X = "L"
X_label = "L: num patterns"
Y = "peak_after_training"
Y_label = "Peak memory usage after training (GB)"
savefig = True
savename = "mouseSagittal_n3_iterL_SOB"

#%%
data = pd.read_csv(dataFile)
plt.plot(data[X], data[Y])
plt.title(savename)
plt.xlabel(X_label)
plt.ylabel(Y_label)

if savefig:
    plt.savefig("plots/" + savename+".png")
# %%
