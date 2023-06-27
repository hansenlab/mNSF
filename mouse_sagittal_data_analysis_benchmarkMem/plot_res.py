# To plot memory benchmarking results

#%%
import pandas as pd
import matplotlib.pyplot as plt


#%%
#dataFile = "results/memoryUsage_mouseSagittal_n3_iterL_runSeperately_SOB_062323.csv"
#X = "L"
#X_label = "L: num patterns"
#Y = "peak_after_training"
#Y_label = "Peak memory usage after training (GB)"
#savefig = True
#savename = "mouseSagittal_n3_iterL_SOB"

#%%
#data = pd.read_csv(dataFile)
#plt.plot(data[X], data[Y])
#plt.title(savename)
#plt.xlabel(X_label)
#plt.ylabel(Y_label)

#if savefig:
#    plt.savefig("plots/" + savename+".png")


# %%
dataFile = "results/memoryUsage_mouseSagittal_n3_lr1e4_epoch100_tol0_iterL_runSeperately_SOB_062623.csv"
data_full = pd.read_csv(dataFile)
data = data_full.groupby("L").mean().reset_index()

X = "L"
X_label = "L: num patterns"
Y = "peak_after_training"
Y_label = "Peak memory usage after training (GB)"
savefig = True
savename = "mouseSagittal_n3_epoch100_lr1e4_tol0_iterL_SOB"

plt.plot(data[X], data[Y]) # plot averages with line
plt.scatter(data_full[X], data_full[Y]) # plot all points
plt.title(savename)
plt.xlabel(X_label)
plt.ylabel(Y_label)

if savefig:
    plt.savefig("plots/" + savename+".png")



# %%
