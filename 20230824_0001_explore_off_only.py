import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
import os
import h5py
import matplotlib as mpl
import pylab as plt
import numpy as np
import lmfit
from dataclasses import dataclass
from typing import List
from collections import OrderedDict
import devel
import ljhfiles
plt.close("all")
plt.ion()
basepath = "20230824//1001//20230824_run1001_chan1"
pulse_files = [basepath+".off"]
ljh = ljhfiles.LJHFile(basepath+".ljh")
dataoff = ChannelGroup(pulse_files)
dataoff.setDefaultBinsize(2.5e3) # set the default bin size in eV for fits
dsoff = dataoff[2]
dsoff.plotHist(np.arange(0,10000,10), "filtValue")
dsoff.plotHist(np.arange(0,40,0.5), "residualStdDev")
med_ph = np.median(dsoff.filtValue)
dsoff.cutAdd("cutResidualStdDev", lambda residualStdDev: residualStdDev < 7, setDefault=False, overwrite=True)
dsoff.cutAdd("cutROI", lambda filtValue: (filtValue > med_ph*0.9) & (filtValue < med_ph*1.1))
dsoff.cutAdd("cutROIandResidualStdDev", lambda cutROI, cutResidualStdDev: cutROI & cutResidualStdDev)
dsoff.plotAvsB("relTimeSec", "residualStdDev")
dsoff.plotAvsB("relTimeSec", "residualStdDev", cutRecipeName="cutROIandResidualStdDev", axis=plt.gca())
plt.legend(["all", "surive cutROIandResidualStdDev"])

dsoff.learnDriftCorrection("pretriggerMean", "filtValue", cutRecipeName="cutROIandResidualStdDev")


dsoff.calibrationPlanInit("filtValue")
dsoff.calibrationPlanAddPoint(med_ph, "Am241", energy=5.486e6)
dsoff.calibrateFollowingPlan("filtValue", dlo=1e5, dhi=1e5)
dsoff.calibrateFollowingPlan("filtValueDC", dlo=1e5, dhi=1e5, calibratedName="energyDC")
dsoff.diagnoseCalibration()

dsoff.plotAvsB("pretriggerMean", "energy", cutRecipeName="cutROIandResidualStdDev")
dsoff.plotAvsB("pretriggerDelta", "energy", cutRecipeName="cutROIandResidualStdDev")
dsoff.plotAvsB("pretriggerDelta", "energyDC", cutRecipeName="cutROIandResidualStdDev", axis=plt.gca())
plt.legend(["energy", "energyDC"])

dsoff.plotAvsB("energy", "residualStdDev")


dsoff.linefit(5.486e6, dlo=1e5, dhi=1e5)
dsoff.linefit(5.486e6, dlo=1e5, dhi=1e5, attr="energyDC")


# dsoff.cutAdd("cutROIandResidualStdDev", lambda cutResidualStdDev, energyDC: (energyDC > 5.3e6)&(energyDC<5.6e6)&(cutResidualStdDev), overwrite=True)
# RPF - make text file of time & energyDC with cuts: to do- save in efficient binary format?
energyAndRelTimeSec = dsoff.getAttr(["relTimeSec","energyDC"], indsOrStates="START", cutRecipeName="cutResidualStdDev")
energyAndRelTimeSec = np.vstack(energyAndRelTimeSec).T # convert to 2d array in right order for two column text file
np.savetxt(basepath+"listmode.txt", energyAndRelTimeSec, header="time since first trigger (s), energy (eV)")
def ceildiv(a, b):
     return -(a // -b)


def plot_blocks_by_residualStdDev(inds, title, blocksize=20, max_n_blocks=4):
    residualStdDev = dsoff.residualStdDev[inds]
    sort_inds = np.argsort(residualStdDev)
    sorted_inds = inds[sort_inds]
    sorted_residualStdDev = residualStdDev[sort_inds]
    n_blocks = min(ceildiv(len(inds), blocksize), max_n_blocks)
    for i_block in range(n_blocks):
        plt.figure(figsize=(8,4))
        plt.title(f"{title}\nblock {i_block}")
        for i in range(blocksize):
            j = i_block*blocksize+i
            if j >= len(inds):
                break
            ind = sorted_inds[j]
            r = sorted_residualStdDev[j]
            trace = ljh.read_trace(ind)
            plt.plot(trace, label=f"ind={ind} resid_$\sigma$={r:.2f}")
        plt.legend()
            

# look at the weird ones
inds = np.nonzero(~dsoff.cutResidualStdDev)[0]
skipstep = max(len(inds)//100,1)

plot_blocks_by_residualStdDev(inds[::skipstep], 
title=f"the {len(inds)} pulses excluded from timestamp/energy list by residualStdDev only")

for i in range(dsoff.offFile["extraCoefs"].shape[1]):
    plt.figure()
    plt.plot(dsoff.offFile.basis[:,3+i])
    plt.title(f"extraCoefs{i} associated basis")
    inds_extra_coefs_temp = np.argsort(dsoff.offFile["extraCoefs"][:,i])[-20:]
    plot_blocks_by_residualStdDev(inds_extra_coefs_temp, title=f"extraCoefs{i} top 20")


fulltime = dsoff.relTimeSec[-1]
print(f"total triggers = {len(dsoff)}")
print(f"high residual stddev, aka 1, 0, or 2 counts, and deadtime? = {len(inds)}")
print(f"elapsed time (approx) = {fulltime:0.2f} s")
print(f"count rate calculation with total triggers and no deadtime correction\n = ({len(dsoff)} counts)/({fulltime:2f} s) = {len(dsoff)/fulltime:.4f} Bq")

dsoff.plotAvsB2d("relTimeSec", "pretriggerMean", 
[np.linspace(0,dsoff.relTimeSec[-1],101), np.linspace(-50,50,40)+np.median(dsoff.pretriggerMean)])

dsoff.plotAvsB2d("relTimeSec", "energyDC", 
[np.linspace(0,dsoff.relTimeSec[-1],101), np.linspace(-1e5,1e5,11)+np.median(dsoff.energy)])

dsoff.plotAvsB2d("pretriggerMean", "energyDC", 
[np.linspace(-50,50,40)+np.median(dsoff.pretriggerMean), np.linspace(-1e5,1e5,11)+np.median(dsoff.energy)])


dsoff.plotHist(np.arange(0,40,0.1), "residualStdDev")
plt.grid(True)
plt.yscale("log")

dsoff.plotHist(np.arange(0,12,0.01)*1e6, "energy")
plt.grid(True)
plt.yscale("log")

dsoff.plotHist(np.arange(0,12,0.01)*1e6, "energy", cutRecipeName="cutResidualStdDev")
plt.grid(True)
plt.yscale("log")


dsoff.plotAvsB2d("pretriggerDelta","energy", 
(np.arange(-40,10,2.5), np.linspace(5.3e6, 5.6e6,100)),
norm=plt.matplotlib.colors.LogNorm())

dsoff.plotAvsB2d("pretriggerDelta","energyDC", 
(np.arange(-40,10,2.5), np.linspace(5.3e6, 5.6e6,100)),
norm=plt.matplotlib.colors.LogNorm())