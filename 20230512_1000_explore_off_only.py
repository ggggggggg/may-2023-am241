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
plt.close("all")
plt.ion()

pulse_files = ["20230512//1000/20230512_run1000_chan2.off"]
ljh = mass.files.LJHFile('20230512/1000/20230512_run1000_chan2.ljh')
dataoff = ChannelGroup(pulse_files)
dataoff.setDefaultBinsize(2.5e3) # set the default bin size in eV for fits
dsoff = dataoff[2]
dsoff.plotHist(np.arange(0,40,0.5), "residualStdDev")
dsoff.cutAdd("cutResidualStdDev", lambda residualStdDev: residualStdDev < 10, setDefault=False, overwrite=True)
dsoff.cutAdd("cutROI", lambda filtValue: (filtValue > 1300) & (filtValue < 1500))
dsoff.cutAdd("cutROIandResidualStdDev", lambda cutROI, cutResidualStdDev: cutROI & cutResidualStdDev)
dsoff.plotAvsB("relTimeSec", "residualStdDev")
dsoff.plotAvsB("relTimeSec", "residualStdDev", cutRecipeName="cutROIandResidualStdDev", axis=plt.gca())
plt.legend(["all", "surive cutROIandResidualStdDev"])

dsoff.learnDriftCorrection("pretriggerMean", "filtValue", cutRecipeName="cutROIandResidualStdDev")


dsoff.calibrationPlanInit("filtValue")
dsoff.calibrationPlanAddPoint(1441, "Am241", energy=5.486e6)
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
energyAndRelTimeSec = dsoff.getAttr(["relTimeSec","energyDC"], indsOrStates="START", cutRecipeName="cutResidualStdDev")
np.savetxt("jan2023_10_min_data_try1.txt", energyAndRelTimeSec, header="time since first trigger (s), energy (eV)")


def plot_blocks_by_residualStdDev(inds, title, blocksize=20, n_blocks=4):
    residualStdDev = dsoff.residualStdDev[inds]
    sort_inds = np.argsort(residualStdDev)
    sorted_inds = inds[sort_inds]
    sorted_residualStdDev = residualStdDev[sort_inds]
    for i_block in range(n_blocks):
        plt.figure(figsize=(8,4))
        plt.title(f"{title}\nblock {i_block}")
        for i in range(blocksize):
            j = i_block*blocksize+i
            if j >= len(inds):
                return
            ind = sorted_inds[j]
            r = sorted_residualStdDev[j]
            trace = ljh.read_trace(ind)
            plt.plot(trace, label=f"ind={ind} resid_$\sigma$={r:.2f}")
        plt.legend()
            

# look at the weird ones
inds = np.nonzero(~dsoff.cutResidualStdDev)[0]
plot_blocks_by_residualStdDev(inds[::80], 
title=f"the {len(inds)} pulses excluded from timestamp/energy list by residualStdDev only")
# plt.figure()
# for ind in inds:
#     trace = ljh.read_trace(ind)
#     plt.plot(trace, label=f"ind={ind}")
# plt.title(f"the {len(inds)} pulses excluded from timestamp/energy list by residualStdDev only")

# inds = np.nonzero(~dsoff.cutROIandResidualStdDev)[0]
# plt.figure()
# for ind in inds:
#     trace = ljh.read_trace(ind)
#     plt.plot(trace, label=f"ind={ind}")
# assert ljh.nPulses == len(dsoff)
# plt.title(f"the {len(inds)} pulses from the total count of {len(dsoff)-len(inds)} by residualStdDev and energy")

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


dsoff.plotHist(np.arange(0,20,0.1), "residualStdDev")
plt.grid(True)
plt.yscale("log")

dsoff.plotHist(np.arange(0,8,0.01)*1e6, "energy")
plt.grid(True)
plt.yscale("log")

dsoff.plotHist(np.arange(0,8,0.01)*1e6, "energy", cutRecipeName="cutResidualStdDev")
plt.grid(True)
plt.yscale("log")

dsoff.plotAvsB2d("pretriggerDelta","energy", 
(np.arange(-40,10,2.5), np.linspace(5.3e6, 5.6e6,100)),
norm=plt.matplotlib.colors.LogNorm())

dsoff.plotAvsB2d("pretriggerDelta","energyDC", 
(np.arange(-40,10,2.5), np.linspace(5.3e6, 5.6e6,100)),
norm=plt.matplotlib.colors.LogNorm())