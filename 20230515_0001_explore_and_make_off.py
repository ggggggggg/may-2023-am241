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

### get the filesnames we're going to use
try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()

def file_pattern(runnum):
    p = os.path.join(f"20230515",f"{runnum}",f"20230515_run{runnum}_chan*.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

pulse_files = file_pattern("1001") # 1000
noise_files = file_pattern("2001") # 2000

mass.line_models.VALIDATE_BIN_SIZE = False
data = mass.TESGroup(filenames=pulse_files, noise_filenames=noise_files,
max_chans=12, overwrite_hdf5_file=True)
data.set_all_chan_good()
data.summarize_data()
data.compute_noise()
data.avg_pulses_auto_masks()
data.compute_5lag_filter()
data.filter_data()
ds = data.first_good_dataset
# raise Exception()
# data.auto_cuts()
ds.calibration["p_filt_value"]=mass.EnergyCalibration()
ds.calibration["p_filt_value"].add_cal_point(np.median(ds.p_filt_value), 5.486e6)
data.convert_to_energy("p_filt_value")
ds.linefit(5.486e6, dlo=2e5, dhi=2e5, binsize=1e4)
data.summarize_filters(std_energy=5.486e6)
ds = data.first_good_dataset
plt.plot(ds.p_filt_value[:],".")
ds.plot_hist(np.arange(0,8,.1),"p_energy")
plt.xlabel("p_energy (MeV)")

data.auto_cuts(forceNew=True, nsigma_pt_rms=8)
ds.plot_traces(np.nonzero(~ds.good())[0])

plt.figure()
plt.plot(ds.p_timestamp[:]-ds.p_timestamp[0], ds.p_pretrig_mean[:], ".")

basename = mass.ljh_util.ljh_basename_channum(ds.filename)[0]
model_hdf5=f"{basename}_model.hdf5"
with h5py.File(model_hdf5,"w") as h5:
    mass.make_projectors(pulse_files=pulse_files,
        noise_files=noise_files,
        h5=h5,
        n_sigma_pt_rms=1000, # we want tails of previous pulses in our basis
        n_sigma_max_deriv=10,
        n_basis=7,
        maximum_n_pulses=5000,
        mass_hdf5_path=ds.hdf5_group.file.filename+"_for_make_projectors",
        mass_hdf5_noise_path=ds.noise_records.hdf5_group.file.filename+"_for_make_projectors",
        invert_data=False,
        optimize_dp_dt=False, # seems to work better for gamma data
        extra_n_basis_5lag=0, # mostly for testing, might help you make a more efficient basis for gamma rays, but doesn't seem neccesary
        noise_weight_basis=True) # only for testing, may not even work right to set to False


with h5py.File(model_hdf5,"r") as h5:
    models = {int(ch) : mass.pulse_model.PulseModel.fromHDF5(h5[ch]) for ch in h5.keys()}
models[2].plot()
output_dir = os.path.dirname(ds.filename)
os.makedirs(output_dir, exist_ok=True)
r = mass.ljh2off.ljh2off_loop(ljhpath = ds.filename,
    h5_path = model_hdf5,
    output_dir = output_dir,
    max_channels = 240,
    n_ignore_presamples = 0,
    require_experiment_state=False,
    show_progress=True)
ljh_filenames, off_filenames = r

# write a dummy experiment state file, since the data didn't come with one
with open(os.path.join(ds.filename[:-9]+"experiment_state.txt"),"w") as f:
    f.write("# placeholder comment\n")
    f.write("0, START\n")

dataoff = ChannelGroup(off_filenames)
dataoff.setDefaultBinsize(1e4) # set the default bin size in eV for fits
dsoff = dataoff[1] #2 for ch 2
dsoff.cutAdd("cutResidualStdDev", lambda residualStdDev: residualStdDev < 8, setDefault=False, overwrite=True)
dsoff.plotAvsB("relTimeSec", "residualStdDev")
dsoff.plotAvsB("relTimeSec", "residualStdDev", cutRecipeName="cutResidualStdDev", axis=plt.gca())
plt.legend(["all", "surive cutResidualStdDev"])

dsoff.learnDriftCorrection("pretriggerMean", "filtValue", cutRecipeName="cutResidualStdDev")

dsoff.calibrationPlanInit("filtValue")
dsoff.calibrationPlanAddPoint(np.median(dsoff.filtValue), "Am241", energy=5.486e6)
dsoff.calibrateFollowingPlan("filtValue", dlo=1e5, dhi=1e5)
dsoff.calibrateFollowingPlan("filtValueDC", dlo=1e5, dhi=1e5, calibratedName="energyDC")
dsoff.diagnoseCalibration()

dsoff.plotAvsB("pretriggerMean", "energy", cutRecipeName="cutResidualStdDev")
dsoff.plotAvsB("pretriggerMean", "energyDC", cutRecipeName="cutResidualStdDev", axis=plt.gca())
plt.legend(["energy", "energyDC"])

dsoff.linefit(5.486e6, dlo=1e5, dhi=1e5)

dsoff.cutAdd("cutROIandResidualStdDev", lambda cutResidualStdDev, energyDC: (energyDC > 5.1e6)&(energyDC<5.6e6)&(cutResidualStdDev), overwrite=True)
energyAndRelTimeSec = dsoff.getAttr(["relTimeSec","energyDC"], indsOrStates="START", cutRecipeName="cutNone")
energyAndRelTimeSec = np.vstack(energyAndRelTimeSec).T # convert to 2d array in right order for two column text file
cut_val = dsoff.getAttr("cutResidualStdDev", indsOrStates="START", cutRecipeName="cutNone")
cut_inds = np.nonzero(np.logical_not(cut_val))

energyAndRelTimeSecWithFake = energyAndRelTimeSec[:]
energyAndRelTimeSecWithFake[cut_inds,1] = 0
np.savetxt(os.path.join(ds.filename[:-9]+"listmode.txt"), energyAndRelTimeSecWithFake, header="time since first trigger (s), energy (eV)")


# look at the weird ones
inds = np.nonzero(~dsoff.cutResidualStdDev)[0]
plt.figure()
n = min(len(inds), 20)
ds.plot_traces(inds[:n])
plt.title(f"the {n} pulses cut by cutResidualStdDev")
print(f"number ommited by cutResidualStdDev = {len(inds)}")

inds = np.nonzero(~dsoff.cutROIandResidualStdDev)[0]
plt.figure()
n = min(len(inds), 20)
ds.plot_traces(n)
plt.title(f"the first {n} cut by cutROIandResidualStdDev {len(dsoff)}")

print(f"number ommited by cutROIandResidualStdDev = {len(inds)}")

n=len(inds) # don't limit to 20 as in the graph
counts = len(dsoff)-n
fulltime = dsoff.relTimeSec[-1]
print(f"count rate calculation [ignoring cutROIandResidualStdDev]= ({counts} counts)/({fulltime} s) = {counts/fulltime}")


