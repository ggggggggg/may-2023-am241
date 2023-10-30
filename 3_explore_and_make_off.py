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
import csv
import truebqlines
plt.close("all")
plt.ion()



## +++++ BEGIN INPUTS +++++ ##


my_dir = "/home/pcuser/data"
my_folder = "20231017"
my_runnum = "0000"
my_chan = "3" # to do : implement make this "*" for processing all channels

## +++++ _END_ INPUTS +++++ ##



### get the filesnames we're going to use
try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()


def file_pattern(runnum):
 #   p = os.path.join("/home","pcuser","data",f"20230913",f"{runnum}",f"20230913_run{runnum}_chan4.ljh")
    p = os.path.join(f"{my_dir}",f"{my_folder}",f"{runnum}",f"{my_folder}_run{runnum}_chan{my_chan}.ljh")
 #   p = os.path.join(f"20230517",f"{runnum}",f"20230517_run{runnum}_chan*.ljh")
 # p = os.path.join(f"20230912",f"{runnum}",f"20230912_run{runnum}_chan1.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

pulse_files = file_pattern(str(int(my_runnum)+1000)) # 1000 runnum orig + 1000
noise_files = file_pattern(str(int(my_runnum)+2000)) # 2000 runnum orig + 2000

mass.line_models.VALIDATE_BIN_SIZE = False
data = mass.TESGroup(filenames=pulse_files, noise_filenames=noise_files,
max_chans=12, overwrite_hdf5_file=True)
data.set_all_chan_good()
ds = data.channel[int(my_chan)] # NEED TO UPDATE THIS WITH CHANNEL
ds.summarize_data(use_cython=True)
data.compute_noise()
# for ds in data:
#     ds.compute_noise_nlags(n_lags=100000)
data.avg_pulses_auto_masks()
data.compute_5lag_filter()
data.filter_data()
ds = data.first_good_dataset
# raise Exception()
# data.auto_cuts()
ds.calibration["p_filt_value"]=mass.EnergyCalibration()
ds.calibration["p_filt_value"].add_cal_point(np.median(ds.p_filt_value), 5637.82e3)
data.convert_to_energy("p_filt_value")
data.summarize_filters(std_energy=5637.82e3)
ds = data.first_good_dataset # RPF - is this a problem
plt.figure()
plt.plot(ds.p_filt_value[:],".")
plt.xlabel("pulse index")
plt.ylabel("p_filt_value")

result = ds.linefit("Am241Q", binsize=1e3, dlo=2e5, dhi=2e5, has_tails=True)




# raise Exception()

# Why can't I save histogram as csv?
#np.savetxt("p_energy_test.txt",ds.p_energy[:])
counts1,en1=np.histogram(ds.p_energy,bins=300,range=(5.4E6,5.7E6))
np.savetxt(os.path.join(ds.filename[:-4]+"_p_energy_hist.txt"),counts1,header="300 5.4E6 5.7E6")

data.auto_cuts(forceNew=True, nsigma_pt_rms=8)
ds.plot_traces(np.nonzero(~ds.good())[0])

plt.figure()
plt.plot(ds.p_timestamp[:]-ds.p_timestamp[0], ds.p_pretrig_mean[:], ".")
plt.xlabel("time since start (s)")
plt.ylabel("p_pretrig_mean (arb)")

# %matplotlib qt to load figure in background /zoom etc.
basename = mass.ljh_util.ljh_basename_channum(ds.filename)[0]
model_hdf5=f"{basename}_model.hdf5"
with h5py.File(model_hdf5,"w") as h5:
    mass.make_projectors(pulse_files=pulse_files,
        noise_files=noise_files,
        h5=h5,
        n_sigma_pt_rms=1000, # we want tails of previous pulses in our basis 1000
        n_sigma_max_deriv=10, # 10
        n_basis=10, # 7
        maximum_n_pulses=5000, # 5000
        mass_hdf5_path=ds.hdf5_group.file.filename+"_for_make_projectors",
        mass_hdf5_noise_path=ds.noise_records.hdf5_group.file.filename+"_for_make_projectors",
        invert_data=False, #False
        optimize_dp_dt=False, # seems to work better for gamma data
        extra_n_basis_5lag=0, # mostly for testing, might help you make a more efficient basis for gamma rays, but doesn't seem neccesary
        noise_weight_basis=True) # only for testing, may not even work right to set to False


with h5py.File(model_hdf5,"r") as h5:
    models = {int(ch) : mass.pulse_model.PulseModel.fromHDF5(h5[ch]) for ch in h5.keys()}
models[ds.channum].plot()
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
for channum, dsoff in dataoff.items():
    # define recipes for "filtValue5Lag", "peakX5Lag" and "cba5Lag"
    # where cba refers to the coefficiencts of a polynomial fit to the 5 lags of the filter
    filter_5lag = models[channum].f_5lag
    dsoff.add5LagRecipes(filter_5lag)

dataoff.setDefaultBinsize(2e3) # set the default bin size in eV for fits
dsoff = dataoff[ds.channum]
dsoff.cutAdd("cutResidualStdDev", lambda residualStdDev: residualStdDev < 15, setDefault=False, overwrite=True)
dsoff.plotAvsB("relTimeSec", "residualStdDev")
dsoff.plotAvsB("relTimeSec", "residualStdDev", cutRecipeName="cutResidualStdDev", axis=plt.gca())
plt.legend(["all", "surive cutResidualStdDev"])

dsoff.learnDriftCorrection("pretriggerMean", "filtValue5Lag", cutRecipeName="cutResidualStdDev")

dsoff.calibrationPlanInit("filtValue5Lag")
dsoff.calibrationPlanAddPoint(np.median(dsoff.filtValue), "Am241Q")
dsoff.calibrateFollowingPlan("filtValue5Lag", dlo=1e5, dhi=1e5)
dsoff.calibrateFollowingPlan("filtValue5LagDC", dlo=1e5, dhi=1e5, calibratedName="energyDC")
dsoff.diagnoseCalibration()

#Plot energyDC over time; Should do this for uncorrected energy too to look for drift

dsoff.plotAvsB("relTimeSec","energy")
dsoff.plotAvsB("relTimeSec","energyDC")
plt.legend(["energy", "energyDC"])
plt.title("Pulse amplitude vs time")
# to do, save this or find a way to average is



dsoff.plotAvsB("pretriggerMean", "energy", cutRecipeName="cutResidualStdDev")
dsoff.plotAvsB("pretriggerMean", "energyDC", cutRecipeName="cutResidualStdDev", axis=plt.gca())
plt.legend(["energy", "energyDC"])
result = dsoff.linefit("Am241Q", dlo=1e5, dhi=1e5, plot=False)
result2 = dsoff.linefit("Am241Q", dlo=1e5, dhi=1e5, has_tails=True)

dsoff.cutAdd("cutROIandResidualStdDev", lambda cutResidualStdDev, energyDC: (energyDC > 5.1e6)&(energyDC<5.6e6)&(cutResidualStdDev), overwrite=True)
energyAndRelTimeSec = dsoff.getAttr(["relTimeSec","energyDC"], indsOrStates="START", cutRecipeName="cutNone")
energyAndRelTimeSec = np.vstack(energyAndRelTimeSec).T # convert to 2d array in right order for two column text file
cut_val = dsoff.getAttr("cutResidualStdDev", indsOrStates="START", cutRecipeName="cutNone")
cut_inds = np.nonzero(np.logical_not(cut_val))
energyAndRelTimeSecWithFake = energyAndRelTimeSec[:]
energyAndRelTimeSecWithFake[cut_inds,1] = 0
np.savetxt(os.path.join(ds.filename[:-4]+"listmode.txt"), energyAndRelTimeSecWithFake, header="time since first trigger (s), energy (eV)")


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


#-----------------#
# plot histograms #
#-----------------#

# to do: save histograms #
dsoff.plotHist(np.arange(5.55,5.75,0.0005)*1e6, "energy",cutRecipeName=None)
#dsoff.plotHist(np.arange(5.55,5.75,0.001)*1e6, "energy", cutRecipeName="cutResidualStdDev", axis=plt.gca())
dsoff.plotHist(np.arange(5.55,5.75,0.0005)*1e6, "energyDC", cutRecipeName=None, axis=plt.gca())
dsoff.plotHist(np.arange(5.55,5.75,0.0005)*1e6, "energyDC", cutRecipeName="cutResidualStdDev", axis=plt.gca())
# dsoff.linefit(lineNameOrEnergy="Am241Q", attr="energyDC",dlo=1.e4, dhi=1.2e4,cutRecipeName="cutResidualStdDev", axis=plt.gca())
plt.legend(["No cuts","rsd cut", "energyDC","rsd cut+eNergyDC"])
plt.xlabel("energy or energyDC")
plt.grid(True)
dsoff.plotHist(np.arange(5.55,5.75,0.0005)*1e6, "energyDC", cutRecipeName="cutResidualStdDev")

# save histogram
counts1,en1=np.histogram(dsoff.energyDC[:],bins=800,range=(5.4E6,5.8E6))
np.savetxt(os.path.join(ds.filename[:-4]+"_p_energy_hist_dsof.txt"),counts1,header="800 5.4E6 5.8E6")

# NOW GAMMA RANGE
dsoff.plotHist(np.arange(0,200,0.2)*1e3, "energy",cutRecipeName=None)
dsoff.plotHist(np.arange(0,200,0.2)*1e3, "energyDC", cutRecipeName=None, axis=plt.gca())
dsoff.plotHist(np.arange(0,200,0.2)*1e3, "energyDC", cutRecipeName="cutResidualStdDev", axis=plt.gca())

counts1,en1=np.histogram(dsoff.energyDC[:],bins=500,range=(0,200E3))
np.savetxt(os.path.join(ds.filename[:-4]+"_p_energy_hist_dsof_g.txt"),counts1,header="500 0 200E3")

counts1,en1=np.histogram(dsoff.energyDC[:],bins=10000,range=(0,6E6)) # wide histogram
np.savetxt(os.path.join(ds.filename[:-4]+"_p_energy_hist_dsof_w.txt"),counts1,header="10000 0 6E6")