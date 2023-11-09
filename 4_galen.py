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
import ljhfiles
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
        n_basis=9, # 7
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


# we're going to try to sort the pulses in 4 bins
# A = 0 - foil pulses, energy assigned
# B = 1 - foil pulses with pile up, no energy assigned
# C = 2 - foil + non-foil, no enegy assigned
# D = 3 - non-foil, no energy assigned

plt.figure()
plt.plot(dsoff.offFile.basis)
plt.xlabel("sample number")
plt.ylabel("basis vector signal (arb)")
plt.legend([f"{i}" for i in range(dsoff.offFile.basis.shape[0])])
plt.title(dsoff.shortName)

event_g = None
def onpick(event):
    global event_g
    event_g = event
    print(event_g.artist.get_label())

def plot_pulses(N, o):
    cmap = plt.matplotlib.colormaps.get_cmap("rainbow")
    plt.figure()
    for i in np.arange(N)+o:
        color = cmap((i-o)/N)
        plt.plot(ds.read_trace(i), "--", color=color, label=f"{i}", picker=True, pickradius=5)
        plt.plot(dsoff.offFile.modeledPulse(i), color=color)
    plt.legend()
    plt.title(dsoff.shortName)
    plt.xlabel("sample number")
    plt.ylabel("signal (arb)")
    plt.gcf().canvas.mpl_connect('pick_event', onpick)

for i in np.arange(10)+10:
    N = 40
    o = i*N
    plot_pulses(N, o)

# manual binning
inds_A = np.array(np.arange(30))
inds_B = np.array([31, 80, 79, 31, 791, 792, 769, 758, 744, 745, 718, 719, 506, 505, 429, 431])
inds_C = np.array([52, 222, 750])
inds_D = np.array([675, 371, 372, 528, 529, 435])

def plot_inds(inds, label, max_pulses_to_plot=40, plot_modeled=True):
    cmap = plt.matplotlib.colormaps.get_cmap("rainbow")
    plt.figure()
    for j, i in enumerate(inds):
        if j >= max_pulses_to_plot:
            break
        color = cmap(j/min(len(inds), max_pulses_to_plot))
        plt.plot(ds.read_trace(i), "--", color=color, label=f"{i}", picker=True, pickradius=5)
        if plot_modeled:
            plt.plot(dsoff.offFile.modeledPulse(i), color=color)       
    plt.legend()
    plt.title(f"{label} {dsoff.shortName}")
    plt.xlabel("sample number")
    plt.ylabel("signal (arb)")
    plt.gcf().canvas.mpl_connect('pick_event', onpick)

plot_inds(inds_A, label="A foil pulses with energy")
plot_inds(inds_B, label="B foil pulses with pile-up, no energy", plot_modeled=False)
plot_inds(inds_C, label="C foil + non-foil, no energy")
plot_inds(inds_D, label="D non-foil", plot_modeled=False)

def calc_hist_mean(counts):
    return np.sum(np.arange(len(counts))*counts)/np.sum(counts)

def calc_hist_std(counts):
    m = calc_hist_mean(counts)
    s = (counts-m)**2
    return np.sqrt(np.sum(np.arange(len(counts))*s)/(len(counts)-1))

pulse_bin = np.zeros(len(dsoff), dtype=np.int64)
residualStdDev = np.zeros(len(dsoff))
direct_coef = np.zeros(len(dsoff))
pulse_coef = np.zeros(len(dsoff))
hist_std = np.zeros(len(dsoff))
peak_to_avg = np.zeros(len(dsoff))
for i in range(len(dsoff)):
    trace = ds.read_trace(i)
    trace_pt_sub = trace-np.mean(trace[:10])
    residualStdDev[i] = dsoff.residualStdDev[i]
    direct_coef[i]  = dsoff.offFile._mmap_with_coefs["coefs"][i,3] + dsoff.offFile._mmap_with_coefs["coefs"][i,4]
    pulse_coef[i]  = dsoff.offFile._mmap_with_coefs["coefs"][i,2]
    hist_std[i]  = calc_hist_std(trace)
    peak_to_avg[i] = np.amax(trace_pt_sub)/np.mean(trace_pt_sub)

    

    if np.abs(peak_to_avg[i]) > 15:
        # probably D
        pulse_bin[i] = 3
    elif np.abs(peak_to_avg[i]) > 3.5:
        # C
        pulse_bin[i] = 2
    elif residualStdDev[i] > 15:
        # B
        pulse_bin[i] = 1
    else:
        pulse_bin[i] = 0

plot_inds(np.nonzero(pulse_bin==0)[0], label="auto A foil pulses with energy")
plot_inds(np.nonzero(pulse_bin==1)[0], label="auto B foil pulses with pile-up, no energy", plot_modeled=False)
plot_inds(np.nonzero(pulse_bin==2)[0], label="auto C foil + non-foil, no energy")
plot_inds(np.nonzero(pulse_bin==3)[0], label="auto D non-foil", plot_modeled=False)

# inds_mystery = np.array([791, 744, 718,429] )

# plot_inds(inds_mystery, label="mystery", plot_modeled=False)

a, b = dsoff.rowcount[0], dsoff.rowcount[1]
aa, bb = dsoff.unixnano[0], dsoff.unixnano[1]
rowPeriodSeconds_better = 1e-9*(bb-aa)/(b-a)

energy_category = dsoff.energy[:]
energy_category[pulse_bin==1]=-1000
energy_category[pulse_bin==2]=-2000
energy_category[pulse_bin==3]=-3000
save_d = {"energy_category": energy_category,
          "pulse_bin": pulse_bin, 
          "framecount": dsoff.rowcount//ds.number_of_rows,
          "frame_period_s": rowPeriodSeconds_better*ds.number_of_rows, 
          "n_samples": ds.nSamples,
          "n_presamples": ds.nPresamples}
import pickle
with open(f"{dsoff.shortName}.pkl","wb") as f:
    pickle.dump(save_d,f)

def spectroscopic_extending_live_time(timestamps_s, min_before_s, min_after_s):
    dead_time_after_s = min_after_s
    spectroscopic_inds = []
    live_time_s = 0
    for i in np.arange(len(timestamps_s)-2):
        a, b, c = timestamps_s[i], timestamps_s[i+1], timestamps_s[i+2]
        if b-a > min_before_s and c-b > min_after_s:
            spectroscopic_inds.append(i)
            live_time_s += b-a-dead_time_after_s
    return live_time_s, np.array(spectroscopic_inds)






timestamps_s = save_d["framecount"]*save_d["frame_period_s"]
min_before_s = save_d["n_presamples"]*save_d["frame_period_s"]
min_after_s = (save_d["n_samples"]-save_d["n_presamples"])*save_d["frame_period_s"]
live_time_s, spectroscopic_inds = spectroscopic_extending_live_time(timestamps_s,
                                  min_before_s=min_before_s, 
                                  min_after_s=min_after_s)
total_time = timestamps_s[-1]-timestamps_s[0]

N_foil_pileup = np.sum(save_d["pulse_bin"]==1)
N_foil_and_non_foil = np.sum(save_d["pulse_bin"]==2)
N_foil_with_energy = np.sum(save_d["pulse_bin"]==0)
N_non_foil = np.sum(save_d["pulse_bin"]==3)
N_total = len(save_d["pulse_bin"])
foil_unresolved_fraction = (N_foil_pileup+N_foil_and_non_foil)/(N_total-N_non_foil)

live_energies = save_d["energy_category"][spectroscopic_inds]
def midpoints(x):
    return (x[1:]+x[:-1])/2

plt.figure()
bin_edges = np.arange(-4e3,6e6,1e3)
counts, _ = np.histogram(live_energies, bin_edges)
plt.plot(midpoints(bin_edges), counts, drawstyle="steps-mid")
plt.xlabel("energy / eV (- is category)")

E_roi_lo = 5.55e6
E_roi_hi = 5.66e6
N_peak = np.sum(np.logical_and(live_energies>E_roi_lo, live_energies<E_roi_hi))
print(f"{total_time=} s")
print(f"f{live_time_s=} s")
print(f"{N_total=}")
print(f"{N_foil_with_energy=}")
print(f"{N_foil_pileup=}")
print(f"{N_foil_and_non_foil=}")
print(f"{N_non_foil=}")
print(f"{foil_unresolved_fraction=:.3f}")
print(f"{E_roi_lo=} eV, {E_roi_hi=} eV")
print(f"{N_peak=}")
print(f"{N_peak/live_time_s/(1-foil_unresolved_fraction)=:.3f} Bq +/- {100/np.sqrt(N_peak):.2}% roughly")


def live_ranges_framecount(trig_framecounts, dead_after_frames):
    live_ranges = []
    spectroscopic_inds = []
    live_start = 0+dead_after_frames # start dead as if we just saw a trigger
    a,b = 0, 0 # assume we just saw a trigger
    for i in range(len(trig_framecounts)):
        a = b
        b = trig_framecounts[i]
        if live_start>b:
            # if the next pulse happens before live start, we reject that pulse and
            # extend the dead time by not starting the live time
            live_start = b+dead_after_frames
        else:
            # otherwise the pulse is accepted, and the next live time starts after it
            live_end=b
            spectroscopic_inds.append(b)
            live_ranges.append((live_start, live_end))
            live_start = b+dead_after_frames
    return live_ranges, np.array(spectroscopic_inds)

live_ranges, spectroscopic_inds = live_ranges_framecount(save_d["framecount"], 10000)


ljh = ljhfiles.LJHFile(file_pattern("0000"))

def live_time_plot(j_start, n_samples, title=""):
    raw_data = ljh.get_long_record_at(j=j_start, n_samples=n_samples, npre=0)
    baseline=np.median(raw_data)
    trigs_in_range = save_d["framecount"][np.logical_and(j_start<=save_d["framecount"], save_d["framecount"]<(j_start+n_samples))]
    spect_inds_in_range = spectroscopic_inds[np.logical_and(j_start<=spectroscopic_inds, spectroscopic_inds<(j_start+n_samples))]
    plt.figure(figsize=(16,6))
    plt.plot(np.arange(len(raw_data))+j_start,-(raw_data-baseline))
    for i, (live_start, live_end) in enumerate(live_ranges):
        if live_end < j_start:
            continue
        if live_start > len(raw_data)+j_start:
            break
        line_live_time, = plt.plot(np.arange(live_start, live_end), (np.zeros((live_end-live_start))), "r-", 
                drawstyle="steps", lw=10, solid_capstyle="butt", label="live time", alpha=0.5)
    line_deadtime_pulse, = plt.plot(trigs_in_range, -(raw_data[trigs_in_range-j_start]-baseline),"kx", label="pulse during deadtime extends deadtime")
    line_livetime_pulse, = plt.plot(spect_inds_in_range, -(raw_data[spect_inds_in_range-j_start]-baseline),"ko", label="pulse during live time counted")
    plt.ylabel("pulse signal / arb")
    plt.xlabel("sample number")
    plt.gca().legend(handles=[line_livetime_pulse, line_deadtime_pulse, line_live_time])
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(j_start, j_start+n_samples)




live_time_plot(n_samples = 500000, j_start = 0, title="start of data")
live_time_plot(n_samples = 100000, j_start = ds.rowcount[inds_A[-1]]//ds.number_of_rows-50000, title="A example")
live_time_plot(n_samples = 100000, j_start = ds.rowcount[inds_B[-1]]//ds.number_of_rows-50000, title="B example")
live_time_plot(n_samples = 100000, j_start = ds.rowcount[inds_C[-1]]//ds.number_of_rows-50000, title="C example")
live_time_plot(n_samples = 100000, j_start = ds.rowcount[inds_D[1]]//ds.number_of_rows-50000, title="D example big")
live_time_plot(n_samples = 100000, j_start = ds.rowcount[inds_D[0]]//ds.number_of_rows-50000, title="D example small")
