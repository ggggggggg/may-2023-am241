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
import live_time_algo_and_sim
plt.close("all")
plt.ion()



## +++++ BEGIN INPUTS +++++ ##


my_dir = "/home/pcuser/data"
my_folder = "20231003"
my_runnum = "0002"
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

# choosen dead_after_s
time_since_last = np.diff(ds.p_timestamp[:], prepend=ds.p_timestamp[0])
plt.plot(time_since_last, ds.p_energy[:], ".")
plt.xlabel("time since last pulse (s)")
plt.ylabel("p_energy (eV)")
plt.xlim(0,0.6)
plt.ylim(5637.82e3-400e3, 5637.82e3+100e3)

# 	1. Get timestamps
timestamps_s = ds.p_timestamp[:]
must_be_clear_after_s = 4e-3
dead_after_s = 0.4
# 	2. Calculate live ranges with pulses and relegated inds
# _x = live_time_algo_and_sim.live_ranges_both_directions_simple(timestamps_s, 
#                                 dead_after_arb=dead_after_s, 
#                                 must_be_clear_after_arb=must_be_clear_after_s)
# live_ranges_s, live_triggerd_inds, relegated_inds = _x
# print(f"{len(live_ranges_s)=}\n{len(live_triggerd_inds)=}\n{len(relegated_inds)=}\n{len(relegated_inds)+len(live_triggerd_inds)=}\n{len(ds.p_energy)=}")
# 	3. Of remaining pulses, classify into
class_meaning = {
    0: "anomalous",
    1: "next trigger too close",
    2: "previous trigger too close",
    3: "foil",
    4: "non-foil",
    5: "foil + non-foil",
}
ds.residual_std_dev, ds.fv = live_time_algo_and_sim.jank_residual_std_dev(ds)
ds.time_since_last = np.diff(ds.p_timestamp[:], prepend=ds.p_timestamp[0])
ds.time_to_next = np.diff(ds.p_timestamp[:], append=ds.p_timestamp[-1])
ds.peak_to_avg = ds.p_peak_value[:]/ds.p_pulse_average[:]
ds.classification = np.zeros(len(ds.p_energy), dtype=int)
ds.classification[:]=3
ds.classification[ds.time_since_last<dead_after_s]=2
ds.classification[ds.time_to_next<must_be_clear_after_s]=1
threshold_foil_plus_non_foil = 1.8
threshold_non_foil = 3
ds.classification[np.logical_and(ds.classification==3,ds.peak_to_avg>threshold_non_foil)] = 4
ds.classification[np.logical_and(ds.classification==3,ds.peak_to_avg>threshold_foil_plus_non_foil)] = 5
ds.classification[np.logical_and(ds.classification==3,ds.peak_to_avg<0)]=0
ds.classification[np.logical_and(ds.classification==3,ds.residual_std_dev>15)]=0
energies = ds.p_energy[:]
energies[ds.classification==0]=-1
energies[ds.classification==1]=-2
energies[ds.classification==2]=-3
energies[ds.classification==4]=-4
energies[ds.classification==5]=5600e3
bin_edges = np.arange(0,6e6,1e3)
# 	4. Calc ROI
roi_lo = 5537000
roi_hi = 5665000
inds_roi = np.nonzero((energies>roi_lo) & (energies<roi_hi))[0]
total_counts_roi = len(inds_roi)
live_time_s_roi = np.sum(ds.time_since_last[inds_roi]-dead_after_s)
#   5. Calc Bq
am241_bq_roi_bq = total_counts_roi/live_time_s_roi
am241_bq_roi_bq_sigma = np.sqrt(total_counts_roi)/live_time_s_roi
# 	4. Plot histogram
# live_time_s = live_time_algo_and_sim.live_time_from_live_ranges(live_ranges_s)
def binsize(x):
    return x[1]-x[0]
def midpoints(x):
    return (x[1:]+x[:-1])/2
counts, _ = np.histogram(energies, bin_edges)
total_counts = counts.sum()
live_time_s = np.sum(ds.time_since_last[energies>0]-dead_after_s)
total_activity = total_counts/live_time_s
total_activity_uncertainty = np.sqrt(total_counts)/live_time_s
plt.figure()
plt.plot(midpoints(bin_edges), counts, drawstyle="steps-mid", label="live spectrum")
roi_plot_inds = (midpoints(bin_edges) > roi_lo) & (midpoints(bin_edges)<roi_hi)
plt.plot(midpoints(bin_edges)[roi_plot_inds], counts[roi_plot_inds],"r", drawstyle="steps-mid", label="Am241 ROI")
plt.fill_between(midpoints(bin_edges)[roi_plot_inds], counts[roi_plot_inds], step="mid", color="r", alpha=0.5)
plt.xlabel("energy / eV (with category having specific values)")
plt.ylabel(f"counts per {binsize(bin_edges):0.1f} eV")
plt.title(f"""Histogram Live Time = {live_time_s:0.2f} s. Total Counts = {total_counts}
          Total activity = {total_activity:.3f}+/-{total_activity_uncertainty:.3f} events/s
          ROI activity = {am241_bq_roi_bq:0.3f}+/-{am241_bq_roi_bq_sigma:0.3f}""")
plt.legend()


# plot representatives of pulse classifications
for c in range(6):
    inds = np.nonzero(ds.classification==c)[0]
    c_meaning = class_meaning[c]
    print(f"{c=} {c_meaning} {len(inds)=}")
    live_time_algo_and_sim.plot_inds(ds, inds, label=f"{c=} {c_meaning}", max_pulses_to_plot=50)
    

just_below_inds = np.nonzero((energies<roi_lo) & (np.abs(energies-roi_lo)<100000))[0]
just_above_inds = np.nonzero((energies>roi_hi) & (np.abs(energies-roi_hi)<100000))[0]
live_time_algo_and_sim.plot_inds(ds, just_below_inds, label=f"just_below_inds", max_pulses_to_plot=50)
live_time_algo_and_sim.plot_inds(ds, just_above_inds, label=f"just_above_inds", max_pulses_to_plot=50)
