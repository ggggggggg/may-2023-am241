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
import scipy
import scipy.signal
import time
import ljhfiles
plt.close("all")
plt.ion()

## +++++ BEGIN INPUTS +++++ ##

my_dir = "/home/pcuser/data"
my_folder = "20231003"
my_runnum = "0002"
my_chan = "3" # to do : implement make this "*" for processing all channels

my_polarity = -1 # make this -1 for negative pulses

def my_irange(npulses): 
 #   return int(npulses)        # npulses - used in filter and in calculated real time of analysis
    return int(npulses)        # npulses - used in filter and in calculated real time of analysis
def my_imin(npulses): 
    return 0                # 0 - used in filter and in calculated real time of analysis

my_threshold = 50
my_n_dead_sam = 50000
my_long_noise_record_n_samples = 50000
my_tau = 10                 # holdoff, non-extending dead-time, in samples
## +++++ _END_ INPUTS +++++ ##


### get the filesnames we're going to use
try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()

def file_pattern(runnum):
 #  p = os.path.join("/home","pcuser","data",f"20230825",f"{runnum}",f"20230825_run{runnum}_chan*.ljh")
    p = os.path.join(f"{my_dir}",f"{my_folder}",f"{my_runnum}",f"{my_folder}_run{my_runnum}_chan{my_chan}.ljh")
#   p = os.path.join(f"20230825",f"{runnum}",f"20230825_run{runnum}_chan*.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

# pulse_files = files("0001") #runnum 0000 is short run. 0001 is long one
pulse_files = files(my_runnum) #runnum 0000 is short run. 0001 is long one
ljhs = [ljhfiles.LJHFile(fname) for fname in pulse_files]


# To do: put runnum and chan in file name. Then only need to change "p" for each run
ljh = ljhs[0] #RPF [0] is ch 1 ; if using "*" need to change code here to loop over all channels. Someday :)
ljh.set_output_npre_npost(400,400) ## To do, decouple output file rec len from input file rec len

#ljh.plot_median_value_vs_time()
ljh.plot_median_value_vs_time(median_filter_len=100, chunk_skip_size=100)


filter = np.array([-1]*10+[+1]*10) * my_polarity # negate for negative pulses

my_imax = my_imin(ljh.nPulses)+my_irange(ljh.nPulses)

pulse_inds_filter = ljh.fasttrig_filter(imax=my_imax, filter=filter, 
                                        threshold=my_threshold,
                                        imin=my_imin(ljh.nPulses),tau=my_tau) # imax=ljh.nPulses//10
noise_inds = ljhfiles.get_noise_trigger_inds(pulse_inds_filter, n_dead_samples=my_n_dead_sam, 
                                             n_record_samples=ljh.output_npre+ljh.output_npost,
                                             max_inds=5000)
long_noise_inds = ljhfiles.get_noise_trigger_inds(pulse_inds_filter, n_dead_samples=my_n_dead_sam, 
                                             n_record_samples=my_long_noise_record_n_samples,
                                             max_inds=100)

record = ljh.get_record_at(pulse_inds_filter[1])


v = record["data"]
plt.figure()
plt.plot(np.arange(len(v))-len(v)//2,v, label="record")
plt.plot(np.arange(len(filter))-len(filter)//2, np.std(v)*filter+np.mean(v), label="filter scaled")
plt.xlabel("sample number in record")
plt.ylabel("ljh data (arb)")
plt.title(f"one chosen record\n{ljh.filename}",fontsize=10)



ljh.plot_first_n_samples_with_inds(4000000, pulse_inds=pulse_inds_filter, 
                                   noise_inds=noise_inds, filter=filter, 
                                   imin=my_imin(ljh.nPulses)*ljh.nSamples)
plt.title(ljh.filename)

if my_polarity == -1:
    # takes negative going pulses to positive going pulses while maining approx same baseline
    offset = np.uint16(2**15)
    scaling = -1
elif my_polarity == 1:
    offset = 0
    scaling = 1
ljh.write_traces_to_new_ljh_with_offset_and_scaling(pulse_inds_filter, ljh.path_with_incremented_runnum(1000), 
                                                    offset=offset, scaling=scaling, overwrite=True)
ljh.write_traces_to_new_ljh_with_offset_and_scaling(noise_inds, ljh.path_with_incremented_runnum(2000), 
                                                    offset=offset, scaling=scaling, overwrite=True)
# ljh.set_output_npre_npost(0,my_long_noise_record_n_samples)
# ljh.write_traces_to_new_ljh_with_offset_and_scaling(long_noise_inds, ljh.path_with_incremented_runnum(3000), 
#                                                     offset=offset, scaling=scaling, overwrite=True)

spectrum = mass.mathstat.power_spectrum.PowerSpectrum(my_long_noise_record_n_samples // 2, dt=ljh.timebase)
# long_noise_ljh = ljhfiles.LJHFile(ljh.path_with_incremented_runnum(3000))
window = np.ones(my_long_noise_record_n_samples)
n_long_records = min(len(long_noise_inds), 20)
for i in range(n_long_records):
    long_record = ljh.get_long_record_at(long_noise_inds[i], my_long_noise_record_n_samples, 0)
    if long_record is None:
        break # wasn't enough data to get that long record
    spectrum.addDataSegment(long_record, window=window)
f = spectrum.frequencies()
psd = spectrum.spectrum()
plt.figure()
plt.plot(f[1:], psd[1:])
plt.xlabel("f")
plt.ylabel("psd")
plt.yscale("log")
plt.title(f"{n_long_records=} {my_long_noise_record_n_samples=}")
plt.grid(True)

plt.figure()
for i in range(n_long_records):
    long_record = ljh.get_long_record_at(long_noise_inds[i], my_long_noise_record_n_samples, 0)
    if long_record is None:
        break # wasn't enough data to get that long record
    plt.plot(long_record)
    # plt.plot(ljh.get_record_at(long_noise_inds[i])["data"])

# long_noise_records = mass.NoiseRecords(ljh.path_with_incremented_runnum(3000))
# long_noise_records.nSamples = my_long_noise_record_n_samples
# long_noise_records.timebase = ljh.timebase
# long_noise_records.hdf5_group = None
# spectrum = long_noise_records.compute_power_spectrum_reshape(
#                 max_excursion=100, seg_length=my_long_noise_record_n_samples)
# # long_noise_records.noise_psd.attrs['delta_f'] = spectrum.frequencies()[1]-spectrum.frequencies()[0]            
# long_noise_records.plot_power_spectrum(sqrt_psd=False)

#summary of triggers and time
n_trig = len(pulse_inds_filter)
print(f"my_irange = {my_irange(ljh.nPulses)}, my_imax = {my_imax}")
dt_s = my_irange(ljh.nPulses)*ljh.nSamples*ljh.timebase # should be my_range
rate_trig = n_trig/dt_s
rate_trig = n_trig/dt_s
print(f"number_of_triggers_= {n_trig}")
print(f"dt_(s)_= {dt_s:0.5f}")
print(f"trigger_rate_(/s)_= {rate_trig:0.4f} +/- {np.sqrt(n_trig)/dt_s:0.4f}")
print(f"u_(%)_= {100/np.sqrt(n_trig):0.2f}")
print(f"timebase_= {ljh.timebase:0.6E}")
t02 = ljh.timestamp_offset # this is the right start time; to do, get end of last record

#fo = open(ljh.filename,"r")
#fo.close()

#make sure the output is what we intend
# pulse_files_output = files("1001")
# ljh_output = ljhfiles.LJHFile(pulse_files_output[1])
# assert ljh_output._mmap[0]["posix_usec"] == ljh.get_record_at(pulse_inds_filter[0])["posix_usec"]
# assert ljh_output._mmap[0]["rowcount"] == ljh.get_record_at(pulse_inds_filter[0])["rowcount"]
# assert all(ljh_output._mmap[0]["data"][:] == ljh.get_record_at(pulse_inds_filter[0])["data"][:])


# =====CHECK FOR DROPPED RECORDS =============================================
# plt.figure()
# plt.plot(ljh._mmap["posix_usec"][1:]-ljh._mmap["posix_usec"][0], np.diff(ljh._mmap["posix_usec"])*1e-6,".")
# plt.xlabel("time since start of file (s)")
# plt.ylabel("time between ljh records (s)")
# plt.tight_layout()
# 
# plt.figure()
# plt.plot(ljh._mmap["posix_usec"][1:]-ljh._mmap["posix_usec"][0], np.diff(ljh._mmap["rowcount"]),".")
# plt.xlabel("time since start of file (s)")
# plt.ylabel("rowcount diff between ljh records (arb)")
# plt.tight_layout()
# =============================================================================


