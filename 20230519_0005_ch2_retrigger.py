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

### get the filesnames we're going to use
try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()

def file_pattern(runnum):
    p = os.path.join("/home","pcuser","data",f"20230519",f"{runnum}",f"20230519_run{runnum}_chan*.ljh")
    # p = os.path.join(f"20230519",f"{runnum}",f"20230519_run{runnum}_chan*.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

pulse_files = files("0005") #runnum 0000 is short run. 0001 is long one

ljhs = [ljhfiles.LJHFile(fname) for fname in pulse_files]


ljh = ljhs[1] #RPF ljhs[1] is ch2 10,000 us fall, 10/s rate, 10 us rise
ljh.set_output_npre_npost(500,500)
# trig_vec = -np.array([-1]*50+[1]*50)
# comment out debug line, uncomment other, when doing large fraction of file
# pulse_inds = ljh.edge_trigger_many_chunks_debug(trig_vec, threshold=1000, i0=0, imax=100, verbose=True)
# pulse_inds = ljh.edge_trigger_many_chunks(trig_vec, threshold=1000, i0=0, imax=100, verbose=True)
# pulse_inds = ljh.fasttrig(100000, threshold=50, closest_trig=50)
filter = np.array([-1]*10+[1]*10)

my_imax = ljh.nPulses # used in filter and in calculated real time of analysis

pulse_inds_filter = ljh.fasttrig_filter(imax=my_imax, filter=filter, threshold=2000) # imax=ljh.nPulses//10
noise_inds = ljhfiles.get_noise_trigger_inds(pulse_inds_filter, n_dead_samples=10000, 
                                           n_record_samples=ljh.output_npre+ljh.output_npost,
                                            max_inds=5000) 

record = ljh.get_record_at(pulse_inds_filter[1])
v = record["data"]
plt.figure()
plt.plot(np.arange(len(v))-len(v)//2,v, label="record")
plt.plot(np.arange(len(filter))-len(filter)//2, np.std(v)*filter+np.mean(v), label="filter scaled")
plt.xlabel("sample number in record")
plt.ylabel("ljh data (arb)")
plt.title(f"one chosen record\n{ljh.filename}")


ljh.plot_first_n_samples_with_inds(100000, pulse_inds=pulse_inds_filter, noise_inds=noise_inds, filter=filter)
plt.title(ljh.filename)

#ljh.plot_median_value_vs_time() # the slowest part of the process

ljh.write_traces_to_new_ljh(pulse_inds_filter, ljh.path_with_incremented_runnum(1000), overwrite=True)
ljh.write_traces_to_new_ljh(noise_inds, ljh.path_with_incremented_runnum(2000), overwrite=True)
print(f"{ljh.path_with_incremented_runnum(1000)}")
#summary of triggers and time
n_trig = len(pulse_inds_filter)
dt_s = my_imax*ljh.nSamples*ljh.timebase
rate_trig = n_trig/dt_s
print(f"number of triggers = {n_trig}")
print(f"dt (s) = {dt_s:0.5f}")
print(f"trigger rate (/s) = {rate_trig:0.4f} +/- {np.sqrt(n_trig)/dt_s:0.4f} ({100/np.sqrt(n_trig):0.2f} %)")
print(f"timebase = {ljh.timebase:0.6E}")
t02 = ljh.timestamp_offset # this is the right start time; to do, get end of last record



#make sure the output is what we intend
pulse_files_output = files("1005")
ljh_output = ljhfiles.LJHFile(pulse_files_output[1])
assert ljh_output._mmap[0]["posix_usec"] == ljh.get_record_at(pulse_inds_filter[0])["posix_usec"]
assert ljh_output._mmap[0]["rowcount"] == ljh.get_record_at(pulse_inds_filter[0])["rowcount"]
assert all(ljh_output._mmap[0]["data"][:] == ljh.get_record_at(pulse_inds_filter[0])["data"][:])

