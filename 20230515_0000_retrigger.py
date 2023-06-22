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
    # p = os.path.join("/home","pcuser","data",f"20230106",f"{runnum}",f"20230106_run{runnum}_chan*.ljh")
    p = os.path.join(f"20230515",f"{runnum}",f"20230515_run{runnum}_chan*.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

pulse_files = files("0000")

ljhs = [ljhfiles.LJHFile(fname) for fname in pulse_files]



ljh = ljhs[1]
ljh.set_output_npre_npost(500,500)

# trig_vec = -np.array([-1]*50+[1]*50)
# comment out debug line, uncomment other, when doing large fraction of file
# pulse_inds = ljh.edge_trigger_many_chunks_debug(trig_vec, threshold=1000, i0=0, imax=100, verbose=True)
# pulse_inds = ljh.edge_trigger_many_chunks(trig_vec, threshold=1000, i0=0, imax=100, verbose=True)
# pulse_inds = ljh.fasttrig(100000, threshold=50, closest_trig=50)
filter = np.array([-1]*10+[1]*10)
pulse_inds_filter = ljh.fasttrig_filter(imax=ljh.nPulses//10, filter=filter, threshold=2000)
noise_inds = ljhfiles.get_noise_trigger_inds(pulse_inds_filter, n_dead_samples=100000, 
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


ljh.plot_first_n_samples_with_inds(1000000, pulse_inds=pulse_inds_filter, noise_inds=noise_inds, filter=filter)
plt.title(ljh.filename)

ljh.plot_median_value_vs_time()

ljh.write_traces_to_new_ljh(pulse_inds_filter, ljh.path_with_incremented_runnum(1000), overwrite=True)
ljh.write_traces_to_new_ljh(noise_inds, ljh.path_with_incremented_runnum(2000), overwrite=True)


#make sure the output is what we intend
pulse_files_output = files("1000")
ljh_output = ljhfiles.LJHFile(pulse_files_output[0])
assert ljh_output._mmap[0]["posix_usec"] == ljh.get_record_at(pulse_inds_filter[0])["posix_usec"]
assert ljh_output._mmap[0]["rowcount"] == ljh.get_record_at(pulse_inds_filter[0])["rowcount"]
assert all(ljh_output._mmap[0]["data"][:] == ljh.get_record_at(pulse_inds_filter[0])["data"][:])