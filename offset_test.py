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

OFFSET = 54000

ljh = ljhs[1]
ljh.set_output_npre_npost(500,500)

plt.figure()
plt.plot(ljh.get_record_at(897987, offset=None)["data"], label="none")
plt.plot(ljh.get_record_at(897987, offset=0)["data"], label="0")
plt.plot(ljh.get_record_at(897987, offset=OFFSET)["data"], label=f"{OFFSET}")
plt.legend()

filter = np.array([-1]*10+[1]*10)
pulse_inds_filter = ljh.fasttrig_filter(imax=ljh.nPulses//10, filter=filter, threshold=2000)
noise_inds = ljhfiles.get_noise_trigger_inds(pulse_inds_filter, n_dead_samples=100000, 
                                             n_record_samples=ljh.output_npre+ljh.output_npost,
                                             max_inds=5000)

record = ljh.get_record_at(pulse_inds_filter[1], offset = OFFSET)
v = record["data"]
plt.figure()
plt.plot(np.arange(len(v))-len(v)//2,v, label="record")
plt.plot(np.arange(len(filter))-len(filter)//2, np.std(v)*filter+np.mean(v), label="filter scaled")
plt.xlabel("sample number in record")
plt.ylabel("ljh data (arb)")
plt.title(f"one chosen record\n{ljh.filename}")

ljh.plot_first_n_samples_with_inds(1000000, pulse_inds=pulse_inds_filter, noise_inds=noise_inds, filter=filter, offset=OFFSET)
plt.title(ljh.filename)


ljh.plot_median_value_vs_time(offset=OFFSET)

ljh.write_traces_to_new_ljh(pulse_inds_filter, ljh.path_with_incremented_runnum(5000), overwrite=True, offset=OFFSET)
ljh.write_traces_to_new_ljh(noise_inds, ljh.path_with_incremented_runnum(6000), overwrite=True, offset=OFFSET)


#make sure the output is what we intend
pulse_files_output = files("5000")
ljh_output = ljhfiles.LJHFile(pulse_files_output[0])
assert ljh_output._mmap[0]["posix_usec"] == ljh.get_record_at(pulse_inds_filter[0], offset=OFFSET)["posix_usec"]
assert ljh_output._mmap[0]["rowcount"] == ljh.get_record_at(pulse_inds_filter[0], offset=OFFSET)["rowcount"]
assert all(ljh_output._mmap[0]["data"][:] == ljh.get_record_at(pulse_inds_filter[0], offset=OFFSET)["data"][:])