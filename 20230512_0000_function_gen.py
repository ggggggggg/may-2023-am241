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
import retrigger
plt.close("all")
plt.ion()

### get the filesnames we're going to use
try:
    d0 = os.path.dirname(os.path.realpath(__file__))
except:
    d0 = os.getcwd()

def file_pattern(runnum):
    # p = os.path.join("/home","pcuser","data",f"20230106",f"{runnum}",f"20230106_run{runnum}_chan*.ljh")
    p = os.path.join(f"20230512",f"{runnum}",f"20230512_run{runnum}_chan*.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

pulse_files = file_pattern("0000")
noise_files = file_pattern("0000")

data = mass.TESGroup(filenames=pulse_files, noise_filenames=noise_files,
max_chans=12, overwrite_hdf5_file=True)
data.set_all_chan_good()
# data.summarize_data(use_cython=False)
ds = data.channel[3]



filter_from_data = retrigger.filter_from_trigger(ds, 62, 50)
plt.figure()
# plt.plot(retrigger.filter_simple(10),label="simple10")
# plt.plot(filter_simple(50),label="simple50")
plt.plot(filter_from_data, label="from data 50")
plt.legend()

filter = np.array([-1,1])
ct = retrigger.ChunkedTrigger(trig_vec=filter, threshold=30)
ct.trigger_ds(ds)
tg = retrigger.TriggerGetter(ds, 50, 50)
plt.figure()
for i in list(range(5))+[len(ct.trig_inds)-1]:
    plt.plot(tg.get_ljh_record_at(ct.trig_inds[i])["data"].T)
plt.title(ds.filename+" triggers with b-a>30")
plt.xlabel("sample number")
plt.ylabel("dac units")