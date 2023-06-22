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
max_chans=12, overwrite_hdf5_file=False)
data.set_all_chan_good()
# data.summarize_data(use_cython=False)
ds = data.channel[2]
ds.number_of_rows=ds.pulse_records.datafile.number_of_rows # i think it failes to reload when the hdf5 file exists?
ds.row_timebase=ds.pulse_records.datafile.timebase



# index chosen by eye based on first pulse see in first trigger debug plot
filter_from_data = retrigger.filter_from_trigger(ds, 242227, 50)
plt.figure()
# plt.plot(retrigger.filter_simple(10),label="simple10")
# plt.plot(filter_simple(50),label="simple50")
plt.plot(filter_from_data, label="from data 50")
plt.legend()

trig_filter = np.array([-1,0,0,0,0,0,1])
trig_filter=filter_from_data
# ct = retrigger.ChunkedTrigger(trig_vec=trig_filter, threshold=1000)
# ct.trigger_ds_debug(ds, nseg=1)
# tg = retrigger.TriggerGetter(ds, 500, 2000)
# plt.figure()
# for i in range(max(5,len(ct.trig_inds))):
#     plt.plot(tg.get_ljh_record_at(ct.trig_inds[i])["data"].T)
# plt.title(ds.filename+" triggers with b-a>30")
# plt.xlabel("sample number")
# plt.ylabel("dac units")

# raise Exception()
# debug stuff above, actual retrigger below
ct = retrigger.ChunkedTrigger(trig_vec=trig_filter, threshold=1000)
ct.trigger_ds(ds, nseg=100000000)


plt.figure()
ds.read_segment(0)
d=ds.data.flatten()
plt.plot(d[:1000000])
inds = np.array(ct.trig_inds)
inds = inds[inds<1000000]
plt.plot(inds, d[inds],"o")

        
triggergetter = retrigger.TriggerGetter(ds, 500, 500)

plt.figure()
for i in range(1, 40):
    plt.plot(triggergetter.get_record_at(ct.trig_inds[i])[0], 
    label=f"up {i} {ct.trig_inds[i]} fromlast:{ct.trig_inds[i]-ct.trig_inds[i-1]}")
plt.legend(loc="right")
plt.grid()



# write pulse triggers
new_header_dict = ds.pulse_records.datafile.__dict__.copy()
new_header_dict['asctime'] = time.asctime(time.gmtime())
new_header_dict['version_str'] = '2.2.0'
new_header_dict['nSamples'] = triggergetter.n_post+triggergetter.n_pre
new_header_dict["nPresamples"] = triggergetter.n_pre
dest_path = retrigger.ljh_run_rename_dastard(ds.filename, 1000)
retrigger.ljh_write_traces(ct.trig_inds, triggergetter, new_header_dict,
                 dest_path, overwrite=True)

def get_noise_trigger_inds(pulse_trigger_inds, n_dead_samples, n_record_samples):
    diffs = np.diff(pulse_trigger_inds)
    inds = []
    for i in range(len(diffs)):
        if diffs[i] > n_dead_samples:
            n_make = (diffs[i]-n_dead_samples)//n_record_samples-1
            ind0 = pulse_trigger_inds[i]+n_dead_samples
            for j in range(n_make):
                inds.append(ind0+n_record_samples*j)
    return inds

# write noise triggers
noise_trigger_inds = get_noise_trigger_inds(ct.trig_inds,
    n_dead_samples=int(1e5), 
    n_record_samples=triggergetter.n_pre+triggergetter.n_post)
noise_dest_path = dest_path = retrigger.ljh_run_rename_dastard(ds.filename, 2000)
retrigger.ljh_write_traces(noise_trigger_inds[:min(5000, len(noise_trigger_inds))], triggergetter, new_header_dict,
                 dest_path, overwrite=True)
plt.figure()
ds.read_segment(0)
d=ds.data.flatten()
plt.plot(d[:1000000])
inds = np.array(noise_trigger_inds)
inds = inds[inds<1000000]
plt.plot(inds, d[inds],"o")
plt.title("noise trigger inds")
