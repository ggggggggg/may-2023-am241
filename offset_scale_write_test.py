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
    p = os.path.join("/home","pcuser","data",f"20230106",f"{runnum}",f"20230106_run{runnum}_chan*.ljh")
    # p = os.path.join(f"20230515",f"{runnum}",f"20230515_run{runnum}_chan*.ljh")
    return p

def files(runnum):
    return mass.filename_glob_expand(file_pattern(runnum))

pulse_files = files("0004")

ljhs = [ljhfiles.LJHFile(fname) for fname in pulse_files]

OFFSET = 20000

ljh = ljhs[0]

dest_path = ljh.path_with_incremented_runnum(5000)
ljh.copy_ljh_with_offset_and_scaling(dest_path= dest_path,
offset=20000, scaling=0.5, imax=10, 
overwrite=True)

newljh = ljhfiles.LJHFile(dest_path)
plt.figure()
plt.plot(ljh._mmap[0]["data"],label="orig")
plt.plot(newljh._mmap[0]["data"],label="new")
plt.legend()

