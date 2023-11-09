import numpy as np
import collections
import pylab as plt
import os
import scipy
import scipy.ndimage
import scipy.signal

from numba import njit
import numba.typed

class LJHFile():
    TOO_LONG_HEADER=100
    def __init__(self, filename):
        self.filename = filename
        self.__read_header(self.filename)
        self.dtype = np.dtype([('rowcount', np.int64),
                                           ('posix_usec', np.int64),
                                           ('data', np.uint16, self.nSamples)])
        self._mmap = np.memmap(self.filename, self.dtype, mode="r",
                               offset=self.header_size, shape=(self.nPulses,))
        self._cache_i = -1
        self._cache_data = None

    def set_output_npre_npost(self, npre, npost):
        self.output_dtype = np.dtype([('rowcount', np.int64),
                                           ('posix_usec', np.int64),
                                           ('data', np.uint16, npre+npost)])
        self.output_npre = npre
        self.output_npost = npost
        self.output_record = np.zeros(1,dtype=self.output_dtype)[0]
    
    def plot_median_value_vs_time(self, median_filter_len=20, chunk_skip_size=10):
        t_sec = self._mmap["posix_usec"][::chunk_skip_size]*1e-6
        median = scipy.ndimage.median_filter(self._mmap["data"][::chunk_skip_size,0].flatten(),
                                             median_filter_len)
        
        plt.figure()
        plt.plot(t_sec-t_sec[0], median, ".")
        plt.xlabel("time (seconds)")
        plt.ylabel("median value (arb)")
        plt.title(self.filename)

    def __read_header(self, filename):
        """Read in the text header of an LJH file.

        On success, several attributes will be set: self.timebase, .nSamples,
        and .nPresamples

        Args:
            filename: path to the file to be opened.
        """
        # parse header into a dictionary
        header_dict = collections.OrderedDict()
        with open(filename, "rb") as fp:
            i = 0
            while True:
                i += 1
                line = fp.readline()
                if line.startswith(b"#End of Header"):
                    break
                elif line == b"":
                    raise Exception("reached EOF before #End of Header")
                elif i > self.TOO_LONG_HEADER:
                    raise IOError("header is too long--seems not to contain '#End of Header'\n"
                                    + "in file %s" % filename)
                elif b":" in line:
                    a, b = line.split(b":", 1)  # maxsplits=1, py27 doesnt support keyword
                    a = a.strip()
                    b = b.strip()
                    a = a.decode()
                    b = b.decode()
                    if a in header_dict and a != "Dummy":
                        print("repeated header entry {}".format(a))
                    header_dict[a] = b
                else:
                    continue  # ignore lines without ":"
            self.header_size = fp.tell()
            fp.seek(0)
            self.header_str = fp.read(self.header_size)

        # extract required values from header_dict
        # use header_dict.get for default values
        self.timebase = float(header_dict["Timebase"])
        self.nSamples = int(header_dict["Total Samples"])
        self.nPresamples = int(header_dict["Presamples"])
        # column number and row number have entries like "Column number (from 0-0 inclusive)"
        row_number_k = [k for k in header_dict.keys() if k.startswith("Row number")]
        if len(row_number_k) > 0:
            self.row_number = int(header_dict[row_number_k[0]])
        col_number_k = [k for k in header_dict.keys() if k.startswith("Column number")]
        if len(col_number_k) > 0:
            self.row_number = int(header_dict[col_number_k[0]])
        self.client = header_dict.get("Software Version", "UNKNOWN")
        self.number_of_columns = int(header_dict.get("Number of columns", -1))
        self.number_of_rows = int(header_dict.get("Number of rows", -1))
        self.timestamp_offset = float(header_dict.get("Timestamp offset (s)", "-1"))
        self.version_str = header_dict['Save File Format Version']
        # if Version(self.version_str.decode()) >= Version("2.2.0"):
        self.pulse_size_bytes = (16 + 2 * self.nSamples) # dont bother with old ljh
        # else:
        #     self.pulse_size_bytes = (6 + 2 * self.nSamples)
        self.binary_size = os.stat(filename).st_size - self.header_size
        self.header_dict = header_dict
        self.nPulses = self.binary_size // self.pulse_size_bytes
        # Fix long-standing bug in LJH files made by MATTER or XCALDAQ_client:
        # It adds 3 to the "true value" of nPresamples. For now, assume that only
        # DASTARD clients have this figure correct.
        if "DASTARD" not in self.client:
            self.nPresamples += 3

        # Record the sample times in microseconds
        self.sample_usec = (np.arange(self.nSamples)-self.nPresamples) * self.timebase * 1e6

    # def filter_chunk(self, i, trig_vec):
    #     # Z = nSamples (loaded per i)
    #     # N = length of trig vec
    #     # M = length of 2nd input to np.convolve
    #     # np.convolve with "valid" outputs a vector of length M-N+1
    #     # we want to get Z potential triggers per chunk, so if we look for simply being above
    #     # threshold we need M-N+1=Z, so M=Z+N-1
    #     # but we may want to do an edge trigger with diff, which makes M=Z+N
    #     data = self._mmap["data"][i:i+2].flatten()
    #     output = np.convolve(trig_vec, data[:self.nSamples+len(trig_vec)],"valid")
    #     assert len(output) == self.nSamples+1
    #     return output
    
    
    # def edge_trigger_chunk(self, i, trig_vec, threshold):
    #     filter_output = self.filter_chunk(i, trig_vec)
    #     over_threshold = filter_output>threshold
    #     edge_triggers = np.diff(np.array(over_threshold,dtype=int))==1
    #     return np.nonzero(edge_triggers)[0]+i*self.nSamples # offset inds based on i
    
    # def edge_trigger_many_chunks_debug(self, trig_vec, threshold, i0=0, imax=None, verbose=True):
    #     inds = self.edge_trigger_many_chunks(trig_vec, threshold, i0, imax, verbose)
    #     inds_rem_offset = inds - i0*self.nSamples
    #     alldata = self._mmap["data"][i0:imax+2].flatten()
    #     alldata_filtered = np.convolve(trig_vec, alldata, "valid")

    #     xinds = i0*self.nSamples + np.arange(0, (imax+2-i0)*self.nSamples)
    #     plt.figure()
    #     plt.plot(xinds, alldata, ".", label="alldata")
    #     plt.plot(xinds[:len(alldata_filtered)], alldata_filtered, label="filtered")
    #     plt.plot(inds_rem_offset, alldata[inds_rem_offset],"o", label="trig inds")
    #     plt.plot(inds_rem_offset, alldata_filtered[inds_rem_offset],"o", label="trig inds filtered")
    #     plt.axhline(threshold, label="threshold")
    #     plt.xlabel("framecount")
    #     plt.ylabel("value")
    #     plt.legend()

    #     return inds

    # def edge_trigger_many_chunks(self, trig_vec, threshold, i0=0, imax=None, verbose=True):
    #     inds = []
    #     if imax is None:
    #         imax = len(self._mmap)-1
    #     for i in range(i0, imax):
    #         if verbose and i%10==0:
    #             print(f"{i} in i={i0}, imax={imax}")
    #         inds += list(self.edge_trigger_chunk(i, trig_vec, threshold))
    #     return np.array(inds)
    
    def get_record_at(self, j):
        assert (self.output_npre+self.output_npost) < self.nSamples-1
        i = (j-self.output_npre)//self.nSamples
        if self._cache_i != i:
            self._cache_data = self._mmap["data"][i:i+2].flatten()
            self._cache_posix_usec = self._mmap["posix_usec"][i]
        j_start = (j-self.output_npre) - i*self.nSamples
        self.output_record["data"] = self._cache_data[j_start:j_start+self.output_npre+self.output_npost]
        self.output_record["rowcount"] = j*self.number_of_rows
        self.output_record["posix_usec"] = self._cache_posix_usec+j_start*self.timebase*1e6
        return self.output_record
    
    def get_long_record_at(self, j, n_samples, npre):
        i_start = (j-npre)//self.nSamples
        j_start = (j-npre) - i_start*self.nSamples
        data = np.zeros(n_samples)
        data[:self.nSamples-j_start] = self._mmap["data"][i_start][j_start:]
        i_max = (n_samples-j_start-1)//self.nSamples
        if i_max >= len(self._mmap) or i_max <= 0:
            return None
        for i in 1+np.arange(i_max):
            a = (i)*self.nSamples-j_start
            b = a+self.nSamples
            data[a:b] = self._mmap["data"][i_start+i]
        data[b:] = self._mmap["data"][i_start+i+1][:len(data)-b] # fill in end
        return data     
    
    def write_traces_to_new_ljh(self, inds, dest_path, overwrite=False):
        if os.path.exists(dest_path) and not overwrite:
            raise IOError(f"The ljhfile {dest_path} exists and overwrite was not set to True")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, "wb") as dest_fp:
            dest_fp.write(ljh_header_str(self.output_header_dict()).encode() )
            printstep = max(10, len(inds)//100)
            for i, ind in enumerate(inds):
                if i%printstep==0:
                    print(f"{dest_path} {i}/{len(inds)}")
                record = self.get_record_at(ind)
                record.tofile(dest_fp)

    def write_traces_to_new_ljh_with_offset_and_scaling(self, inds, dest_path, scaling, offset, overwrite=False):
        if os.path.exists(dest_path) and not overwrite:
            raise IOError(f"The ljhfile {dest_path} exists and overwrite was not set to True")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, "wb") as dest_fp:
            dest_fp.write(ljh_header_str(self.output_header_dict()).encode() )
            printstep = max(10, len(inds)//100)
            for i, ind in enumerate(inds):
                if i%printstep==0:
                    print(f"{dest_path} {i}/{len(inds)}")
                record = self.get_record_at(ind)
                record["data"] += offset
                record["data"] = np.array(record["data"]*scaling,dtype="uint16")
                record.tofile(dest_fp)

    def copy_ljh_with_offset_and_scaling(self, dest_path, offset, scaling, imax, overwrite=False):
        if os.path.exists(dest_path) and not overwrite:
            raise IOError(f"The ljhfile {dest_path} exists and overwrite was not set to True")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        imax = min(imax, len(self._mmap))

        with open(dest_path, "wb") as dest_fp:
            dest_fp.write(ljh_header_str(self.header_dict).encode() ) 
            printstep = max(10, imax//100)
            for i in range(imax):
                if i%printstep==0:
                    print(f"{dest_path} {i}/{imax}") #was inds
                record = self._mmap[i].copy()
                record["data"] += offset
                record["data"] = np.array(record["data"]*scaling,dtype="uint16")
                record.tofile(dest_fp)

    def output_header_dict(self):
        d = self.header_dict.copy()
        d["Presamples"] = self.output_npre
        d["Total Samples"] = self.output_npost+self.output_npre
        return d     
    
    def output_header_str(self):
        return ljh_header_str(self.output_header_dict())
    
    def __repr__(self):
        return f"LJHFile {self.filename}"
    
    def path_with_incremented_runnum(self, inc):
        return path_with_incremented_runnum(os.path.abspath(self.filename), inc)
    
    def plot_first_n_samples_with_inds(self, n, pulse_inds, noise_inds, filter=None, imin=0):
        n = min(n, len(self._mmap)*self.nSamples)
        jmin = imin//self.nSamples # this one should floor
        jmax = (imin+n)//self.nSamples+1 # this one shoudl ceil, approx by +1
        qmin = imin%self.nSamples
        qmax = qmin+n
        imax = imin+n
        data = self._mmap["data"][jmin:jmax].flatten()[qmin:qmax]
        plt.figure()
        plt.plot(np.arange(imin, imax), data)
        pulse_inds = np.array(pulse_inds,dtype="int64")
        noise_inds = np.array(noise_inds,dtype="int64")
        pi = pulse_inds[np.logical_and(imin<pulse_inds, pulse_inds<imax)]
        ni = noise_inds[np.logical_and(imin<noise_inds, noise_inds<imax)]
        plt.plot(pi, data[pi-imin], "o", label="pulse_inds")
        plt.plot(ni, data[ni-imin], "o", label="pulse_inds")
        if filter is not None:
            fv = np.convolve(filter[::-1], data, "valid")
            plt.plot(np.arange(imin, imin+len(fv)), fv, label="filter")
            plt.plot(pi-len(filter)//2, fv[pi-len(filter)//2-imin], "o", label="pulse_inds filter")
        plt.xlabel("frame index")
        plt.ylabel("ljh data value (arb)")
        plt.legend()
        plt.grid(True)
      #  plt.tight_layout()

    # def fasttrig(self, imax, threshold, closest_trig):
    #     imax = min(imax, len(self._mmap))
    #     inds = numba.typed.List([0])[:0] # get an integer typed list?
    #     datas = self._mmap["data"]
    #     data = np.zeros(self.nSamples, dtype="int64")
    #     dnext = datas[0][0]
    #     n_since_pulse=closest_trig
    #     printstep = max(10, imax//100)
    #     for i in range(imax):
    #         data[:] = datas[i]
    #         if i%printstep == 0:
    #             print(f"fasttrig {i=}/{imax=}")
    #         n_since_pulse, dnext = fasttrig_segment(data, threshold, dnext, inds, 
    #                                                 ind_offset=i*self.nSamples, n_since_pulse=n_since_pulse, 
    #                                                 closest_trig=closest_trig)
    #     return inds
    
    def fasttrig_filter(self, imax, filter, threshold, imin=0,tau=0):
        imax = min(imax, len(self._mmap))
        inds = numba.typed.List([0])[:0] # get an integer typed list?
        datas = self._mmap["data"]
        filter = np.array(filter, dtype="float64")
        data = np.zeros(self.nSamples, dtype="int64")
        data[:] = datas[0]
        cache = np.zeros(len(filter), dtype="float64")
        cache[:] = data[:len(filter)]
        filtered_abc = (0,0,0)
        printstep = max(10, imax//100)
        for i in range(imin, imax):
            data[:] = datas[i]
            if i%printstep == 0:
                print(f"fasttrig_filtered {i=}/{imax=}")
            cache, filtered_abc = fasttrig_filter_segment(data, filter, cache, filtered_abc, threshold, 
                                                          inds, i*self.nSamples-len(filter)//2,tau)
        return inds        
    
    def read_trace(self, i):
        return self._mmap[i]["data"]

# @njit
# def fasttrig_segment(data, threshold, d_initial, inds, ind_offset, n_since_pulse, closest_trig):
#     # print(f"{threshold=} {d_initial=}, {inds=}\n{ind_offset=}, {in_pulse=}")
#     # print(f"{data[-10:]=}")
#     dnext = d_initial
#     for j in range(len(data)):
#         d = dnext
#         dnext = data[j]
#         diff = dnext-d
#         if diff > threshold and n_since_pulse >= closest_trig:
#             inds.append(j+ind_offset)  
#             n_since_pulse = 0
#         n_since_pulse+=1

#     # print(f"{diff=} {dnext=} {d=} {j=} {data[j]=}")      
#     return n_since_pulse, dnext

@njit 
def fasttrig_filter_segment(data, filter, cache, filtered_abc, threshold, inds, ind_offset,tau=0): # include deadtime (holdoff)
    a,b,c = filtered_abc
    for j in range(len(data)):
        running_replace_last_in_place(cache, data[j])
        a,b = b,c
        c = np.dot(cache, filter)
        
        if b>threshold and b>=c and b>a:
            if len(inds)<1: # if first trigger, then tlast = 0
                tlast = 0
            else:
                tlast = inds[-1] # else, get previous index
            dt = j+ind_offset-tlast # apply non-extending dead-time (holdoff)
            if dt > tau:
                inds.append(j+ind_offset)
    return cache, filtered_abc

# =============================================================================
# def fasttrig_filter_segment(data, filter, cache, filtered_abc, threshold, inds, ind_offset):
#     a,b,c = filtered_abc
#     for j in range(len(data)):
#         running_replace_last_in_place(cache, data[j])
#         a,b = b,c
#         c = np.dot(cache, filter)
#         if b>threshold and b>=c and b>a:
#             inds.append(j+ind_offset)
#     return cache, filtered_abc
# =============================================================================



@njit
def running_replace_last_in_place(cache, v):
    for i in range(len(cache)-1):
        cache[i] = cache[i+1]
    cache[i+1] = v





import mass

def ljh_header_str(header_dict):
    header_list = []
    for k,v in header_dict.items():
        header_list += [f"{k}:{v}"]
    header_list += ["#End of Header\n"]
    return "\n".join(header_list)

def path_with_incremented_runnum(path, inc=1000):
    # return a new path with the run number incremented by inc
    # assumings format like '20230106\\0000\\20230106_run0000_chan2.ljh'
    basename, channum = mass.ljh_util.ljh_basename_channum(path)
    b, c = os.path.split(basename)
    a, b = os.path.split(b)
    # a = '20230106'
    # b = '0000'
    # c = '20230106_run0000'
    runnum = int(b)
    new_runnum = runnum+inc
    return os.path.join(a, f"{new_runnum:03}", 
        c[:-4]+f"{new_runnum:03}_chan{channum}.ljh")

def get_noise_trigger_inds(pulse_trigger_inds, n_dead_samples, n_record_samples, max_inds):
    diffs = np.diff(pulse_trigger_inds)
    inds = []
    for i in range(len(diffs)):
        if diffs[i] > n_dead_samples:
            n_make = (diffs[i]-n_dead_samples)//n_record_samples-1
            ind0 = pulse_trigger_inds[i]+n_dead_samples
            for j in range(n_make):
                inds.append(ind0+n_record_samples*j)
                if len(inds) == max_inds:
                    return inds
    return inds
