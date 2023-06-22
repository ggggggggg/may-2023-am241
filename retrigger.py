import numpy as np
import scipy
import scipy.signal
import pylab as plt
import os
import mass


def getoverlappingsegment(ds, segnum):
    # for recording pulses near segment breaks
    (a,b) = ds.read_segment(segnum)
    tmp = ds.data.copy()
    tmp_time = ds.pulse_records.datafile.datatimes_float*1e6
    if a<0:
        raise Exception(f"segnum={segnum} too large for {ds}")
    size1=tmp.size
    (c,d) = ds.read_segment(segnum+1)
    data = np.hstack([tmp.flatten(),ds.data.flatten()])
    posix_usec = np.hstack([tmp_time, ds.pulse_records.datafile.datatimes_float*1e6])
    return data, posix_usec

def getat(ds, ind, nsamp):
    samples_per_segment = ds.pulse_records.pulses_per_seg*ds.nSamples
    segnum = ind // samples_per_segment
    i = ind % samples_per_segment
    (a, b) = ds.read_segment(segnum)
    return ds.data.flatten()[i-nsamp:i+nsamp]   

def plotat(ds, ind, nsamp=1000, label=""):
    x = getat(ds, ind, nsamp)
    plt.plot(x, label=label)

def filter_from_trigger(ds, ind, nsamp):
    raw = getat(ds, ind, nsamp)
    filter = scipy.signal.detrend(raw)
    return filter/np.sqrt(np.dot(filter,filter))

def filter_simple(n):
    filter_raw = np.array([0]*n+list(range(1, n+1)))
    filter = scipy.signal.detrend(filter_raw)
    return filter/np.dot(filter,filter)


class ChunkedTrigger():
    def __init__(self, trig_vec, threshold):
        self.trig_vec = np.array(trig_vec)
        self.trig_n_samples = len(self.trig_vec)
        self.n_last_vals= self.trig_n_samples
        # assumme Z = length of a chunk, N = length of trig vec, and M=Z+self.n_last_vals
        # np.convolve with "valid" outputs a vector of length M-N+1
        # we want to get Z potential triggers per chunk, so if we look for simply being above
        # threshold we need M-N+1=Z, so M=Z+N-1
        # but we may want to do an edge trigger with diff, which makes M=Z+N
        # or we may want to look for local maxima, in which case, depending on our method, M may be larger
        # currently the code does and edge trigger so M=Z+N
        self.threshold=threshold
        self.chunks_triggered=0
        self._last_trigger_ind = self.n_last_vals-100
        self.trig_inds = []
        self._no_more_chunks = False

    def trigger_chunk_conv(self, chunk, debugplot=False):
        if self.chunks_triggered==0:
            self.before_trigger_first_chunk(chunk)
        if len(chunk) != self._len_chunk:
            if not self._no_more_chunks:
                self._no_more_chunks = True
            else:
                raise Exception("only the last chunk can have a different length than the first")
        chunk_extended = np.hstack((self._last_vals, chunk))
        self._last_vals = chunk[-self.n_last_vals:]
        chunk_num = self.chunks_triggered
        ind_offset = chunk_num*self._len_chunk-self.trig_n_samples//2
        trig_magnitude_vec = -np.convolve(self.trig_vec, chunk_extended,"valid")

        over_threshold = trig_magnitude_vec>self.threshold
        edge_triggers = np.diff(np.array(over_threshold,dtype=int))==1
        assert(len(edge_triggers))==len(chunk)
        inds = np.nonzero(edge_triggers)[0]
        self.trig_inds.extend(inds+ind_offset)

        if debugplot:
            plt.figure()
            plt.plot(chunk_extended[:1000000],".", label="data 1M pts")
            plt.plot(trig_magnitude_vec[:1000000],".r", label="trigger conv output")
            plt.axhline(self.threshold, label="threshold")
            plt.plot(over_threshold[:1000000]*60,"k", label="over threshold")
            plt.plot(inds[inds<1000000], chunk_extended[inds[inds<1000000]],"o", label="found triggers")
            plt.legend(loc="right")
            plt.xlabel("sample number")
            plt.title(f"chunk num {self.chunks_triggered}")
        # for i in trig_magnitude_vec:
        #     ind = i + ind_offset
        #     if ind-self._last_trigger_ind>self.deadsamples:
        #         if trig_magnitude_vec[i] >= self.threshold:
        #             self.trig_inds.append(ind)
        #             self._last_trigger_ind = ind
        self.chunks_triggered += 1
        return trig_magnitude_vec, chunk_extended

    def before_trigger_first_chunk(self, chunk):
        self._last_vals = np.array([chunk[0]]*self.n_last_vals)
        self._len_chunk = len(chunk)

    def get_trig_inds(self):
        return self.trig_inds.copy()
    
    def trigger_ds(self, ds, nseg=5, debugplot=True):
        nseg = min(nseg, ds.pulse_records.n_segments)
        for segnum in range(ds.pulse_records.n_segments):
            if segnum == nseg:
                return
            print(f"triggering segment {segnum}/{ds.pulse_records.n_segments-1}, will stop before {nseg}")
            (a, b) = ds.read_segment(segnum)
            magvec, chunk_extended = self.trigger_chunk_conv(ds.data.flatten(), debugplot=segnum==0)

    def trigger_ds_debug(self, ds, nseg=5):
        nseg = min(nseg, ds.pulse_records.n_segments)
        for segnum in range(ds.pulse_records.n_segments):
            if segnum == nseg:
                return
            print(f"triggering segment {segnum}/{ds.pulse_records.n_segments-1}, will stop before {nseg}")
            (a, b) = ds.read_segment(segnum)
            magvec, chunk_extended = self.trigger_chunk_conv(ds.data.flatten(), debugplot=True)



class TriggerGetter():
    def __init__(self, ds, n_pre, n_post):
        self.ds = ds
        self.ljh = self.ds.pulse_records.datafile
        self.n_pre = n_pre
        self.n_post = n_post
        self._last_ind = -1
        self._loaded_segnum = -1
        self._data = None
        self._segment_len = ds.pulse_records.pulses_per_seg*ds.nSamples
        self._ljh_record_dtype = np.dtype([('rowcount', np.int64),
                            ('posix_usec', np.int64),
                            ('data', np.uint16, self.n_pre+self.n_post)])
        self._ljh_record = np.zeros(1, 
            dtype=self._ljh_record_dtype)

    def get_record_at(self, ind):
        a = ind-self.n_pre
        segnum = a // self._segment_len
        if segnum != self._loaded_segnum:
            # print(f"getoverlappingsegment {segnum}")
            self._data, self._posix_usec = getoverlappingsegment(self.ds, segnum)
            self._loaded_segnum = segnum
        i = a % self._segment_len
        data = self._data[i:i+self.n_pre+self.n_post]

        ljh_ind = i//self.ds.nSamples
        ljh_ind_rem = i%self.ds.nSamples
        posix_usec=self._posix_usec[ljh_ind] + self.ds.timebase*ljh_ind_rem
        return data, posix_usec     

    def get_ljh_record_at(self, ind):
        # re-uses self._ljh_record to avoid allocation, is that good?
        self._ljh_record["rowcount"] = ind*self.ds.number_of_rows
        data, posix_usec = self.get_record_at(ind)
        self._ljh_record["data"] = data
        self._ljh_record["posix_usec"] = posix_usec
        return self._ljh_record
    
    def ljh_record_block_generator(self, inds, max_records_per_block):
        inds = np.array(inds)
        n_blocks = len(inds)//max_records_per_block+1
        for i_block in range(n_blocks):
            print(f"writing to ljh block {i_block+1}/{n_blocks}")
            if i_block == n_blocks-1:
                # last block, may be smaller
                blocksize = len(inds)%max_records_per_block
            else:
                blocksize = max_records_per_block
            block = np.zeros(blocksize, 
            dtype=self._ljh_record_dtype)
            j0 = i_block*max_records_per_block
            for j in range(blocksize):
                block[j] = self.get_ljh_record_at(inds[j+j0])
            yield block

def ljh_run_rename_dastard(path, inc=1000):
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

def ljh_write_traces(inds, triggergetter, header_dict, dest_path, overwrite=False):
    if os.path.exists(dest_path) and not overwrite:
        raise IOError(f"The ljhfile {dest_path} exists and overwrite was not set to True")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    ljh_header = mass.files.make_ljh_header(header_dict)
    with open(dest_path, "wb") as dest_fp:
        dest_fp.write(ljh_header)
        if True:
            # write blocks, should be faster?
            for i, block in enumerate(triggergetter.ljh_record_block_generator(
                inds, max_records_per_block=1000)):
                block.tofile(dest_fp)
        else:
            # write one record at a time, simpler code
            for i, ind in enumerate(inds):
                if i%100==0:
                    print(f"{dest_path} {i}/{len(inds)}")
                record = triggergetter.get_ljh_record_at(ind)
                record.tofile(dest_fp)

