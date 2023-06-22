import mass
import numpy as np
import pylab as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import h5py
import unittest
import fastdtw
import matplotlib as mpl
import devel

from uncertainties import ufloat
from uncertainties.umath import *
from collections import OrderedDict


def data_loadStateLabels(self):
    """ Loads state labels and makes categorical cuts"""
    # Load up experiment state file and extract timestamped state labels
    basename, _ = mass.ljh_util.ljh_basename_channum(self.first_good_dataset.filename)
    experimentStateFilename = basename + "_experiment_state.txt"
    startTimes, stateLabels = np.loadtxt(experimentStateFilename, skiprows=1, delimiter=', ',unpack=True, dtype=str)
    startTimes = np.array(startTimes, dtype=float)*1e-9    
    # Clear categorical cuts with state_label category if they already exist
    if self.cut_field_categories('state_label') != {}:
        self.unregister_categorical_cut_field("state_label")
    # Create state_label categorical cuts using timestamps and labels from experimental state file
    stateLabelsUnique = list(np.unique(stateLabels))
    self.register_categorical_cut_field("state_label", stateLabelsUnique)
    stateLabelInts = self.cut_field_categories('state_label')
    for ds in self:
        stateCodes0 = np.searchsorted(startTimes, ds.p_timestamp[:])
        # remap codes to align with stateLabelsUnique
        stateCodes = np.array([stateLabelInts[stateLabels[c-1]] for c in stateCodes0])
        ds.cuts.cut("state_label", stateCodes)

def ds_CombinedStateMask(self, statesList):
    """ Combines all states in input array to a mask """
    combinedMask = np.zeros(self.nPulses, dtype=bool)
    for iState in statesList:
        combinedMask = np.logical_or(combinedMask, self.good(state_label=iState))
    return combinedMask

def ds_shortname(self):
    """return a string containing part of the filename and the channel number, useful for labelling plots"""
    s = os.path.split(self.filename)[-1]
    chanstr = "chan%g"%self.channum
    if not chanstr in s:
        s+=chanstr
    return s

def data_shortname(self):
    """return a string containning part of the filename and the number of good channels"""
    ngoodchan = len([ds for ds in self])
    return mass.ljh_util.ljh_basename_channum(os.path.split(self.datasets[0].filename)[-1])[0]+", %g chans"%ngoodchan

def ds_hist(self,bin_edges,attr="p_energy",t0=0,tlast=1e20,category={},g_func=None, stateMask=None):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    vals = getattr(self, attr)[:]
    # sanitize the data bit
    tg = np.logical_and(self.p_timestamp[:]>t0,self.p_timestamp[:]<tlast)
    g = np.logical_and(tg,self.good(**category))
    g = np.logical_and(g,~np.isnan(vals))
    if g_func is not None:
        g=np.logical_and(g,g_func(self))
    if stateMask is not None:
        g=np.logical_and(g,stateMask)

    counts, _ = np.histogram(vals[g],bin_edges)
    return bin_centers, counts

def data_hists(self,bin_edges,attr="p_energy",t0=0,tlast=1e20,category={},g_func=None):
    """return a tuple of (bin_centers, countsdict). automatically filters out nan values
    where countsdict is a dictionary mapping channel numbers to numpy arrays of counts
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    countsdict = {ds.channum:ds.hist(bin_edges, attr,t0,tlast,category,g_func)[1] for ds in self}
    return bin_centers, countsdict

def data_hist(self, bin_edges, attr="p_energy",t0=0,tlast=1e20,category={},g_func=None):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses in all good datasets (use .hists to get the histograms individually). filters out nan values
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    bin_centers, countsdict = self.hists(bin_edges, attr,t0,tlast,category,g_func)
    counts = np.zeros_like(bin_centers, dtype="int")
    for (k,v) in countsdict.items():
        counts+=v
    return bin_centers, counts

def plot_hist(self,bin_edges,attr="p_energy",axis=None,label_lines=[],category={},g_func=None, stateMask=None):
    """plot a coadded histogram from all good datasets and all good pulses
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    axis -- if None, then create a new figure, otherwise plot onto this axis
    annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    if axis is None:
        plt.figure()
        axis=plt.gca()
    x,y = self.hist(bin_edges, attr, category=category, g_func=g_func, stateMask=stateMask)
    axis.plot(x,y,drawstyle="steps-mid")
    axis.set_xlabel(attr)
    axis.set_ylabel("counts per %0.1f unit bin"%(bin_edges[1]-bin_edges[0]))
    axis.set_title(self.shortname())
    annotate_lines(axis, label_lines)

def data_plot_hist(self,bin_edges,attr="p_energy",axis=None,label_lines=[],category={},g_func=None, stateMask=None):
    """plot a coadded histogram from all good datasets and all good pulses
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    axis -- if None, then create a new figure, otherwise plot onto this axis
    annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    if axis is None:
        plt.figure()
        axis=plt.gca()
    x,y = self.hist(bin_edges, attr, category=category, g_func=g_func)
    axis.plot(x,y,drawstyle="steps-mid")
    axis.set_xlabel(attr)
    axis.set_ylabel("counts per %0.1f unit bin"%(bin_edges[1]-bin_edges[0]))
    axis.set_title(self.shortname())
    annotate_lines(axis, label_lines)

def annotate_lines(axis,label_lines, label_lines_color2=[],color1 = "k",color2="r"):
    """Annotate plot on axis with line names.
    label_lines -- eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    label_lines_color2 -- optional,eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    color1 -- text color for label_lines
    color2 -- text color for label_lines_color2
    """
    n=len(label_lines)+len(label_lines_color2)
    yscale = plt.gca().get_yscale()
    for (i,label_line) in enumerate(label_lines):
        energy = mass.STANDARD_FEATURES[label_line]
        if yscale=="linear":
            axis.annotate(label_line, (energy, (1+i)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color1)
        elif yscale=="log":
            axis.annotate(label_line, (energy, np.exp((1+i)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data",color=color1)
    for (j,label_line) in enumerate(label_lines_color2):
        energy = mass.STANDARD_FEATURES[label_line]
        if yscale=="linear":
            axis.annotate(label_line, (energy, (2+i+j)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color2)
        elif yscale=="log":
            axis.annotate(label_line, (energy, np.exp((2+i+j)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data",color=color2)

def ds_linefit(self,line_name="MnKAlpha", t0=0,tlast=1e20,axis=None,dlo=50,dhi=50,
               binsize=1,bin_edges=None, attr="p_energy",label="full",plot=True,
               guess_params=None, ph_units="eV", category={}, g_func=None,holdvals={},
               stateMask=None):
    """Do a fit to `line_name` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other way you like.
    line_name -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
    dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
    bin_edges -- pass the bin_edges you want as a numpy array
    attr -- default is "p_energy", you could pick "p_filt_value" or others. be sure to pass in bin_edges as well because the default calculation will probably fail for anything other than p_energy
    label -- passed to fitter.plot
    plot -- passed to fitter.fit, determine if plot happens
    guess_params -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
    ph_units -- passed to fitter.fit, used in plot label
    category -- pass {"side":"A"} or similar to use categorical cuts
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
    holdvals -- a dictionary mapping keys from fitter.params_meaning to values... eg {"background":0, "dP_dE":1}
        This vector is anded with the vector calculated by the histogrammer
    """
    if isinstance(line_name, mass.LineFitter):
        fitter = line_name
        nominal_peak_energy = fitter.spect.nominal_peak_energy
    elif isinstance(line_name,str):
        fitter = mass.fitter_classes[line_name]()
        nominal_peak_energy = fitter.spect.nominal_peak_energy
    else:
        fitter = mass.GaussianFitter()
        nominal_peak_energy = float(line_name)
    if bin_edges is None:
        bin_edges = np.arange(nominal_peak_energy-dlo, nominal_peak_energy+dhi, binsize)
    if axis is None and plot:
        plt.figure()
        axis = plt.gca()

    bin_centers, counts = self.hist(bin_edges, attr, t0, tlast, category, g_func, stateMask=stateMask)

    if guess_params is None:
        guess_params = fitter.guess_starting_params(counts,bin_centers)
    hold = []
    for (k,v) in holdvals.items():
        i = fitter.param_meaning[k]
        guess_params[i]=v
        hold.append(i)
    params, covar = fitter.fit(counts, bin_centers,params=guess_params,axis=axis,label=label, ph_units=ph_units,plot=plot, hold=hold)
    if plot:
        axis.set_title(self.shortname()+", {}".format(line_name))

    return fitter

def data_linefit(self,line_name="MnKAlpha", t0=0,tlast=1e20,axis=None,dlo=50,dhi=50,
               binsize=1,bin_edges=None, attr="p_energy",label="full",plot=True,
               guess_params=None, ph_units="eV", category={}, g_func=None,holdvals={},):
    """Do a fit to `line_name` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other way you like.
    line_name -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
    dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
    bin_edges -- pass the bin_edges you want as a numpy array
    attr -- default is "p_energy", you could pick "p_filt_value" or others. be sure to pass in bin_edges as well because the default calculation will probably fail for anything other than p_energy
    label -- passed to fitter.plot
    plot -- passed to fitter.fit, determine if plot happens
    guess_params -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
    ph_units -- passed to fitter.fit, used in plot label
    category -- pass {"side":"A"} or similar to use categorical cuts
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
    holdvals -- a dictionary mapping keys from fitter.params_meaning to values... eg {"background":0, "dP_dE":1}
        This vector is anded with the vector calculated by the histogrammer
    """
    if isinstance(line_name, mass.LineFitter):
        fitter = line_name
        nominal_peak_energy = fitter.spect.nominal_peak_energy
    elif isinstance(line_name,str):
        fitter = mass.fitter_classes[line_name]()
        nominal_peak_energy = fitter.spect.nominal_peak_energy
    else:
        fitter = mass.GaussianFitter()
        nominal_peak_energy = float(line_name)
    if bin_edges is None:
        bin_edges = np.arange(nominal_peak_energy-dlo, nominal_peak_energy+dhi, binsize)
    if axis is None and plot:
        plt.figure()
        axis = plt.gca()

    bin_centers, counts = self.hist(bin_edges, attr, t0, tlast, category, g_func)

    if guess_params is None:
        guess_params = fitter.guess_starting_params(counts,bin_centers)
    hold = []
    for (k,v) in holdvals.items():
        i = fitter.param_meaning[k]
        guess_params[i]=v
        hold.append(i)
    params, covar = fitter.fit(counts, bin_centers,params=guess_params,axis=axis,label=label, ph_units=ph_units,plot=plot, hold=hold)
    if plot:
        axis.set_title(self.shortname()+", {}".format(line_name))

    return fitter


def samepeaks(bin_centers, countsdict, npeaks, refchannel, gaussian_fwhm):
    raise ValueError("Not done!")
    refcounts = countsdict[refchannel]
    peak_locations, peak_intensities = mass.find_local_maxima(refcounts, peak_intensities)



def ds_rowtime(self):
    """
    Return the row time in seconds. The row time required to make a single sample for a simple row, and the frame time is equal to the row time times the number of rows.
    """
    nrow = self.pulse_records.datafile.number_of_rows
    rowtime = self.timebase/nrow
    return rowtime

def ds_cut_calculated(ds):
    """
    If you open a pope hdf5 file there will be no cuts applied, but there will be ranges for cuts. This function
    looks up those ranges, and uses them to do actual cuts.
    """
    ds.cuts.clear_cut("pretrigger_rms")
    ds.cuts.clear_cut("postpeak_deriv")
    ds.cuts.clear_cut("pretrigger_mean")
    ds.cuts.cut_parameter(ds.p_pretrig_rms, ds.hdf5_group["calculated_cuts"]["pretrig_rms"][:], 'pretrigger_rms')
    ds.cuts.cut_parameter(ds.p_postpeak_deriv, ds.hdf5_group["calculated_cuts"]["postpeak_deriv"][:], 'postpeak_deriv')

def ds_plot_ptmean_vs_time(ds,t0,tlast):
    plt.figure()
    plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()])
    plt.xlabel("time after first pulse (s)")
    plt.ylabel("p_pretrig_mean (arb)")


mass.TESGroup.loadStateLabels = data_loadStateLabels
mass.TESGroup.plot_hist = data_plot_hist
mass.TESGroup.hist = data_hist
mass.TESGroup.hists = data_hists
mass.TESGroup.shortname = data_shortname
mass.TESGroup.linefit = data_linefit

mass.MicrocalDataSet.CombinedStateMask =ds_CombinedStateMask
mass.MicrocalDataSet.hist = ds_hist
mass.MicrocalDataSet.plot_hist = plot_hist
mass.MicrocalDataSet.shortname = ds_shortname
mass.MicrocalDataSet.linefit = ds_linefit
mass.MicrocalDataSet.rowtime = ds_rowtime
mass.MicrocalDataSet.cut_calculated = ds_cut_calculated
mass.MicrocalDataSet.plot_ptmean_vs_time = ds_plot_ptmean_vs_time

def expand_cal_lines(s):
    """Return a list of line names, eg ["MnKAlpha","MnKBeta"]
    s -- a list containing line names and/or element symbols eg ["Mn","TiKAlpha"]
    """
    out = []
    for symbol in s:
        if symbol == "": continue
        if len(symbol) == 2:
            out.append(symbol+"KBeta")
            out.append(symbol+"KAlpha")
        elif not symbol in out:
            out.append(symbol)
    return out





class PredictedVsAchieved():
    def __init__(self, data, calibration, fitters):
        """
        Call this.plot() after initialization. For each ds in data calculated the predicted energy resolution at the average pulse (using the
        calibration to get the average pulse energy, does no nonlinearity correction). Then looks up the
        achieved energy resolution in the fitters. And plots it all.
        data -- a TESChannelGroup
        calibration -- a calibraiton name, eg "p_filt_value_tdc"
        fitters -- a dictionary mapping channel number to a fitter at line
        """
        self.data = data
        self.calibration = calibration
        self.fitters = fitters

    @property
    def vdvs(self):
        preds = []
        for ds in self.data:
            d = ds.filter.predicted_v_over_dv
            if d.has_key("filt_noconst"):
                preds.append(ds.filter.predicted_v_over_dv["filt_noconst"])
            elif d.has_key("noconst"):
                preds.append(ds.filter.predicted_v_over_dv["noconst"])

        return np.array(preds)
    @property
    def average_pulse_energies(self):
        energies = []
        for ds in self.data:
            calibration = ds.calibration[self.calibration]
            energy = calibration(np.amax(ds.average_pulse)-np.amin(ds.average_pulse))
            energies.append(energy)
        return np.array(energies)
    @property
    def channels(self):
        return [ds.channum for ds in self.data]
    @property
    def predicted_at_average_pulse(self):
        return self.average_pulse_energies/self.vdvs
    @property
    def achieved(self):
        return np.array([self.fitters[ds.channum].last_fit_params_dict["resolution"][0] for ds in self.data])
    @property
    def fitter_line_name(self):
        fitter = self.fitters.values()[0]
        return fitter.spect.name
    def plot(self):
        plt.figure()
        predicted = self.predicted_at_average_pulse
        achieved = self.achieved
        med_pred = np.median(predicted)
        med_ach = np.median(achieved)
        xlim_max = min(2*med_pred, np.amax(predicted)*1.05)
        ylim_max = min(2*med_ach, np.amax(achieved)*1.05)
        lim_max = max(xlim_max, ylim_max)
        plt.plot(predicted, achieved,"o")
        plt.plot([0, lim_max], [0, lim_max],"k")
        plt.xlim(0,lim_max)
        plt.ylim(0,lim_max)
        plt.xlabel("predicted res at avg pulse (eV)")
        plt.ylabel("achieved at %s"%self.fitter_line_name)
        plt.title("median predicted %0.2f, median achieved %0.2f\n%s"%(med_pred, med_ach,self.data.shortname()))







class TestPlotAndHistMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data = mass.TESGroupHDF5("/Users/oneilg/Documents/molecular movies/Horton Wiring/horton_2017_07/20171006_B.ljh_pope.hdf5")
        self.data.set_chan_good(self.data.why_chan_bad.keys())
        for ds in self.data:
            ds_cut_calculated(ds)
            print("Chan %s, %g bad of %g"%(ds.channum, ds.bad().sum(), ds.nPulses))
        self.ds = self.data.first_good_dataset

        self.bin_edges  = np.arange(0,10000,1)

    def test_ds_hist(self):
        x,y = self.ds.hist(self.bin_edges)
        self.assertEqual(x[0], 0.5*(self.bin_edges[1]+self.bin_edges[0]))
        self.assertEqual(np.argmax(y),4511)

    def test_data_hists(self):
        x,countsdict = self.data.hists(self.bin_edges)
        self.assertTrue(all(countsdict[self.ds.channum]==self.ds.hist(self.bin_edges)[1]))
        self.assertTrue(len(x)==len(self.bin_edges)-1)
        len(countsdict.keys())==len([ds for ds in self.data])

    def test_plots(self):
        self.ds.plot_hist(self.bin_edges, label_lines = ["MnKAlpha","MnKBeta"])
        self.data.plot_hist(self.bin_edges, label_lines = ["MnKAlpha","MnKBeta"])

    def test_linefit(self):
        fitter = self.ds.linefit("MnKAlpha")
        self.assertTrue(fitter.success)

    def test_linefit_pass_fitter(self):
        fitter = self.ds.linefit(mass.MnKAlphaFitter(), bin_edges = np.arange(5850,5950), attr="p_energy")
        self.assertTrue(fitter.success)

    def test_rank_hists_chisq(self):
        ws=WorstSpectra(self.data)
        ws.output()
        ws.plot()

def find_pulses_with_properties(self, dlohi: dict):
    """takes a dict mapping attr name (eg "p_filt_value") to a tuple (lo, hi)
    returns the indicies of pulses that have each attr between the corresponding lo, hi
    """
    assert len(dlohi) > 0
    valid = self.good()
    for attr_name, (lo, hi) in dlohi.items():
        attr = getattr(self, attr_name)[:]
        valid = np.logical_and(valid, attr > lo)
        valid = np.logical_and(valid, attr < hi)
    inds = np.where(valid)[0]
    return inds
mass.MicrocalDataSet.find_pulses_with_properties = find_pulses_with_properties

def plot_hist2d(self, attr1, attr2, bins1, bins2, norm=mpl.colors.LogNorm(), **kws):
    plt.figure()
    g = self.good()
    a1 = getattr(self, attr1)[g]
    a2 = getattr(self, attr2)[g]
    counts, x_edges, y_edges, _ = plt.hist2d(a1, a2, bins=[bins1, bins2], norm=norm, **kws)
    plt.xlabel(attr1)
    plt.ylabel(attr2)
    plt.colorbar()
    plt.title(f"{self.shortname()}")
mass.MicrocalDataSet.plot_hist2d = plot_hist2d

def plot_pulses_by_energy(self, e_centers_ev, indss, xlim=None):
    # not general enough for mass
    plt.figure()
    traces = []
    e_ev_matching_traces = []
    for i, (e_ev, inds) in enumerate(zip(e_centers_ev, indss)):
        if len(inds) == 0: # skip when we have no data
            continue
        # if i%2==0: # plot fewer traces
        #     continue
        # if e_ev>1200: # plot fewer traces
        #     break
        lw = 1
        if e_ev == 700:
            lw=2
        plt.plot(np.arange(self.nSamples)*self.timebase, 
        self.read_trace(inds[0]), label=f"{e_ev:0.1f} eV", lw=lw)
    plt.xlabel("time (s)")
    plt.ylabel("mix value (arb)")
    plt.legend()
    plt.title(f"{self.shortname()}, predicted={self.predicted_fwhm_felalpha_ev:.2} eV")
    if xlim is not None:
        plt.xlim(xlim)
mass.MicrocalDataSet.plot_pulses_by_energy = plot_pulses_by_energy

def plot_slew_rate(self, e_centers_ev, indss, i_hi, i_lo, n_trace_avg):
    delta_mix = []
    delta_mix_min = []
    e_ev_matching_delta_mix = []
    traces = []
    for e_ev, inds in zip(e_centers_ev, indss):
        if len(inds) < n_trace_avg:
            continue
        trace = np.zeros(self.nSamples)
        for i in range(n_trace_avg):
            trace += self.read_trace(inds[i])
        trace /= n_trace_avg
        diff = np.diff(trace[i_lo:i_hi]) 
        # define slew rate as peak difference in mix units between two samples
        delta_mix.append(np.amax(diff))
        # also look for the peak downward slew rate
        delta_mix_min.append(np.amin(diff))
        e_ev_matching_delta_mix.append(e_ev)
        traces.append(trace)
    v_per_dac = 1/(2**16) # in ljh record with mix on, the full range is 2**16, 
    # even though the hardware dac range is 2**14
    Min_over_Mfb = 3.46 # ratio of tes (in) current to fb current
    # Min and M_fb have units flux/current
    Rfb_ohm = 4e3
    delta_t_s = self.timebase
    amps_per_mix = v_per_dac/Rfb_ohm/Min_over_Mfb
    slew_per_delta_mix = amps_per_mix/delta_t_s
    plt.figure()
    plt.plot(e_ev_matching_delta_mix, np.array(delta_mix)*slew_per_delta_mix, "o", label="up")
    plt.plot(e_ev_matching_delta_mix, -np.array(delta_mix_min)*slew_per_delta_mix, "o", label="-down")
    plt.xlabel("energy (from pulse area) (eV)")
    plt.ylabel("initial slew rate (A/s)")
    plt.title(f"""{self.shortname()}, predicted={self.predicted_fwhm_felalpha_ev:.2} eV
    i_hi={i_hi} i_lo={i_lo} n_trace_avg={n_trace_avg} slew_per_delta_mix={slew_per_delta_mix:.2g}""")
    return (e_ev_matching_delta_mix, np.array(delta_mix)*slew_per_delta_mix, 
    np.array(delta_mix_min)*slew_per_delta_mix, traces, amps_per_mix)
mass.MicrocalDataSet.plot_slew_rate = plot_slew_rate

def plot_noise_ds(self, axis=None, amps_per_mix=None):
    if axis is None:
        plt.figure()
        axis=plt.gca()
    axis.plot()
    df = self.noise_psd.attrs['delta_f']
    if amps_per_mix is None:
        yvalue = self.noise_psd[:]**0.5
    else:
        yvalue = amps_per_mix*self.noise_psd[:]**0.5
    freq = np.arange(1, 1 + len(yvalue)) * df
    axis.plot(freq, yvalue)
    axis.set_xlim([freq[1] * 0.9, freq[-1] * 1.1])
    if amps_per_mix is None:
        axis.set_ylabel("PSD$^{1/2}$ (mix/Hz$^{1/2}$)")
    else:
        axis.set_ylabel("PSD$^{1/2}$ (A/Hz$^{1/2}$)")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_title(self.shortname())
    axis.set_xscale("log")
mass.MicrocalDataSet.plot_noise = plot_noise_ds

def get_noise_lo_f_hi_f(self, n_bins_lo, n_bins_hi, amps_per_mix):
    df = self.noise_psd.attrs['delta_f']
    freq = np.arange(1, 1 + len(self.noise_psd)) * df
    noise_lo = amps_per_mix*np.mean(self.noise_psd[:n_bins_lo])**0.5
    noise_hi = amps_per_mix*self.noise_psd[-n_bins_hi:]**0.5
    freq_lo = np.mean(freq[:n_bins_lo])
    freq_hi = np.mean(freq[-n_bins_hi:])
    return  (freq_lo, noise_lo, freq_hi, noise_hi)
mass.MicrocalDataSet.get_noise_lo_f_hi_f = get_noise_lo_f_hi_f


if __name__ == "__main__":
    unittest.findTestCases("__main__").debug()
    unittest.main()
    plt.show()
