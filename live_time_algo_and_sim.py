import numpy as np


def generate_timestamps(count_rate_hz, duration_s):
    """generate timestamps from a poisson process at count_rate_hz for an experiment of duration_s seconds"""
    N_expected = duration_s*count_rate_hz
    sigma = max(1, np.sqrt(N_expected))
    N = int(N_expected+15*sigma)
    time_separations_s = np.random.exponential(scale=1/count_rate_hz, size=N)
    timestamps_s = np.cumsum(time_separations_s)
    ind = np.searchsorted(timestamps_s, duration_s)
    return timestamps_s[:ind]

def live_ranges(trig_times_arb, dead_after_arb):
    """takes 
    trig_times_arb - a list of trigger times for an experiment that started at time zero, may be in any units
    dead_after_arb - the dead time to enforce after each trigger, must be in same units as first argument
    return live_ranges_arb, live_triggered_inds
    live_ranges_arb a list of tuples of values in same units representing live range
    each live range ends at an observed count
    live_triggered_inds a numpy array of indicies into trig_times_arb corresponding to the end of each live_range, 
    so corresponding to the event which a trigger was observed during a live_time"""
    assert issorted(trig_times_arb)
    live_ranges_arb = []
    live_triggered_inds = []
    live_start = 0 # start live, should we start dead?
    a,b = -np.inf, -np.inf # assume we just saw a trigger
    for i in range(len(trig_times_arb)):
        a = b
        b = trig_times_arb[i]
        if live_start>b:
            # if the next pulse happens before live start, we reject that pulse and
            # extend the dead time by not starting the live time
            live_start = b+dead_after_arb
        else:
            # otherwise the pulse is accepted, and the next live time starts after it
            live_end=b
            live_ranges_arb.append((live_start, live_end))
            live_triggered_inds.append(i)
            live_start = b+dead_after_arb
    return live_ranges_arb, np.array(live_triggered_inds)

def live_time_from_live_ranges(live_ranges_s):
    live_time_s = 0
    for (a,b) in live_ranges_s:
        live_time_s += b-a
    return live_time_s

def issorted(x):
    return np.all(np.diff(x) >= 0)

def merge_sorted_arrays_with_source_indicator(*arrays):
    merged_array = np.concatenate(arrays)
    sorting_inds = np.argsort(merged_array, kind='mergesort')
    
    source_indicator = np.zeros_like(merged_array)
    a,b=0, len(arrays[0])
    for i in range(1, len(arrays)):
        a=b
        b = a+len(arrays[i])
        source_indicator[a:b] = i
    
    return merged_array[sorting_inds], source_indicator[sorting_inds]

def random_bools(N, frac_true):
    return np.random.random(N)<frac_true


if __name__ == "__main__":
    chosen_count_rate_hz = 2.0
    N_wanted = 1e6
    chosen_duration_s = N_wanted/chosen_count_rate_hz
    timestamps_s = generate_timestamps(chosen_count_rate_hz, chosen_duration_s)
    print("class 1 events")
    for chosen_deadtime_s in np.linspace(0,4, 5):
        live_ranges_s, live_triggerd_inds = live_ranges(timestamps_s, dead_after_arb=chosen_deadtime_s)
        N_observed = len(live_ranges_s)
        if N_observed<=1:
            break
        live_time_s = live_time_from_live_ranges(live_ranges_s)
        measured_rate = N_observed/live_time_s
        measured_rate_sigma = np.sqrt(N_observed)/live_time_s
        print(f"{chosen_deadtime_s=:.2f} s, {live_time_s=:0.2f} {N_observed=}, {measured_rate:0.3f} hz +/- {measured_rate_sigma:.3f} and true {chosen_count_rate_hz:0.2f}")
    
    print()
    print("class 1 (A) and 2 (B) events")
    # now we're going to do an experiment where we have two kinds of counts from 
    # two different sources of events
    # we merge the timestamps
    for chosen_deadtime_s in np.linspace(0,4, 5):
        chosen_count_rate_hz_A = 1.0
        chosen_count_rate_hz_B = 0.01
        timestamps_s_A = generate_timestamps(chosen_count_rate_hz_A, chosen_duration_s)
        timestamps_s_B = generate_timestamps(chosen_count_rate_hz_B, chosen_duration_s)
        merged_timestamps_s, source_indicator = merge_sorted_arrays_with_source_indicator(timestamps_s_A, timestamps_s_B)
        merged_live_ranges_s, merged_live_triggered_inds = live_ranges(merged_timestamps_s, dead_after_arb=chosen_deadtime_s)
        N_observed_A = np.sum(source_indicator[merged_live_triggered_inds]==0)
        N_observed_B = np.sum(source_indicator[merged_live_triggered_inds]==1)
        live_time_s = live_time_from_live_ranges(merged_live_ranges_s)
        measured_rate_A = N_observed_A/live_time_s
        measure_rate_A_sigma = np.sqrt(N_observed_A)/live_time_s
        measured_rate_B = N_observed_B/live_time_s
        measure_rate_B_sigma = np.sqrt(N_observed_B)/live_time_s
        print(f"{chosen_deadtime_s=:.2f} s, {live_time_s=:0.3f} {N_observed_A=}, {measured_rate_A:0.3f} hz +/- {measure_rate_A_sigma:.3f} and true {chosen_count_rate_hz_A:0.2f}")
        print(f"{chosen_deadtime_s=:.2f} s, {live_time_s=:0.3f} {N_observed_B=}, {measured_rate_B:0.3f} hz +/- {measure_rate_B_sigma:.3f} and true {chosen_count_rate_hz_B:0.2f}")


    print()
    print("class 1 and class 3/4 (bad) events")
    print("since we're randomly assigning good or bad, the definition is symmetric. Therefore I can either throw away live ranges previous to bad triggers when calculate the count rate of good events, or I can throw away live ranges previous to good events to determine the count rate of bad events. In both cases I should recoved the full count rate, not the partial count rate.")
    # now we're going to look only at one population of pulses, but we're going to mark some 
    # fraciton of them bad
    # bad can represent a foil + non_foil event or an event we can't analyze for energy for another reason
    # strategy is going to be to pretend the detector was off during the live range ending in
    # this bad event
    bad_frac = 0.1
    for chosen_deadtime_s in np.linspace(0,2, 5):
        live_ranges_s_for_good, live_triggerd_inds_for_good = live_ranges(timestamps_s, dead_after_arb=chosen_deadtime_s)
        is_bad = random_bools(len(live_triggerd_inds_for_good), frac_true=bad_frac)
        N_observed_good = np.sum(is_bad==False)
        N_observed_bad = np.sum(is_bad==True)
        live_ranges_s_good = []
        live_ranges_s_bad = []
        for i in range(len(is_bad)):
            if is_bad[i] == True:
                live_ranges_s_bad.append(live_ranges_s_for_good[i])     
            else:
                live_ranges_s_good.append(live_ranges_s_for_good[i])     
        live_time_s_good = live_time_from_live_ranges(live_ranges_s_good)
        live_time_s_bad = live_time_from_live_ranges(live_ranges_s_bad)
        measured_rate_good = N_observed_good/live_time_s_good
        measured_rate_good_sigma = np.sqrt(N_observed_good)/live_time_s_good
        measured_rate_bad = N_observed_bad/live_time_s_bad
        measured_rate_bad_sigma = np.sqrt(N_observed_bad)/live_time_s_bad

        print(f"{chosen_deadtime_s=:.2f} s, {bad_frac=} {live_time_s_good=:0.2f} {N_observed_good=}, {measured_rate_good:0.3f} hz +/- {measured_rate_good_sigma:.3f} and true {chosen_count_rate_hz:0.2f}")
        print(f"{chosen_deadtime_s=:.2f} s, {bad_frac=} {live_time_s_bad=:0.2f} {N_observed_bad=}, {measured_rate_bad:0.3f} hz +/- {measured_rate_bad_sigma:.3f} and true {chosen_count_rate_hz:0.2f}")

