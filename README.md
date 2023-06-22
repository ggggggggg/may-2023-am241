## Walkthrough for existing code for 20230515_0000 chan 2

# retrigger
Here we open the existing continuous LJH files and re-trigger. The reason for this is that since we're aiming for absolute count rate, we want to be able to re-analyze the data different ways to make sure we get the same answer, and don't "lock in" a poor triggering choice by using the built in triggering in DASTARD. This is not a normal mode of operation, so all the code is one off and a bit janky. I'm showing how to do one channel, and then you'll have to take over and figure out how to manage this on many channels and to generalize and move forward.

Also all of this code is fairly slow. Python is a terrible language for this kind of processing, but I used it anyway because that's what the rest of our tools are in. I did take efforts to write it in a faster way for python, but it's still quite slow.

Upon re-running the code, the various created files will be overwritten. Sometimes you will need to exit ipython and restart to get the code to re-run if you already have a file open that the code wants to overwrite.

1. cd to this directory, `may 2023 am241`
2. either copy the needed data files into this directory, or edit the paths at the satrt of 20230515_0000_retrigger.py to point to your data dir
2. ipython
3. %run 20230515_0000_retrigger.py
4. on line 48 (as of this writing) is `pulse_inds_filter = ljh.fasttrig_filter(imax=ljh.nPulses//10, filter=filter, threshold=200)`, the imax argument is used to process only part of the ljh file. Here I'm processing the first 10% of the file. Remove the `//10` to process the whole file. 
5. inspect plots, the triggering plot should identify the pulses well, the one record plot should have a record with the rise at x value of 0, the pretrigger mean plot vs time has a jump about halfway through (this is bad for energy resolution without extra code)
6. check for output ljh files. In your data directory there should now be a run number 1000 and run number 2000 ljh directory. 1000 is the pulse files, 2000 is noise files (taken from between pulses in our continuous data)

# explore a bit and create off files

1. if you copied files to this directory, no need to change anything, but if not you need to edit paths to point to your data dir in `20230515_0000_explore_and_make_off.py`
2. exit ipython (to make sure we don't have weird state hanging around that breaks reproducibility)
3. ipython
4. `%run 20230515_0000_explore_and_make_off.py`
5. inspect plots, there should be a lot