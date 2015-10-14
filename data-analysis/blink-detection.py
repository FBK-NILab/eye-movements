import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.measurements import center_of_mass
# This script takes as input an ascii file coming from eye - link detector.
# It returns an ascii file with 3 columns:
# Frame ID | X | Y |
# Usage:
# python frame_windows.py input_file.asc
# where input_file.asc is the ascii file containing the relevant eyelink data.

# Import eyetracker I/O library:
# https://github.com/beOn/cili
from cili.util import *
from cili.cleanup import *

# Load the data from an eyelink: 
samps,events=load_eyelink_dataset(sys.argv[1])
blink_starts=np.argwhere(np.diff(1*np.isnan(np.array(samps.y_l)))==1)
blink_ends=np.argwhere(np.diff(1*np.isnan(np.array(samps.y_l)))==-1)

blink_starts=np.delete(blink_starts,-1)
eye_track=samps.y_l
clean_eye_track=np.vstack([samps.x_l,samps.x_l]).T

std_eval=list()
for b in np.arange(blink_starts.shape[0]):
	blink_length=(blink_ends[b] - blink_starts[b])
	if (blink_length > 100):
		std_eval.append([np.std(eye_track[(blink_starts[b]-i):(blink_starts[b]-1)]) for i in np.arange(2,102)])
		print(blink_starts[b])
		clean_eye_track[(blink_starts[b]-100):(blink_starts[b]),0:2]=0
