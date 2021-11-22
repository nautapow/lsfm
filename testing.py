from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io
import pandas as pd
import TFTool
import lsfm_analysis

df = pd.read_csv('patch_list_USBMAC.csv', dtype={'date':str, '#':str})
ilsfm = df.index[df['type']=='Pure Tones']
fdir = df['path'][29]
t = Tdms()
t.loadtdms(fdir, protocol=1, precise_timing=True)
stim,para = t.get_stim()
resp,_ = t.get_dpk()
sound = t.get_sound()