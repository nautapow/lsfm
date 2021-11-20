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

df = pd.read_csv('patch_list_Q.csv', dtype={'date':str, '#':str})
ilsfm = df.index[df['type']=='Pure Tones']
fdir = df['path'][29]
t = Tdms()
t.loadtdms(fdir, protocol=1)
_,para = t.get_stim()
resp,_ = t.get_dpk()
sound = t.get_sound()
raws,_ = t.get_raw()
raws -= np.mean(raws)
hil = signal.hilbert(raws)

a=650000
b=950000
plt.plot(raws[a:b])
plt.plot(np.abs(hil[a:b]), c='r')
