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
ilsfm = df.index[df['type']=='Log sFM']
fdir = df['path'][45]
t = Tdms()
t.loadtdms(fdir)
_,para = t.get_stim()
resp,_ = t.get_dpk()
res, prop = lsfm_analysis.para_merge(para, resp, axis=0)
power = []
n = len(res[0])
freq = np.fft.fftfreq(n, d=1/25000)
idx = freq>=0

for r in res:
    f = np.fft.fft(r)
    p = np.abs(f)**2
    power.append(p)
    
for pp in power:
    plt.plot(freq[idx], pp[idx])
    plt.xlim(0,64)
    plt.show()
    