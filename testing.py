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
ilsfm = df.index[df['type']=='Log sFM']
fdir = df['path'][45]
t = Tdms()
t.loadtdms(fdir, load_sound=False)
_,para = t.get_stim()
resp,_ = t.get_dpk()
res, prop = lsfm_analysis.para_merge2(para, resp, axis=2)
power = []
n = len(res[0])
freq = np.fft.fftfreq(n, d=1/25000)
target_freq = np.arange(1.0,257.0)
#label_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
oct_freq = [4.0, 32.0, 256.0]
idx_freq = [i for i,a in enumerate(freq) if a in target_freq]
mask = freq>=0

for r in res:
    f = np.fft.fft(r)
    p = np.abs(f)**2
    power.append(p)

power_at_freq=[]
for pp in power:
    power_at_freq.append(pp[idx_freq])

for i,a in enumerate(power_at_freq):
    plt.scatter(target_freq, a, s=12)
    plt.xscale('log')
    ax = plt.subplot()
    txt = prop['axis'] + #'\n %i Hz' % prop['set1'][i] + '\n %i' %prop['set2'][i]
    ax.text(0.9,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
    for xc in label_freq:
        plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
    for xc in oct_freq:
        plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
    plt.show()
    
for i,a in enumerate(power_at_freq):
    plt.scatter(target_freq, a, s=12)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.subplot()
    txt = prop['axis'] + #'\n %i Hz' % prop['set'][i]
    ax.text(0.9,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
    for xc in label_freq:
        plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
    for xc in oct_freq:
        plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
    plt.show()

    

    