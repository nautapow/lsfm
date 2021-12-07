from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
import scipy.io
from ssqueezepy import ssq_cwt, ssq_stft
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import cwt, ssq_cwt, ssq_stft
import TFTool
import pandas as pd
import lsfm_analysis

if  __name__ == "__main__":
    df = pd.read_csv('patch_list_USBMAC.csv', dtype={'date':str, '#':str})
    ilsfm = df.index[df['type']=='Log sFM']
    fdir = df['path'][28]
    t = Tdms()
    t.loadtdms(fdir, load_sound=False)
    _,para = t.get_stim()
    resp,_ = t.get_dpk()
    n = len(resp[0])
    freq = np.fft.fftfreq(n, d=1/25000)
    mask = freq>=0
    resp_z = stats.zscore(resp)
    resp_fft = np.abs(np.fft.fft(resp_z)**2)
    
    
    res, prop = lsfm_analysis.para_merge(para, resp_fft, axis=0)
    power = []
    target_freq = np.arange(1.0,257.0)
    ext_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    oct_freq = [4.0, 32.0, 256.0]
    idx_freq = [i for i, a in enumerate(freq) if a in target_freq]

    power_at_freq=[]
    for pp in res:
        power_at_freq.append(pp[idx_freq])

    for i,a in enumerate(power_at_freq):
        plt.scatter(target_freq, a, s=12)
        plt.xscale('log')
        ax = plt.subplot()
        if prop['axis'] == 1:
            txt = prop['parameter'] + '\n %.1f kHz' % prop['set'][i]
        elif prop['axis'] == 0:
            txt = prop['parameter'] + '\n %i Hz' % prop['set'][i]
        elif prop['axis'] == 2:
            txt = prop['parameter'] + '\n %.5f octave' % prop['set'][i]
        ax.text(0.95,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
        for xc in label_freq:
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
        for xc in oct_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
        plt.show()
    
    """merge 2
    res, prop = lsfm_analysis.para_merge2(para, resp_fft, axis=2)
    power = []
    target_freq = np.arange(1.0,257.0)
    #label_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    oct_freq = [4.0, 32.0, 256.0]
    idx_freq = [i for i,a in enumerate(freq) if a in target_freq]

    power_at_freq=[]
    for pp in res:
        power_at_freq.append(pp[idx_freq])

    for i,a in enumerate(power_at_freq):
        plt.scatter(target_freq, a, s=12)
        plt.xscale('log')
        ax = plt.subplot()
        if prop['axis'] == 0:
            txt = prop['parameter'] + '\n %.1f kHz' % prop['set1'][i] + '\n %.5f octave' % prop['set2'][i]
        elif prop['axis'] == 1:
            txt = prop['parameter'] + '\n %i Hz' % prop['set1'][i] + '\n %.5f octave' % prop['set2'][i]
        else:
            txt = prop['parameter'] + '\n %i Hz' % prop['set1'][i] + '\n %.1f kHz' % prop['set2'][i]
        ax.text(0.95,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
        for xc in label_freq:
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
        for xc in oct_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
        plt.show()
    """