from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io

t1 = Tdms(r'E:\Documents\PythonCoding\20210730_002_2021_07_30_13_53_09.tdms')
t1.loadtdms()


x1 = [base_left[0], base_right[0]]
y1 = resp[452][x1]

peak1s, para1s = signal.find_peaks(resp, prominence=0.2, height=[None, None], rel_height=0.1, width=[0,100])
for i in range(len(peak1s)):
    _re = resp[peak1s[i]-50:peak1s[i]+200]
    plt.plot(_re)
    ax = plt.subplot()
    txt = 'Peak: '+ str(i) +'\nHeight: '+str(para1s['peak_heights'][i])+'\nProm: '+str(para1s['prominences'][i])+\
        '\nWidth: '+str(para1s['widths'][i])
    ax.text(0.9,0.9,txt, transform=ax.transAxes)
    plt.show()
    plt.clf()
    
    

peaks,_ = signal.find_peaks(resp, prominence=0.2, height=[None, None], rel_height=0.1, width=[0,100])
base_left = []
base_right = []

for peak in peaks:
        _re = resp[peak-50:peak+200]
        _re_diff = np.convolve(np.diff(_re), np.ones(10)/10, mode='same') 
        index = [i for i in range(len(_re_diff)) if np.abs(_re_diff[i] - 0) > 0.001]
        
        if index[0] > 40:
            index[0] = 25
        if index[-1] < 100:
            index[-1] = 150
        
        base_left.append(peak-50+index[0])
        base_right.append(peak-50+index[-1])
        
m = np.zeros(len(resp), dtype=bool)
for i in range(len(base_left)):
    m[base_left[i]:base_right[i]] = True
    
nopeak = resp[:]
nopeak[m] = np.nan
nopeak = pd.Series(nopeak)
nopeak = list(nopeak.interpolate(limit_direction='both', kind='cubic'))
