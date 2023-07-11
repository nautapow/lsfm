from TDMS_ver5 import Tdms_V1, Tdms_V2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy import interpolate
import scipy.io
import mne
import TFTool
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd
import lsfm
import lsfm_psth
import lsfm_slope
import math
import lsfm_strf


    
if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[df['type']=='Log sFM']
    tlsfm = [23,24,25,28,30,35,37,45,49,60,62,71,73,74,76,78,81,82,86,87,89,91,92,95,97,98,100,101]
    cell_note = pd.read_csv('cell_note_all.csv')
    
    
    df_loc = 74
    if df_loc == 74:
    #for df_loc in tlsfm:
        
        """Step 1:
            Preprocess Data
            Response -- prefilter to reduce noise
            Save fir-inverted stimulus and resp to .mat file
        """
        
        i = int([i for i,a in enumerate(tlsfm) if a == df_loc][0])
        filename = df['filename'][df_loc]
        version = df['Py_Version'][df_loc]
        cell_data = np.load(f'{filename}_lsfm.npy', allow_pickle=True)
        
        para = cell_data.item().get('para')
        stim = cell_data.item().get('stim')
        resp1 = cell_data.item().get('resp')
        stim_fir = cell_data.item().get('stim_fir')
        
        cf,band,modrate,_=zip(*para)
        band = sorted(set(band))
        cf = sorted(set(cf))
        modrate = sorted(set(modrate))
        
        resp1 = TFTool.prefilter(resp1, 25000)
        
        
        #second resp
        filename2 = df['filename'][73]
        version2 = df['Py_Version'][73]
        cell_data2 = np.load(f'{filename2}_lsfm.npy', allow_pickle=True)
        resp2 = cell_data2.item().get('resp')
        
        resp2 = TFTool.prefilter(resp2, 25000)
        
        resp = (resp1+resp2)/2
        
        
        to_matlab = {'stim':stim_fir, 'resp':resp, 'fc':cf, 'bandwidth':band, 'mod_rate': modrate}
        scipy.io.savemat(f'{filename}_data.mat', to_matlab)
        
        """Step 2:
            Reap cwt.mat file to get frequency band and stimuli
        """
        
        import mat73
        cwt = mat73.loadmat('20220527_avg_cwt.mat')
        
        s = lsfm_strf.STRF(cwt, resp, filename=filename)
        s.strf(saveplot=True)
        
        
# =============================================================================
#         """Generate Artificial Sound:
#            
#         """
#         ff, bb, mm, _ = zip(*para)
#         from sympy import Symbol, Eq, solve
#         fmax = Symbol('fmax')
#         fmin = Symbol('fmin')
#         
#         i = 50
#         eq1 = Eq(fmax*fmin, ff[i]**2)
#         eq2 = Eq(fmin*bb[i], fmax)
#         sol = solve([eq1, eq2], fmax, fmin)
#         
#         from math import sin, pi
#         def f(t, fc, bw, fm):
#             ft = fc*bw^(sin(2*pi*fm*t)/2)
#             return sin^2*pi*ft
#         
#         t = np.arange(0,60,1/25000)
#         
#         from scipy.integrate import odeint
#         fc=ff[i] 
#         bw=bb[i]
#         fm=mm[i]
#         x = odeint(f, 0, t, args=(fc, bw, fm))
#         
#         
#         hs = signal.hilbert(stim[i])
#         env = abs(hs)
#         phase = np.unwrap(np.angle(hs))
#         ins_freq = (np.diff(phase) / (2.0*np.pi) * 25000)    
#         plt.plot(phase)
#         plt.show()
#         plt.clf()
#         
#         plt.plot(np.angle(hs)[9500:11000])
# =============================================================================

        
        
        
        
        
        
        
        
        
        