from TDMS_ver3 import Tdms_V1, Tdms_V2
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
import puretone 
import math


if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_puretone = df.index[df['type']=='Pure Tones']
    idx_tone = [26,29,31,34,36,44,48,50,61,72,75,77,80]
    df['best_frequency'] = np.nan
    df['bandwidth'] = np.nan
    
    df_loc = 34
    if df_loc == 34:
    #for df_loc in idx_tone:
        
        fdir = df['path'][df_loc]
        filename = df['filename'][df_loc]
        version = df['Version'][df_loc]
        if version == 1:
            t = Tdms_V1()
            t.loadtdms(fdir, protocol=1, load_sound=False, precise_timing=True)
        if version == 2:
            t = Tdms_V2()
            t.loadtdms(fdir, protocol=1, load_sound=False)
            

        para = t.Para
        resp = np.array(t.Rdpk)
        #sound = t.rawS
        stim = t.Sound
        

        
        bf = puretone.tunning(resp, para, filename=filename, set_x_intime=False, saveplot=False)
        puretone.psth(resp, filename, set_x_intime=False, saveplot=False)
        #df_copy = df.copy()
        #df_copy['best_frequency'].iloc[df_loc] = bf
        #df.iloc[df_loc, df.columns.get_loc('best_frequency')] = bf['best_frequency']
        #df.iloc[df_loc, df.columns.get_loc('bandwidth')] = bf['bandwidth']
        #puretone.psth(resp, filename, set_x_intime=True, saveplot=True)
    
    #df.to_csv('patch_list_E.csv', index=False)
       
        
        

        
        
        