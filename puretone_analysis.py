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
    
    #df_loc = 26
    #if df_loc == 26:
    for df_loc in idx_puretone:
        try:
            fdir = df['path'][df_loc]
            filename = df['date'][df_loc]+'_'+df['#'][df_loc]
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
            #stim = t.Sound
            
            puretone.tunning(resp, para, filename=filename, saveplot=True)
            print(filename)
        except:
            pass
       
        
        

        
        
        