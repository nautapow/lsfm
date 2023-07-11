from TDMS_ver4 import Tdms_V1, Tdms_V2
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
    idx_lsfm = df.index[(df['type']=='Log sFM')&(df['hard_exclude']!='exclude')]
    mod_oc, band_oc, combo_oc = [],[],[]
    
    
    #df_loc = 142
    #if df_loc == 142:
    for df_loc in idx_lsfm:
        #i = int([i for i,a in enumerate(tlsfm) if a == df_loc][0])
        filename = df['filename'][df_loc]
        print(filename)
        try:
            version = df['Py_version'][df_loc]
            cell_data = np.load(f'{filename}_lsfm.npy', allow_pickle=True)
            
            para = cell_data.item().get('para')
            stim = cell_data.item().get('stim')
            resp = cell_data.item().get('resp')
            #slope_lags = cell_data.item().get('slope_lags')
            
            cf,band,modrate,_=zip(*para)
            band = sorted(set(band))
            cf = sorted(set(cf))
            modrate = sorted(set(modrate))
            
            resp_filt = TFTool.prefilter(resp, 25000)
            resp_adjust = [r - np.mean(r[400:500]) for r in resp_filt]
            resp_intensity = [np.mean(r[500:3000]) for r in resp_adjust]
            n = int(len(resp)*1/5)
            idx_top20 = np.argpartition(resp_intensity, -n)[-n:]
            band20 = [para[i][1] for i in idx_top20]
            mod20 = [para[i][2] for i in idx_top20]
                    
            for i in idx_top20:
                combo_oc.append((para[i][1],para[i][2]))
            
            
            from collections import Counter
            _m = Counter(mod20).most_common(3)
            _m = [_m[i][0] for i in range(3)]
            mod_oc.append(_m)
            
            _b = Counter(band20).most_common(3)
            _b = [_b[i][0] for i in range(3)]
            band_oc.append(_b)
        except:
            print(f'fail for {filename}')

    mod_top = [i for k in mod_oc for i in k]
    band_top = [i for k in band_oc for i in k]
    combo_top = Counter(combo_oc)
        
        
            