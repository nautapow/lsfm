from TDMS_ver5 import Tdms_V1, Tdms_V2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    idx_tone = [26,29,31,34,36,44,48,61,72,75,77,80,83,84,85,88,90,93,94,96,99]
    cell_note = pd.read_csv('cell_note_all.csv')
    
    #df_loc = 88
    #if df_loc == 88:
    for df_loc in idx_tone:
        #i = int([i for i,a in enumerate(idx_tone) if a == df_loc][0])
        filename = df['filename'][df_loc]
        version = df['Version'][df_loc]
# =============================================================================
#         cell_data = np.load(f'{filename}_tone_data.npy', allow_pickle=True)
#         
#         para = cell_data.item().get('para')
#         stim = cell_data.item().get('stim')
#         resp = cell_data.item().get('resp')
#         
#         try:
#             n = cell_note.index[cell_note['associate tone']==filename][0]
#         except:    
#             pass
#         if n:    
#             bf = cell_note['best frequency'].loc[n]
#             #puretone.psth_bf(resp, para, bf, filename, set_x_intime=True, saveplot=True)
#         
#         bf = puretone.tunning(resp, para, filename=filename, saveplot=False)
#         print(f'{filename}: {bf}')
#         
# =============================================================================
        
# =============================================================================
#         for i,p in enumerate(para):
#             puretone.tone_stim_resp(i, stim[i], resp[i], p[:2], filename)
# =============================================================================
        
        #_ = puretone.tunning(resp, para, filename=filename, saveplot=True)
        #puretone.psth(resp, filename, set_x_intime=True, saveplot=True)
        
        
        fdir = df['path'][df_loc]
        filename = df['filename'][df_loc]
        version = df['Version'][df_loc]
        if version == 1:
            t = Tdms_V1()
            t.loadtdms(fdir, protocol=1, load_sound=True, precise_timing=True)
        if version == 2 or version == 3:
            t = Tdms_V2()
            t.loadtdms(fdir, protocol=1, load_sound=True)
            

        para = t.para
        resp = np.array(t.resp_dpk)
        stim = t.sound
        LabView_ver = t.version
        
        if LabView_ver == 1.5:
            resp_merge, para_merge = puretone.resp_merge(resp, para)
            loud, freq = zip(*para_merge)
        else:
            resp_merge = resp
            loud, freq, _ = zip(*para)
        
        
# =============================================================================
#         loud = sorted(set(loud))
#         freq = sorted(set(freq))
#         resp_mesh = np.reshape(resp_merge, (len(loud), len(freq), -1))
# =============================================================================
        peak_x = puretone.psth(resp_merge, filename)
        if peak_x < 2000:
            window_peak=[peak_x-250,peak_x+250]
        else:
            window_peak=[1250,1750]
        
        try:
            puretone.tuning(resp_merge, para, filename=f'{filename}_peak', saveplot=True, window=window_peak)
            puretone.tuning(resp_merge, para, filename=f'{filename}_sust', saveplot=True, window=[2500,3000])
        except:
            pass


# =============================================================================
#         def peak_avg(arr):
#             base = np.mean(arr[:500])
#             arr = (arr-base)*100            
#             return np.mean(arr[1000:1500])
#         def sustain_avg(arr):
#             base = np.mean(arr[:500])
#             arr = (arr-base)*100            
#             return np.mean(arr[2000:2500])
#         
#         plus_1, plus_2, minus_1, minus_2 = [],[],[],[]
#         for min_loud in resp_mesh[0]:
#             resp_peak = on_avg(min_loud)
#             #resp@tone_period
#             if resp_tp>=0:
#                 base_plus.append(resp_tp)
#             elif resp_tp<=0:
#                 base_minus.append(resp_tp)
#         
#         CI_plus = stats.t.interval(alpha=0.99, df=len(base_plus)-1, loc=np.mean(base_plus), scale=stats.sem(base_plus))
#         CI_minus = stats.t.interval(alpha=0.99, df=len(base_minus)-1, loc=np.mean(base_minus), scale=stats.sem(base_minus))
#         
#         puretone.tuning(resp_merge, para, filename)
#         
#         resp_on = np.apply_along_axis(on_avg, 2, resp_mesh)
#         
#         sig_plus = 1*(resp_on>CI_plus[1])
#         sig_minus = -1*(resp_on<CI_minus[0])
#         sig = sig_plus+sig_minus
# =============================================================================

        
# =============================================================================
#         cell_data = {'stim':stim, 'resp':resp, 'para':para}
#         np.save(f'{filename}_tone_data.npy', cell_data)
# =============================================================================
        
        
        #bf = puretone.tunning(resp, para, filename=filename, set_x_intime=False, saveplot=False)
        #puretone.psth(resp, filename, set_x_intime=False, saveplot=False)
        #df_copy = df.copy()
        #df_copy['best_frequency'].iloc[df_loc] = bf
        #df.iloc[df_loc, df.columns.get_loc('best_frequency')] = bf['best_frequency']
        #df.iloc[df_loc, df.columns.get_loc('bandwidth')] = bf['bandwidth']
        #puretone.psth(resp, filename, set_x_intime=True, saveplot=True)
    
    #df.to_csv('patch_list_E.csv', index=False)
       
        
        

        
        
        