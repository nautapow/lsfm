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
    idx_tone = [26,29,31,34,36,44,48,61,72,75,77,80]
    cell_note = pd.read_csv('cell_note_all.csv')
    bf_cell=[[],[]]
    kk=[]
    
    #df_loc = 72
    #if df_loc == 72:
    for df_loc in idx_tone:
        i = int([i for i,a in enumerate(idx_tone) if a == df_loc][0])
        filename = df['filename'][df_loc]
        version = df['Version'][df_loc]
        cell_data = np.load(f'{filename}_tone_data.npy', allow_pickle=True)
        
        para = cell_data.item().get('para')
        stim = cell_data.item().get('stim')
        resp = cell_data.item().get('resp')
        
        try:
            n = cell_note.index[cell_note['associate tone']==filename][0]
        except:    
            pass
        if n:    
            bf = cell_note['best frequency'].loc[n]
            #puretone.psth_bf(resp, para, bf, filename, set_x_intime=True, saveplot=True)
        
        #bf = puretone.tunning(resp, para, filename=filename, saveplot=False)
        
        
# =============================================================================
#         for i,p in enumerate(para):
#             puretone.tone_stim_resp(i, stim[i], resp[i], p[:2], filename)
# =============================================================================
        
        bf = puretone.tunning(resp, para, filename=filename, saveplot=True)
        #puretone.psth(resp, filename, set_x_intime=True, saveplot=True)
        
# =============================================================================
#         
#         fdir = df['path'][df_loc]
#         filename = df['filename'][df_loc]
#         version = df['Version'][df_loc]
#         if version == 1:
#             t = Tdms_V1()
#             t.loadtdms(fdir, protocol=1, load_sound=True, precise_timing=True)
#         if version == 2:
#             t = Tdms_V2()
#             t.loadtdms(fdir, protocol=1, load_sound=True)
#             
# 
#         para = t.Para
#         resp = np.array(t.Rdpk)
#         stim = t.Sound
#         
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
       
        
        

        
        
        