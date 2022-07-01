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
import lsfm
import lsfm_psth
import lsfm_slope
import math


    
if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[df['type']=='Log sFM']
    tlsfm = [23,24,25,28,30,35,37,45,49,60,62,71,73,74,76,78,81,82]

    lsfm.resp_overcell(df, tlsfm)
    
# =============================================================================
#     #df_loc = 35
#     #if df_loc == 35:
#     #for i in range(len(tlsfm)):
#     for df_loc in tlsfm:
#         i = int([i for i,a in enumerate(tlsfm) if a == df_loc][0])
#         filename = df['filename'][df_loc]
#         version = df['Version'][df_loc]
#         cell_data = np.load(f'{filename}_data.npy', allow_pickle=True)
#         
#         para = cell_data.item().get('para')
#         stim = cell_data.item().get('stim')
#         resp = cell_data.item().get('resp')
#         slope_lags = cell_data.item().get('slope_lag')
# =============================================================================
        

        
# =============================================================================
#         for i,p in enumerate(para):
#             if i == 328:
#                 lsfm.stim_resp(i, stim[i], resp[i], p[:3], filename, saveplot=True)
# =============================================================================
        
            
# =============================================================================
#         
#         if version == 1:
#             p = lsfm_psth.Psth(resp, para, filename)
#         elif version == 2:
#             p = lsfm_psth.Psth_New(resp, para, filename)
#         
#         p.psth_all(plot=False, saveplot=True)
#         resp_by_para = p.psth_para(plot=False, saveplot=True)
#         p.psth_trend(saveplot=True)
#         
#         lags = np.linspace(0, 100, 51)
#         slope_lags = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, filename=filename, plot=False, saveplot=False)
#         cell_data = {'stim':stim, 'resp':resp, 'para':para, 'resp_by_para':resp_by_para, 'slope_lags':slope_lags}
#         np.save(f'{filename}_data.npy', cell_data)
# =============================================================================
        

# =============================================================================
#         resp_at_freq = lsfm.resp_freq(stim, resp, para, lags, bfs[i])
#         best_lag = lsfm.at_freq_lag(resp_at_freq, filename=filename, plot=False, saveplot=True)
# =============================================================================
# =============================================================================
#         slope_lags = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, filename=filename, plot=False, saveplot=True)
#         
#         if df['Version'][df_loc] == 1:
#             slope_lags_window = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, window=(1250,10000), filename=filename, plot=False, saveplot=True)
#         elif df['Version'][df_loc] == 2:
#             slope_lags_window = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, window=(1250,10000), filename=filename, plot=False, saveplot=True)
#         
#         txt = filename+'_slope'
#         lsfm_slope.plot_slope_index(*lsfm_slope.slope_index(slope_lags, bfs[i]), txt, saveplot=True)
#         txt = filename+'_direction'
#         lsfm_slope.plot_slope_index(*lsfm_slope.direction_index(lsfm_slope.direction_map(slope_lags), bfs[i]), txt, saveplot=True)
#         
#         txt = filename+'_slope_window'
#         lsfm_slope.plot_slope_index(*lsfm_slope.slope_index(slope_range, bfs[i]), txt, saveplot=True)
#         txt = filename+'_direction_window'
#         lsfm_slope.plot_slope_index(*lsfm_slope.direction_index(lsfm_slope.direction_map(slope_range), bfs[i]), txt, saveplot=True)
# =============================================================================
        
        
# =============================================================================
#         savedata = []
#         fdir = df['path'][df_loc]
#         filename = df['filename'][df_loc]
#         version = df['Version'][df_loc]
#         if version == 1:
#             t = Tdms_V1()
#             t.loadtdms(fdir, protocol=0, load_sound=True, precise_timing=True)
#         if version == 2:
#             t = Tdms_V2()
#             t.loadtdms(fdir, protocol=0, load_sound=True)
#         
#         para = t.Para
#         resp = np.array(t.Rdpk)
#         sound = t.rawS
#         stim = t.Sound
#         
#         if version == 1:
#             p = lsfm_psth.Psth(resp, para, filename)
#         elif version == 2:
#             p = lsfm_psth.Psth_New(resp, para, filename)
#             
#         lags = np.linspace(0, 50, 11)
#         slope_lags = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, window=(2000,10000), filename=filename, plot=True, saveplot=False)        
# =============================================================================
        
        
                
        
# =============================================================================
#         #p.psth_window((27500,30000), 'offset', tuning=None, saveplot=True, savenotes=True)
#         test = p.psth_trend(saveplot=False, window=(3000,6000))
#         
#         df2 = pd.DataFrame(columns = ['bd', 'cf', 'y', 'err'])
#         for i in range(len(test)):
#             _df = pd.DataFrame(test[i], columns = ['bd', 'cf', 'y', 'err'])
#             df2 = pd.concat([df2, _df])      
#         
#         df2.to_csv(f'{filename}_onset.csv', index=False)
#         
#         test = p.psth_trend(saveplot=False, window=(17500,22500))
#         
#         df3 = pd.DataFrame(columns = ['bd', 'cf', 'y', 'err'])
#         for i in range(len(test)):
#             _df = pd.DataFrame(test[i], columns = ['bd', 'cf', 'y', 'err'])
#             df3 = pd.concat([df3, _df])      
#         
#         df3.to_csv(f'{filename}_sustain.csv', index=False)
#         
#         test = p.psth_trend(saveplot=False, window=(27500,30000))
#         
#         df4 = pd.DataFrame(columns = ['bd', 'cf', 'y', 'err'])
#         for i in range(len(test)):
#             _df = pd.DataFrame(test[i], columns = ['bd', 'cf', 'y', 'err'])
#             df4 = pd.concat([df4, _df])      
#         
#         df4.to_csv(f'{filename}_offset.csv', index=False)
# =============================================================================
        

    
        
# =============================================================================
#         p = lsfm.Psth(resp, para, filename)
#         _ = p.psth_para(plot=False)
#         p.psth_trend(window=(3000,5000))
#         p.psth_all()
#         p.psth_window(((3000,5000)), 'onset')
# =============================================================================
        
# =============================================================================
#         p.psth_window((1250,2750), 'inhibit', saveplot=True, savenotes=False)
#         p.psth_window((5000,10000), 'onset', saveplot=True, savenotes=False)
#         p.psth_window((27500,37500), 'sustain', saveplot=True, savenotes=False)
#         p.psth_window((40000,45000), 'offset', saveplot=True, savenotes=True)
# =============================================================================
        
                                         
            
    
                    
# =============================================================================
#     psth_para.to_csv('PSTH_parameters.csv')
#     types = ['two peaks', 'two peaks', 'two peaks', 'no response', 'two peaks', 'two peaks', 
#              'no response', 'plateau', 'plateau', 'no response', 'plateau', 'no response']    
# =============================================================================
        
    
           
           
# =============================================================================
#         """response at target frequency"""
#         cwt = scipy.io.loadmat(r'E:\In-Vivo_Patch_Results\FIR\cwt_fir_real.mat')
#         #cwt = scipy.io.loadmat(r'R:\In-Vivo_Patch_Results\FIR\cwt_fir_real.mat')
#         atf = lsfm.RespAtFreq()
#         atf.mod_reduction(stim, resp, para, df, df_loc, cwt)
#         atf.resp_at_freq(nth_freq=True, plot=False)
# =============================================================================
    

    
    
    
# =============================================================================
#     resp_r = signal.resample(resp, 500, axis=1)
#     #resp_z = stats.zscore(resp_r)
#     f = cwt['f']
#     f = f[:,0]
#     wt = cwt['wt'].T[:,0]
#     wt_a = []
#     for w in wt:
#         wt_a.append(w)
#     wt_a = np.array(wt_a)
#     
#     _, _, mod, _ = zip(*para)
#     #use mod_rate at 1.0, 2.0, 8.0, 16.0 to avoid response contamination
#     slow = [i for i, a in enumerate(mod) if a >=1.0 and a <= 16.0]
#     para_s, wt_s, resp_s, stim_s = [],[],[],[]
#     for i in slow:
#         para_s.append(para[i])
#         wt_s.append(wt[i])
#         resp_s.append(resp[i])
#         stim_s.append(stim[i])
#     
#     resp_s = np.array(resp_s)
#     
#     
#     #lsfm.nth_resp(resp,para,cwt)
#     
#     fs = 200000
#     
#     inst_freqs = []
#     b,a = signal.butter(2, 6000, btype='low', fs=fs)
#     for stim in stim_s:
#         h = signal.hilbert(stim)
#         phase = np.unwrap(np.angle(h))
#         insfreq = np.diff(phase) / (2*np.pi) * fs
#         insfreq = signal.filtfilt(b,a,insfreq)
#         inst_freqs.append(np.diff(insfreq))
# =============================================================================
        


# =============================================================================
#     with open('FIR_07_27_2021.txt', 'r') as file:
#         fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#     sound, _ = t.get_raw()
#     sound_re = lsfm.inv_fir(sound, fir)
#     sound_re = t.cut(sound_re)
#     scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
#     
#     cwt = scipy.io.loadmat('0730_fir_cwt.mat')
# =============================================================================





