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
    idx_lsfm = df.index[df['type']=='Log sFM']
    tlsfm = [23,24,25,28,30,35,37,45,49,60,62,71,73,74,76,78,81,82]
    #tlsfm = [37, 60]
    cell_note = pd.read_csv('cell_note_all.csv')
    #lsfm.resp_overcell(df, tlsfm, saveplot=False)
    #resp_cell = [[],[],[],[],[],[]]
    #band_cell_mean=[]
    #lsfm.best_lags()
    
# =============================================================================
#     bf = cell_note['best frequency']
#     lag_all = cell_note['best_lag_all']
#     lag_first = cell_note['best_lag_bf']
#     bf_scale = [i/1000 for i in bf]
#     
#     fig, ax1 = plt.subplots(figsize=(4,6))
#     im = ax1.scatter(np.zeros(len(lag_all)), lag_all, zorder=10, c=bf_scale, cmap='plasma', s=100)
#     im = ax1.scatter(np.ones(len(lag_first)), lag_first, zorder=10, c=bf_scale, cmap='plasma', s=100)
#     cbar = plt.colorbar(im)
#     cbar.ax.set_ylabel('best frequency (kHz)', fontsize=16)
#     cbar.ax.tick_params(axis='y', which='major', labelsize=14)
#     for i in range(len(lag_all)):
#         ax1.plot([0,1], [lag_all[i], lag_first[i]], c='k', zorder=0)
#     plt.locator_params(axis='x', nbins=2)    
#     ax1.set_xlim(-0.5,1.5)
#     ax1.set_ylabel('lags (ms)', fontsize=16)
#     ax1.set_xticks([0,1])
#     ax1.set_xticklabels(['All Cross', 'First Cross'])
#     ax1.tick_params(axis='both', which='major', labelsize=14)
# =============================================================================

    
    #df_loc = 76
    #if df_loc == 76:
    for df_loc in tlsfm:
        
        
        
        i = int([i for i,a in enumerate(tlsfm) if a == df_loc][0])
        filename = df['filename'][df_loc]
        version = df['Version'][df_loc]
        cell_data = np.load(f'{filename}_data.npy', allow_pickle=True)
        
        para = cell_data.item().get('para')
        stim = cell_data.item().get('stim')
        resp = cell_data.item().get('resp')
        slope_lags = cell_data.item().get('slope_lags')
        
        cf,band,modrate,_=zip(*para)
        band = sorted(set(band))
        cf = sorted(set(cf))
        modrate = sorted(set(modrate))
        
        n = cell_note.index[cell_note['filename']==filename][0]
        bf = cell_note['best frequency'].loc[n]
        features = cell_note['feature'].loc[n]
        windows = cell_note['window'].loc[n].split(', ')
        
        """0: onset, 1:sustain, 2:offset"""
        window = eval(windows[2])
        #resp_at_freq_cell = np.load('restrain_resp_at_freq_cell.npy', allow_pickle=True)
        #test = lsfm.nXing_cell(resp_at_freq_cell)
        tune = (round(bf/2/1000,1), round(bf*2/1000,1))
        
        import mat73
        if version == 1:
            cwt = mat73.loadmat(r'C:\Users\McGinley3\Documents\GitHub\lsfm\20210730_002_cwt_sound.mat')
        elif version == 2:
            cwt = mat73.loadmat(r'C:\Users\McGinley3\Documents\GitHub\lsfm\20220216_001_cwt_sound.mat')
        s = lsfm_strf.STRF(cwt, resp, filename)
        #strf = s.strf(saveplot=False)
        #resp_simus = s.resp_simu(strf)
        strf_fake, rf_para = s.fake_strf(saveplot=False)
        resp_simus_fake = s.resp_simu(strf_fake)
        
        filename_real = f'{filename}_real'
        filename_fake = f'{filename}_fake'
         
        """PSTH"""
        p = lsfm_psth.Psth(resp_simus_fake, para, filename_real, version=version)
        _,_,_ = p.psth_all(plot=True, saveplot=False)
        #lsfm_psth.psth_wwo_bf(resp, para, bf, version, filename, saveplot=True)
        #p.psth_trend(tuning=tune, plot=True, saveplot=False)
        #p.psth_para(plot=True, saveplot=False)

        
# =============================================================================
#         """load from TDMS"""
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
#         
#         sound = t.stim_raw
# 
#         """reverse FIR"""
#         target_FIR = f'E:\in-vivo_patch\FIR_list\FIR_{df["FIR"][df_loc]}.txt'
#         
#         with open(target_FIR, 'r') as file:
#                  fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#         sound_re = lsfm.inv_fir(sound, fir)
#         sound_re = t.cut(sound_re)
#         scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
# =============================================================================
        

# =============================================================================
#         """plot SD-slope and SD-direction"""
#         m, m_bf = lsfm_slope.slope_index(slope_lags, bf)
#         d, d_bf = lsfm_slope.direction_index(lsfm_slope.direction_map(slope_lags), bf)
#         lsfm_slope.plot_both_index(m, m_bf, d, d_bf, filename, plot=True, saveplot=True)
#         #lsfm_slope.plot_both_index(*lsfm_slope.slope_index(slope_lags, bf), *lsfm_slope.direction_index(lsfm_slope.direction_map(slope_lags), bf), filename)
# =============================================================================
        
        """slope"""
        lags = np.linspace(0, 100, 51)
        #slope_lags = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, filename=filename, plot=False, saveplot=True)
        _ = lsfm_slope.freq_slope_contour(stim, resp_simus_fake, para, lags=lags, filename=f'{filename}_{rf_para}', plot=False, saveplot=True)

        #direction_lag = lsfm_slope.direction_map(slope_lags)
        #lsfm_slope.direction_contour(direction_lag, filename, plot=False, saveplot=True)
        
        #slope_lags = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, filename=filename, plot=True, saveplot=False)

    
# =============================================================================
#         for i,p in enumerate(para):
#             if i in [63,64,328]:
#                 lsfm.stim_resp(i, stim[i], resp[i], p[:3], filename, saveplot=True)
# =============================================================================
        

            
# =============================================================================
#         lags = np.linspace(0, 100, 51)       
#
#         
#         resp_at_freq = lsfm.resp_freq(stim, resp, para, lags, bf)
#         _,_, best_lag = lsfm.at_freq_lag(resp_at_freq, filename=filename, plot=True, saveplot=True)
# =============================================================================



        

        
        
# =============================================================================
#         cell_data = {'stim':stim, 'resp':resp, 'para':para, 'resp_by_para':resp_by_para, 'slope_lags':slope_lags}
#         np.save(f'{filename}_data.npy', cell_data)
# =============================================================================
        

        
# =============================================================================
#         resp_at_freq = lsfm.resp_freq(stim, resp, para, lags, bf)
#         _ = lsfm.at_freq_lag(resp_at_freq, filename=filename, plot=True, saveplot=True)
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
#         lags = np.linspace(0, 50, 11)
#         slope_lags = lsfm_slope.freq_slope_contour(stim, resp, para, lags=lags, window=(2000,10000), filename=filename, plot=True, saveplot=False)        
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
# 
#     with open('FIR_07_27_2021.txt', 'r') as file:
#         fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#     sound, _ = t.get_raw()
#     sound_re = lsfm.inv_fir(sound, fir)
#     sound_re = t.cut(sound_re)
#     scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
# =============================================================================
    
# =============================================================================
#     """plot stimulus power spectrum"""
#     cwt = scipy.io.loadmat('20220216_afterCWT')
# 
#     f = cwt['f']
#     f = f[:,0]
#     wt = cwt['wt'].T[:,0]
#     wt_a = []
#     
#     for w in wt:
#         wt_a.append(w)
#     
#     wt_a = np.array(wt_a)
#     ylabel = [round((i),2) for i in f]
#     
#     
#     x = np.arange(500)
#     plt.pcolormesh(x, f, wt_a[101], cmap='hot_r', vmax=0.05, vmin=0)
#     plt.yscale('log')
#     plt.xlim(0,375)
#     plt.xlabel('time (sec)', fontsize=18)
#     plt.ylabel('frequency (kHz)', fontsize=18)
#     plt.xticks([0,125,250,375], labels=[0,0.5,1,1.5], rotation=45, fontsize=16)
#     plt.ylim(1000,100000)
#     plt.yticks([3000,6000,12000,24000,48000], [3,6,12,24,48], fontsize=16)
#     plt.title('Center Frequency: 12 kHz, bandwidth: 3 octaves, modulation rate: 8 Hz')
#     plt.savefig('sample_lsfm_spectrum.pdf', dpi=500, format='pdf', bbox_inches='tight')
# =============================================================================
    
    
