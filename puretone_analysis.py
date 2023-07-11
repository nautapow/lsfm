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
    idx_puretone = df.index[(df['type']=='Pure Tones') & (df['project']!='Vc') & (df['hard_exclude']!='exclude')]
    #idx_tone = [26,29,31,34,36,44,48,61,72,75,77,80,83,84,85,88,90,93,94,96,99]
    idx_tone = [26,29,31,34,36,44,61,72,75,77,80,83,84,85,88,90,93,96]
    exclude = [0,1,2,3,4,5,6,8,9,12,13,15,16,17,18,22,38,39,41,43,45,47,48,50,52,53,55,57,64,66,68,69,79]
    cell_note = pd.read_csv('cell_note_all.csv')
    idx_tone = [26,29]
    #frequency in octave difference
    frequency  = np.linspace(-5.,5.,num=301)
    #pre-create 2d array to store response of resp_mesh from all neurons
    resp_all_pos_2bf = np.array([[0.0]*len(frequency)]*301)
    widthAt70=[]
    bf_list = []    
    
    #df_loc = 147
    #if df_loc == 147:
    
    index = []
    for i in idx_puretone:
        if i not in exclude:
            index.append(i)
            
    
    for df_loc in index:
        #i = int([i for i,a in enumerate(idx_tone) if a == df_loc][0])
        filename = df['filename'][df_loc]
        version = df['Py_version'][df_loc]
        print(filename)
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
        
        
# =============================================================================
#         fdir = df['path'][df_loc]
#         filename = df['filename'][df_loc]
#         version = df['Version'][df_loc]
#         if version == 1:
#             t = Tdms_V1()
#             t.loadtdms(fdir, protocol=1, load_sound=True, precise_timing=True)
#         if version == 2 or version == 3:
#             t = Tdms_V2()
#             t.loadtdms(fdir, protocol=1, load_sound=True)
#             
# 
#         para = t.para
#         resp = np.array(t.resp_dpk)
#         stim = t.sound
#         LabView_ver = t.version
#         
#         if LabView_ver == 1.5:
#             resp_merge, para_merge = puretone.resp_merge(resp, para)
#             loud, freq = zip(*para_merge)
#         else:
#             resp_merge = resp
#             loud, freq, _ = zip(*para)
#             
#         data={'resp':resp_merge, 'stim':stim, 'para':para, 'loud':loud, 'freq':freq, 'Ver':LabView_ver}
#         np.save(filename, data)
# =============================================================================
        
        data = np.load(f'{filename}.npy', allow_pickle=True)
        resp = data.item()['resp']
        loud = sorted(set(data.item()['loud']))
        freq = sorted(set(data.item()['freq']))
        try:
            loud.remove(0.0)
            freq.remove(0.0)
        except:
            pass
                
        para = data.item()['para']
        
        resp_filt = TFTool.prefilter(resp, 25000)
        resp_adjust = [r - np.mean(r[400:500]) for r in resp_filt]
        
        x = puretone.tuning(resp_adjust, para, filename, plot=True, saveplot=False, data_return=True, window=(500,3000))
        bf70 = x[4][240]
        bf_list.append((df_loc, filename, bf70))
        df.loc[df_loc, 'best_frequency'] = bf70
    
    df.to_csv('patch_list_with_bf.csv', index=False)     
        #second clamp
# =============================================================================
#         filename2 = df['filename'][df_loc+1]
#         data2 = np.load(f'{filename2}.npy', allow_pickle=True)
#         resp2 = data2.item()['resp']
#         resp_filt2 = TFTool.prefilter(resp2, 25000)
#         resp_adjust2 = [r - np.mean(r[400:500]) for r in resp_filt2]
#         
#         for i in range(len(resp)):
#             plt.plot(resp_adjust[i])
#             plt.plot(resp_adjust2[i])
#             plt.show()
# =============================================================================
        
# =============================================================================
#         peak_x = puretone.psth(resp, filename)
#         if peak_x < 2000:
#             window_peak=[peak_x-250,peak_x+250]
#         else:
#             window_peak=[1250,1750]
#         
#         try:
#             puretone.tuning(resp, para, filename=f'{filename}_peak', saveplot=False, window=window_peak)
#             puretone.tuning(resp, para, filename=f'{filename}_sust', saveplot=False, window=[2500,3000])
#         except:
#             pass
# =============================================================================
        
# =============================================================================
#         def octave2bf(bf, freq):
#             oct_bf = []
#             for f in freq:
#                 oct_bf.append(math.log((f/bf),2))
#             
#             return oct_bf
#                 
#         def on_avg(arr, window):
#             base = np.mean(arr[:500])
#             arr = (arr-base)*100            
#             return np.mean(arr[window[0]:window[1]])   
#         
#         def set_hyper2zero(arr):
#             mask = arr < 0
#             import copy
#             arr_pos = copy.deepcopy(arr)
#             arr_pos[mask] = 0
#             
#             return arr_pos
#         
#         def set_hyper2nan(arr):
#             mask = arr < 0
#             import copy
#             arr_nan = copy.deepcopy(arr)
#             arr_nan[mask] = np.nan
#             
#             return arr_nan
#         
#         def min_index(arr, num):
#             arr = np.array(arr)
#             
#             return np.argmin(abs(arr - num))
#         
#         fs=25000
#         resp_filt = TFTool.prefilter(resp, fs)
#         resp_mesh = np.reshape(resp_filt, (len(loud),len(freq),-1))
#         
#         if 90 in loud:    
#             resp_mesh = np.delete(resp_mesh, -1, axis=0)
#             loud.pop()
#             
#         #window = [1250,1750]
#         window = [2500,3000]
#         resp_on = np.apply_along_axis(on_avg, 2, resp_mesh, window)
#         resp_pos = set_hyper2zero(resp_on)
#         
#         x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#         y300 = np.linspace(30,80,301)
#         
#         Nzero = int(300/(len(freq)-1))-1
#         zero2D = np.zeros((6,len(freq),Nzero))        
#         upsampleX = np.dstack((resp_on, zero2D)).reshape((6,-1))[:,:301]
#         
#         filt1D = scipy.ndimage.gaussian_filter1d(upsampleX, Nzero)
#         #test = np.apply_along_axis(TFTool.butter, 1, upsampleX, 1, 1/50, 'low', 1)
#         
# # =============================================================================
# #         interpX, interpXY=[],[]
# #         for loud_layer in resp_on:
# #             interpX.append(np.interp(x300, freq, loud_layer))
# #         interpX = np.array(interpX)
# #         resp_300 = np.swapaxes(interpX, 0, 1)
# # =============================================================================
#         
#         resp_300 = np.swapaxes(filt1D, 0, 1)        
#         
#         interpXY=[]
#         for freq_layer in resp_300:
#             interpXY.append(np.interp(y300, loud, freq_layer))
#         
#         interpXY = np.array(interpXY)
#         interpXY = np.swapaxes(interpXY, 0, 1)
#         
#         XX, YY = np.meshgrid(x300, y300)
#         plt.pcolormesh(XX, YY, interpXY, cmap='RdBu_r', norm=colors.CenteredNorm())
#         plt.xscale('log')
#         plt.colorbar()
# =============================================================================




# =============================================================================
#         """Get Tuning Width Among All Neurons"""
#         x300,y300,resp_smooth, width = puretone.tuning(resp, para, filename, plot=True, saveplot=True, data_return=True, window=(2500,3000))
#         puretone.psth(resp, filename)
#         resp_pos = puretone.set_hyper2zero(resp_smooth)
#         widthAt70.append(width)
#         bf_loud = []
#         for a in resp_pos:
#             bf_loud.append(puretone.center_mass_layer(a, x300))
#         
#         resp_oct2bf=[]
#         for i in range(len(y300)):        
#             resp_oct2bf.append(puretone.octave2bf(bf_loud[i], x300))
#             
#         #i for layering loudness, j for iterate through frequency
#         for i in range(len(y300)):
#             oct2bf = puretone.octave2bf(bf_loud[i], x300)
#             for j in range(len(x300)):
#                 index = puretone.min_index(frequency, oct2bf[j])
#                 base = resp_all_pos_2bf[i][index]
#                 arr = np.array([base, resp_smooth[i][j]])
#                 resp_all_pos_2bf[i][index] = np.mean(arr)
# =============================================================================
        
        
# =============================================================================
#         #resp_pas = TFTool.pascal_filter(resp_on)
#         resp_nan = set_hyper2nan(resp_filt)
#         resp_pos = set_hyper2zero(resp_on)
#         
#         
#         
#         bf_loud = []
#         for i,x in enumerate(resp_pos):
#             bf_loud.append(puretone.center_mass_layer(x, freq))
#         #resp_sum = np.sum(resp_pos, axis=1)    
#         
#         resp_oct2bf=[]
#         for i in range(len(loud)):        
#             resp_oct2bf.append(octave2bf(bf_loud[i], freq))
#             
#         resp_mesh = np.reshape(resp, (len(loud),len(freq),-1))
#         
#         #i for layering loudness, j for iterate through frequency
#         for i in range(len(loud)):
#             oct2bf = octave2bf(bf_loud[i], freq)
#             for j in range(len(freq)):
#                 index = min_index(frequency, oct2bf[j])
#                 base = resp_all_pos_2bf[i][index]
#                 arr = np.array([base, resp_pos[i][j]])
#                 resp_all_pos_2bf[i][index] = np.nanmean(arr)
# =============================================================================
                
# =============================================================================
#     loudness = np.arange(30,100,10)
#     XX,YY = np.meshgrid(frequency, y300)
#     resp2bf = np.array(resp_all_pos_2bf)
#     fig, ax1 = plt.subplots()
#     pcm = ax1.pcolormesh(XX,YY,resp2bf, vmax=0.5, vmin=0)
#     fig.colorbar(pcm, ax=ax1)
# =============================================================================
    
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
       
        
        

        
        
        