from TDMS_ver5 import Tdms_V1, Tdms_V2
import numpy as np
import os
import pandas as pd
import puretone
import lsfm

if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[(df['type']=='Log sFM')&(df['hard_exclude']!='exclude')]
    idx_tone = df.index[(df['type']=='Pure Tones')&(df['hard_exclude']!='exclude')]
    exclude = [108]
    cell_note = pd.read_csv('cell_note_all.csv')
    #lsfm.resp_overcell(df, tlsfm, saveplot=False)
    #resp_cell = [[],[],[],[],[],[]]
    #band_cell_mean=[]
    #lsfm.best_lags()
    

    for df_loc in idx_lsfm:
        if df_loc not in exclude:
            """load from TDMS"""
            fdir = df['path'][df_loc]
            filename = df['filename'][df_loc]
            version = df['Py_version'][df_loc]
            
            if os.path.isfile(f'{filename}_lsfm.npy'):
                pass
            else:
                print(f'working on lsfm:index_{df_loc}-{filename}')
                try:
                    if version == 1:
                        t = Tdms_V1()
                        t.loadtdms(fdir, protocol=0, load_sound=True, precise_timing=True)
                    if version >= 2:
                        t = Tdms_V2()
                        t.loadtdms(fdir, protocol=0, load_sound=True)
                    
                    
                    stim = t.sound
                    resp = np.array(t.resp_dpk)
                    para = t.para
                    LabView_ver = t.version
                    stim_raw = t.stim_raw
                    
                    """reverse FIR"""
                    target_FIR = f'E:\in-vivo_patch\FIR_list\FIR_{df["FIR"][df_loc]}.txt'
                    
                    with open(target_FIR, 'r') as file:
                             fir = np.array(file.read().split('\n')[:-1], dtype='float64')
                    stim_fir = lsfm.inv_fir(stim_raw, fir)
                    stim_fir = t.cut(stim_fir)
                    
                    cell_data = {'stim_fir':stim_fir, 'stim':stim, 'resp':resp, 'para':para, 'Ver':LabView_ver}
                    np.save(f'{filename}_lsfm', cell_data)
                except:
                    print(f'unable to save {filename}')
          
    
    for df_loc in idx_tone:
        if df_loc not in exclude:
            fdir = df['path'][df_loc]
            filename = df['filename'][df_loc]
            version = df['Py_version'][df_loc]
            
            if os.path.isfile(f'{filename}.npy'):
                pass
            else:
                print(f'working on tone:index_{df_loc}-{filename}')
                try:
                    if version == 1:
                        t = Tdms_V1()
                        t.loadtdms(fdir, protocol=1, load_sound=True, precise_timing=True)
                    if version >= 2:
                        t = Tdms_V2()
                        t.loadtdms(fdir, protocol=1, load_sound=True)
                    
                    
                    #sound = t.stim_raw
                    
                    stim = t.sound
                    resp = np.array(t.resp_dpk)
                    para = t.para
                    LabView_ver = t.version
                    loud, freq = zip(*para)
                    
# =============================================================================
#                     if LabView_ver == 1.5:
#                         resp_merge, para_merge = puretone.resp_merge(resp, para)
#                         loud, freq = zip(*para_merge)
#                     else:
#                         resp_merge = resp
#                         loud, freq, _ = zip(*para)
# =============================================================================
                        
                    data={'stim':stim, 'resp':resp, 'para':para, 'loud':loud, 'freq':freq, 'Ver':LabView_ver}
                    np.save(filename, data)
                
                except:
                    name = df.iloc[df_loc]['filename']
                    print(f'unable to save {name}')

        
# =============================================================================
#         """reverse FIR"""
#         target_FIR = f'E:\in-vivo_patch\FIR_list\FIR_{df["FIR"][df_loc]}.txt'
#         
#         with open(target_FIR, 'r') as file:
#                  fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#         sound_re = lsfm.inv_fir(sound, fir)
#         sound_re = t.cut(sound_re)
#         scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
# =============================================================================
    
    
