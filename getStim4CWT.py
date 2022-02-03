from TDMS import Tdms
import numpy as np
import os
from pathlib import Path
import TFTool
import lsfm
import pandas as pd

mdir = r'Z:\Users\cwchiang\in_vivo_patch'
df = pd.read_csv('patch_list_Z.csv', dtype={'date':str, '#':str})
idx_lsfm = df.index[df['type']=='Log sFM']

for i in idx_lsfm:
    if df_tdms['CWT'].iloc[i] == 'no':
        t = Tdms()
        t.loadtdms(df['path'][i], protocol=0)
        sound,_ = t.get_raw()
        fir_dir = f'{mdir}/fir_list/{'df['fir'][i]'}.txt'
        with open(fir_dir, 'r') as file:
            fir = np.array(file.read().split('\n')[:-1], dtype='float64')
        
        sound = lsfm.inv_fir(sound, fir)
        sound = np.array(t.cut(sound), dtype=object)
        
        name = mdir+'/for_cwt/'+df['date'][i]+'_'+\
                    df['#'][i])
        TFTool.mat_out(name, sound)
        df['CWT'][i] = 'yes'
    else:
        continue
    
    #t = Tdms(Path(path))
    #t.loadtdms()
    
df.to_csv('patch_list.csv', index=False)
    


'''
t1 = Tdms('/Users/POW/Desktop/python_learning/20210812_003_2021_08_12_12_41_26.tdms')
t1.loadtdms()
stim, para = t1.get_stim()
resp = t1.get_resp()
sound = t1.get_sound()
'''

'''
Wx, scale, *_ = cwt(sound[245], fs=200000)
plt.imshow(np.abs(Wx), aspect='auto', cmap='jet')
plt.show()
'''    
