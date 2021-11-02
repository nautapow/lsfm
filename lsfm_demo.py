from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io
from ssqueezepy import ssq_cwt, ssq_stft
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import cwt, ssq_cwt, ssq_stft
import TFTool
import pandas as pd

wdir = os.getcwd()
mdir = Path('/Volumes/bcm-pedi-main-neuro-mcginley/Users/cwchiang/in_vivo_patch/')
os.chdir(mdir)
df = pd.read_csv('patch_list.csv', dtype = {'date':str, '#':str})
df_tdms = df.loc[df['type'] == 'Log sFM'].reset_index()
#os.chdir(wdir)
if 'CWT' not in df:
    df['CWT'] = 'no'

for i in range(len(df_tdms)):
    t = Tdms(str(Path(df_tdms.iloc[i]['path'])))
    t.loadtdms()
    sound = np.array(t.get_sound(), dtype=object)
    name = Path(str(mdir)+'/for_cwt/'+df_tdms.iloc[i]['date']+'_'+\
                df_tdms.iloc[i]['#'])
    TFTool.mat_out(name, sound)
    df['CWT'].iloc[df_tdms['index'].iloc[i]] = 'yes'
    
    #t = Tdms(Path(path))
    #t.loadtdms()
    


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