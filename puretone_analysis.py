from TDMS_ver2 import Tdms
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
    #tlsfm = [23,24,25,27,28,30,32,35,37,40,45,49]
    #psth_para = pd.DataFrame(columns = ['name','sum','max','min','average','zmax','zmin',
    #                                    'sum1','sum2','sum3','sum4', 'sum5']) 
    
    df_loc = 69
    if df_loc == 69:
    #for df_loc in tlsfm:
        fdir = df['path'][df_loc]
        filename = df['date'][df_loc]+'_'+df['#'][df_loc]
        t = Tdms()
        t.loadtdms(fdir, protocol=1, load_sound=True)
        
        para = t.Para
        resp = np.array(t.Rdpk)
        sound = t.rawS
        stim = t.Sound
        
        puretone.mem_V(stim, para, resp)
        