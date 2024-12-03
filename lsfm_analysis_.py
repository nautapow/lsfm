from TDMS_ver6 import Tdms_V1, Tdms_V2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy import interpolate
import scipy.io
import TFTool
import pandas as pd
import lsfm
from lsfm_psth import Psth
import lsfm_slope
import math
import lsfm_strf

    
if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[(df['type']=='Log sFM') & (df['project']!='Vc') & (df['hard_exclude']!='exclude')]
    exclude = [20,21,32,33,40,42,54,56,58,59,53,65,67,70]
    cell_note = pd.read_csv('cell_note_all.csv')
    
    
    idx_lsfm = [i for i in idx_lsfm if i > 146]
    
    df_loc = 142
    if df_loc == 142:
    #for df_loc in idx_lsfm:
        
        
        
        #i = int([i for i,a in enumerate(tlsfm) if a == df_loc][0])
        filename = df['filename'][df_loc]
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
        resp = TFTool.prefilter(resp, 25000)
        
        psth = Psth(resp, para, filename, version)
        x,y,err=psth.psth_all(plot=True)