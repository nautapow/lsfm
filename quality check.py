import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy import interpolate
import scipy.io
import TFTool
import pandas as pd
import lsfm
from tone import PureTone
from lsfm_psth import Psth


df = pd.read_excel('matching_neurons.xlsx')
checking = df[df['flagged']=='yes']


for idx in checking.index[6:7]:
    mouseID = checking.loc[idx].tone_ID
    filename = checking.loc[idx].tone_filename
    cell_data = np.load(f'{filename}.npy', allow_pickle=True)
    
    resp = cell_data.item()['resp']
    para = cell_data.item()['para']
    tone = PureTone(resp, para, mouseID, filename)
    tone.get_bf()
    tone.get_bandwidth()
    tone.get_resp_wwobfband(use_band=False)
    tone_resp_in = tone.bfband['resp_in']
    tone_para_in = tone.bfband['para_in']
    for r,p in zip(tone_resp_in, tone_para_in):
        if p[0]>50 and p[0] <=80:
            fig, ax = plt.subplots()
            ax.plot(r)
            [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [500,3000]]
            ax.set_title(f'{mouseID}-{filename}: {p}')   
            plt.show()
    
    if checking.loc[idx].lsfm_actual is np.nan:
        lsfm_actual = lsfm_filename = checking.loc[idx].lsfm_filename
    else:
        lsfm_filename = checking.loc[idx].lsfm_filename
        lsfm_actual = checking.loc[idx].lsfm_actual
    
    cell_data = np.load(f'{lsfm_actual}_lsfm.npy', allow_pickle=True)
    version = 3
    para = cell_data.item()['para']
    stim = cell_data.item()['stim']
    resp = cell_data.item()['resp']
    titlename = f'{mouseID}-{lsfm_filename}'
    _, _, lsfm_resp_in, lsfm_resp_ex, lsfm_para_in, lsfm_para_ex, _, _ = lsfm.resp_bf_or_not(stim, resp, para, tone.bf)
    for r,p in zip(lsfm_resp_in, lsfm_para_in):
            fig, ax = plt.subplots()
            ax.plot(r)
            [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,26250]]
            ax.set_title(f'{mouseID}-{lsfm_filename}: {p[:3]}')   
            plt.show()