import os
from nptdms import TdmsFile
import pandas as pd
from pathlib import Path
import numpy as np

def asign_fir(df, date):
    date = int(date)
    _fdir = Path(str(mdir) + '/' + 'FIR_list')
    FIR_list = next(os.walk(_fdir))[2]
    fir_date = []
    for fir in FIR_list:
        fir_date.append(int(fir[4:12]))
    fir_date = np.array(fir_date)
    _fir_date = date - fir_date        
    
    return fir_date[_fir_date[_fir_date>0].argmin()]



mdir = Path(r'Z:\Users\cwchiang\in_vivo_patch')

folder = os.listdir(mdir)
folder.sort()

try:
    df = pd.read_csv('patch_list.csv')
except:
    df = pd.DataFrame(columns = ['date', '#', 'path', 'type', 'CWT', 'FIR'])  
    
    
frame = []

for i in range(len(folder)):
    if folder[i] not in list(df['date']):
        _fdir = Path(str(mdir) + '/' + folder[i])
        all_files = os.walk(_fdir)
        
        for _,_,files in all_files:
            for file in files:
                if file.endswith('.tdms') and file.find('Sound') == -1:
                    
                    path = str(_fdir) + '/' + file
                    tdms_meta = TdmsFile.read_metadata(Path(path))
                    rtype = tdms_meta['Settings'].\
                        properties['Sound Configuration.Run Config.Tone Type']
                    n = path.find('_00')
                    fdict = {'date' : folder[i], '#' : str(path[n+1:n+4]), 'path' : 
                             path, 'type' : rtype, 'CWT': 'no', 'FIR': asign_fir(df, folder[i])}
                    frame.append(fdict)
    else:
        continue

df = pd.DataFrame.from_dict(frame)
#df = df.append(frame, ignore_index = True)  
df.to_csv('patch_list_new.csv', index=False)


