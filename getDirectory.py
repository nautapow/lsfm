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



mdir = Path(r'E:\in-vivo_patch')

folder = os.listdir(mdir)
folder.sort()

try:
    df = pd.read_csv('patch_list_new.csv')
except:
    df = pd.DataFrame(columns = ['date', '#', 'filename','path'])  
    
    
frame = []
exclude = [0,1,2,3,4,5,6,8,9,12,13,15,16,17,18,20,21,22,32,33,38,39,40,41,42,
           43,45,47,48,50,52,53,54,55,56,57,58,59,64,65,66,67,68,69,70,79,88,89,
           109,121,134,135,136,146,166,171,174,176,177,181,182,187,188,217,229,233,
           239,242]

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
                    project_ver =  float(tdms_meta['Settings'].\
                        properties['Software Version'])
                    n = path.find('_0')
                    
                    if 'StimStart' in list(tdms_meta['Untitled']):
                        version = 2
                    else:
                        version = 1
                    
                    if project_ver >= 1.5:
                        version = 3
                        
                    try: 
                        p = int(folder[i])
                        if p >= 20230607:
                            project = 'Ic_map'
                        elif p < 20230428 and p >= 20230302:
                            project = 'Vc'
                        elif p < 20210611:
                            project = 'test'
                        else:
                            project = 'Ic'
                        
                    except:
                        pass
                            
                                            
                    fdict = {'date' : folder[i], '#' : str(path[n+1:n+4]), 'filename': str(f'{folder[i]}_{path[n+1:n+4]}'),
                             'path' : path, 'type' : rtype, 'CWT': 'no', 'FIR': asign_fir(df, folder[i]), 
                             'LabView ver' : project_ver, 'Py_version' : version, 'project' : project}
                    frame.append(fdict)
    else:
        continue

df = pd.DataFrame.from_dict(frame)
for i in exclude:
    df.loc[i, 'hard_exclude'] = 'exclude'
#df = df.append(frame, ignore_index = True)  
df.to_csv('patch_list_new.csv', index=False)


