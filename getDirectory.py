import os
from nptdms import TdmsFile
import pandas as pd
from pathlib import Path

mdir = Path(r'Q:\[Project] 2020 in-vivo patch with behavior animal\Raw Results')
os.chdir(mdir)

folder = os.listdir(mdir)
folder.sort()

try:
    df = pd.read_csv('patch_list.csv')
except:
    df = pd.DataFrame(columns = ['date', '#', 'path', 'type'])  
    
    
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
                    
                    fdict = {'date' : folder[i], '#' : str(path[83:86]), 'path' : 
                             path, 'type' : rtype}
                    frame.append(fdict)
    else:
        continue

df = df.append(frame, ignore_index = True)  
#df.to_csv('new_patch_list.csv', index=False)

    