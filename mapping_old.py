from skimage import io
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import time
from nptdms import TdmsFile
from pathlib import Path
from joblib import Parallel, delayed


def map_17(directory):
    """Mapping 1.7"""
    imgs = sorted(glob.glob(os.path.join(directory, '*.tif')), key = lambda x: int(x.split('.')[-2].split('_')[-1]))
    category = [p.split('\\')[6].split('_')[2] for p in imgs]
    #remove "k" to sort by frequency
    category = sorted(list(set(category)), key=lambda x:x[:-2])
    
    para_all={}
    act_all={}
    session_all={}
    mapping_all={}
    
    for c in category:
        img_cate = [img for img in imgs if c in img]       
# =============================================================================
#         img_all = np.empty((1,512,512)).astype(np.int16)
#         for img in img_cate:
#             _img = io.imread(img).astype(np.int16)
#             img_all = np.concatenate((img_all, _img))
#         img_all = img_all[1:]
# =============================================================================
        
        img_all = load_image(img_cate)
        
        tdms_dir = glob.glob(os.path.join(directory, f'*{c}*[!Sound].tdms'))[0]
        filename = ('_').join(tdms_dir.split('\\')[-1].split('_')[:3])
        
# =============================================================================
#         tdms_file = TdmsFile.open(tdms_dir)
#         _groups = tdms_file['Untitled']
#         stim_onset = _groups['StimStart'][:]
#         sync = _groups['FVAL'][:]
#         para = _groups['Tone Parameters'][:]
#     
#         freq = [i for i in para[0::2] if i != 0]
#         loud = [i for i in para[1::2] if i != 0]
#         para = list(zip(loud, freq))
#         
#         from collections import Counter
#         repeat = Counter(para)[para[0]]
#     
#         timing = np.diff(np.sign(stim_onset-2.5))>0
#         stim_sync = [i for i,a in enumerate(timing) if a]
#     
#         timing = np.diff(np.sign(sync-1.5))>0
#         img_sync = [i for i,a in enumerate(timing) if a]
#         del timing
# =============================================================================
        para, stim_sync, img_sync = load_tdms(tdms_dir)
        repeat = count_repeat(para)
        check_sync(stim_sync, img_sync, img_all, para, filename)
        
# =============================================================================
#         sync_n = len(img_sync) - len(img_all)
#     
#         if len(stim_sync) == len(para) and sync_n == 0:
#             print(f'{filename}: All sync pulse matched')
#             del stim_onset
#             del sync
#         elif sync_n > 0 and sync_n < 200:
#             print(f'{filename}: Deleted extra {sync_n} TTL from cam')
#             for i in range(sync_n):
#                 img_sync.pop()
#             del stim_onset
#             del sync
#         elif sync_n < 0 and sync_n > -200:
#             print(f'{filename}: frames missing {sync_n} TTL')
#             del stim_onset
#             del sync
#         else:
#             print(f'{filename}: {sync_n} difference between #TTL and #frame')
#             del stim_onset
#             del sync
#             continue
# =============================================================================
        
        activities_cate = []
        around_stim_cate=[]
        for stim_idx, stim_time in enumerate(stim_sync):
            _closest_frame = (np.array(img_sync) - stim_time) > 0
            img_idx = next(i for i, j in enumerate(_closest_frame) if j)
            
            """averaging 10 frame pre-stimulus and 10 frame post-stimulus for comparison"""
            pre_stim = img_all[img_idx-11:img_idx-1]
            post_stim = img_all[img_idx:img_idx+10]
            
            """crop with a window of 60 frames around stimulus start"""
            window=60
            around_stim_cate.append(img_all[img_idx-(window//2):img_idx+(window//2)])
            
            diff = (np.mean(post_stim, axis=0) - np.mean(pre_stim, axis=0))
            activities_cate.append(diff)
            
        para_argsort = [i[0] for i in sorted(enumerate(para), key=lambda x:x[1])]
        para_sort = np.array(para)[para_argsort]
        para_sort = np.mean(np.reshape(para_sort, (-1, repeat, 2)), axis=1)
        act_sort = np.array(activities_cate)[para_argsort]
        act_sort = np.reshape(act_sort, (-1, repeat, 512, 512))
        around_stim_cate = np.array(around_stim_cate)[para_argsort]
        session_cate = np.reshape(around_stim_cate, (-1, repeat, window, 512, 512))
        mapping_cate = np.mean(act_sort, axis=1)
        
        para_all[c] = para
        act_all[c] = activities_cate

    para = np.array([v for values in para_all.values() for v in values])
    activities = np.array([v for values in act_all.values() for v in values])
    para_argsort = [i[0] for i in sorted(enumerate(para), key=lambda x:x[:][1][0])]
    para_sort = np.array(para)[para_argsort]
    para_sort = np.mean(np.reshape(para_sort, (-1, repeat, 2)), axis=1)
    act_sort = np.array(activities)[para_argsort]
    act_sort = np.reshape(act_sort, (-1, repeat, 512, 512))
    mapping = np.mean(act_sort, axis=1)
    
    return para, activities
        
# =============================================================================
#         activities_cate = []
#         around_stim_cate=[]
#         for stim_idx, stim_time in enumerate(stim_sync):
#             _closest_frame = (np.array(img_sync) - stim_time) > 0
#             img_idx = next(i for i, j in enumerate(_closest_frame) if j)
#             
#             """averaging 10 frame pre-stimulus and 10 frame post-stimulus for comparison"""
#             pre_stim = img_all[img_idx-11:img_idx-1]
#             post_stim = img_all[img_idx:img_idx+10]
#             
#             """crop with a window of 60 frames around stimulus start"""
#             window=60
#             around_stim_cate.append(img_all[img_idx-(window//2):img_idx+(window//2)])
#             
#             diff = (np.mean(post_stim, axis=0) - np.mean(pre_stim, axis=0))
#             activities_cate.append(diff)
#             
#         para_argsort = [i[0] for i in sorted(enumerate(para), key=lambda x:x[1])]
#         para_sort = np.array(para)[para_argsort]
#         para_sort = np.mean(np.reshape(para_sort, (-1, repeat, 2)), axis=1)
#         act_sort = np.array(activities_cate)[para_argsort]
#         act_sort = np.reshape(act_sort, (-1, repeat, 512, 512))
#         around_stim_cate = np.array(around_stim_cate)[para_argsort]
#         session_cate = np.reshape(around_stim_cate, (-1, repeat, window, 512, 512))
#         mapping_cate = np.mean(act_sort, axis=1)
#         
#         para_all[c] = para_sort
#         act_all[c] = act_sort
#         session_all[c] = session_cate
#         mapping_all[c] = mapping_cate
#         
#     mapping = np.array([v for values in mapping_all.values() for v in values])    
#     para = np.array([v for values in para_all.values() for v in values])
#     session = np.array([v for values in session_all.values() for v in values])
#     activities = np.array([v for values in act_all.values() for v in values])
# =============================================================================
    
    
    return activities, mapping, para, session


def map_18(directory):
    """Mapping 1.8"""
    imgs = sorted(glob.glob(os.path.join(directory, '*.tif')), key = lambda x: int(x.split('.')[-2].split('_')[-1]))
       
    load_image(imgs)
# =============================================================================
#     img_all = np.empty((1,512,512)).astype(np.int16)
#     for img in imgs:
#         _img = io.imread(img).astype(np.int16)
#         img_all = np.concatenate((img_all, _img))
#     t2 = time.time()
#     img_all = img_all[1:]
# =============================================================================
    
    tdms_dir = glob.glob(os.path.join(directory, '*[!Sound].tdms'))
    filename = ('_').join(tdms_dir.split('\\')[-1].split('_')[:3])
    
# =============================================================================
#     tdms_file = TdmsFile.open(tdms_dir)
#     _groups = tdms_file['Untitled']
#     stim_onset = _groups['StimStart'][:]
#     sync = _groups['FVAL'][:]
#     para = _groups['Tone Parameters'][:]
#     
#     freq = [i for i in para[0::2] if i != 0]
#     loud = [i for i in para[1::2] if i != 0]
#     para = list(zip(loud, freq))
#     
#     timing = np.diff(np.sign(stim_onset-2.5))>0
#     stim_sync = [i for i,a in enumerate(timing) if a]
#     
#     timing = np.diff(np.sign(sync-1.5))>0
#     img_sync = [i for i,a in enumerate(timing) if a]
#     del timing
# =============================================================================
    
    para, stim_sync, img_sync = load_tdms(tdms_dir)
    repeat = count_repeat(para)
    check_sync(stim_sync, img_sync, img_all, para, filename)
    
# =============================================================================
#     sync_n = len(img_sync) - len(img_all)
#     
#     if len(stim_sync) == len(para) and sync_n == 0:
#         print('All sync pulse matched')
#         del stim_onset
#         del sync
#     elif sync_n > 0 and sync_n < 10:
#         print('Deleted extra sync pulse from cam')
#         for i in range(len(sync_n)):
#             img_sync.pop()
#         del stim_onset
#         del sync
#     elif sync_n < 0 and sync_n > -200:
#         del stim_onset
#         del sync
#     else:
#         break
# =============================================================================
        
    activities = []
    around_stim=[]
    for stim_idx, stim_time in enumerate(stim_sync):
        _closest_frame = (np.array(img_sync) - stim_time) > 0
        img_idx = next(i for i, j in enumerate(_closest_frame) if j)
        
        """averaging 10 frame pre-stimulus and 10 frame post-stimulus for comparison"""
        pre_stim = img_all[img_idx-11:img_idx-1]
        post_stim = img_all[img_idx:img_idx+10]
        
        """crop with a window of 60 frames around stimulus start"""
        window=60
        around_stim.append(img_all[img_idx-(window//2):img_idx+(window//2)])
        
        diff = (np.mean(post_stim, axis=0) - np.mean(pre_stim, axis=0))
        activities.append(diff)
        
    para_argsort = [i[0] for i in sorted(enumerate(para), key=lambda x:x[1])]
    para_sort = np.array(para)[para_argsort]
    para_sort = np.mean(np.reshape(para_sort, (-1, repeat, 2)), axis=1)
    act_sort = np.array(activities)[para_argsort]
    act_sort = np.reshape(act_sort, (-1, repeat, 512, 512))
    around_stim = np.array(around_stim)[para_argsort]
    session = np.reshape(around_stim, (-1, repeat, window, 512, 512))
    mapping = np.mean(act_sort, axis=1)
    
    return act_sort, mapping, para_sort, session


def load_image(imgs):
    img_all = np.empty((1,512,512)).astype(np.int16)
    for img in imgs:
        _img = io.imread(img).astype(np.int16)
        img_all = np.concatenate((img_all, _img))
    img_all = img_all[1:]
    
    return img_all


def load_tdms(tdms_dir):
    tdms_file = TdmsFile.open(tdms_dir)
    _groups = tdms_file['Untitled']
    stim_onset = _groups['StimStart'][:]
    sync = _groups['FVAL'][:]
    para = _groups['Tone Parameters'][:]

    freq = [i for i in para[0::2] if i != 0]
    loud = [i for i in para[1::2] if i != 0]
    para = list(zip(loud, freq))
    
    from collections import Counter
    repeat = Counter(para)[para[0]]

    time_stim = np.diff(np.sign(stim_onset-2.5))>0
    stim_sync = [i for i,a in enumerate(time_stim) if a]
    
    time_ttl = np.diff(np.sign(sync-1.5))>0
    img_sync = [i for i,a in enumerate(time_ttl) if a]
    
    return para, stim_sync, img_sync


def count_repeat(para):
    from collections import Counter
    return Counter(para)[para[0]] 
    

def check_sync(stim_sync, img_sync, img_all, para, filename):
    sync_n = len(img_sync) - len(img_all)

    if len(stim_sync) == len(para) and sync_n == 0:
        print(f'{filename}: All sync pulse matched')
    elif abs(len(stim_sync)-len(para))>1:
        print(f'{filename}: stimulus out of sync')
    elif sync_n != 0:
        print(f'{filename}: {sync_n} difference between #TTL and #frame')


def plot_map(mapping):
    map_plot = np.reshape(mapping, (3,4,512,512))
    vmax = np.max(mapping)/2
    vmin = np.min(mapping)/2
    fig, ax = plt.subplots(3,4)
    for x in range(3):
        for y in range(4):
            ax[x,y].imshow(map_plot[x][y], vmin=vmin, vmax=vmax, aspect='auto')
    plt.show()
    plt.clf()


def check_pixel(session, window, repeat):
    def onclick(click):
        global coord
        x = click.xdata
        y = click.ydata
        print(click.inaxes)
        print(dir(click.inaxes))
        coord.append([x,y])
        plt.close()
        
    
    def act_pixel(session):
        %matplotlib tk
        fig, ax = plt.subplots(3,4)
        for x in range(3):
            for y in range(4):
                ax[x,y].imshow(map_plot[x][y], vmin=-0.5, vmax=0.5, aspect='auto')
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)      
        plt.show()
    
    global coord
    coord = []
    act_pixel(session)
    plt.waitforbuttonpress()
    plt.close()
    
    x = coord[0][2]
    y = coord[0][3]
    row = coord[0][0]
    col = coord[0][1]
    
    ss = np.moveaxis(session, (3,4), (0,1))
    ss = ss[int(x)][int(y)]
    ss = np.reshape(ss, (3,4,repeat,window))
    print(row, col)
    plt.imshow(ss[row-1][col-1])
    #plt.plot(np.mean(ss[11], axis=1))
        

if __name__ == "__main__":
    directory = r'Z:\Users\cwchiang\mapping\TP018\1'
    tdms_dir = glob.glob(os.path.join(directory, '*[!Sound].tdms'))[0]
    with TdmsFile.open(tdms_dir) as tdms_file:
        version = float(tdms_file['Settings'].properties['Software Version'])
    if version == 1.7:
        para, activities = map_17(directory)
    elif version == 1.8:
        mapping, activities, para, session = map_18(directory)
    
    
    
    
    