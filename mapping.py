from skimage import io
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import time
from nptdms import TdmsFile
from pathlib import Path
from joblib import Parallel, delayed
from numba import jit, njit, cuda
from PIL import Image, ImageSequence

def map_17(directory):
    """Mapping 1.7"""
    def im_category(c, imgs):
        img_cate = [img for img in imgs if c in img]
        img_all = load_image(img_cate)
        
        tdms_dir = glob.glob(os.path.join(directory, f'*{c}*[!Sound].tdms'))[0]
        filename = ('_').join(tdms_dir.split('\\')[-1].split('_')[:3])
        
        para_cate, stim_sync, img_sync = load_tdms(tdms_dir)
        check_sync(stim_sync, img_sync, img_all, para_cate, filename)
        
        activities_cate, around_stim_cate = get_stim_resp(img_all, stim_sync, img_sync)
        
        para_all[c] = para_cate
        act_all[c] = activities_cate
        around_stim_all[c] = around_stim_cate
        
        return para_all, act_all, around_stim_all
    
    imgs = sorted(glob.glob(os.path.join(directory, '*.tif')), key = lambda x: int(x.split('.')[-2].split('_')[-1]))
    category = [p.split('\\')[6].split('_')[2] for p in imgs]
    #remove "k" to sort by frequency
    category = sorted(list(set(category)), key=lambda x:x[:-2])
    
    para_all={}
    act_all={}
    around_stim_all={}
    t1 = time.time()
    
    for c in category:
        img_cate = [img for img in imgs if c in img]          
        img_all = load_image(img_cate)
        
        tdms_dir = glob.glob(os.path.join(directory, f'*{c}*[!Sound].tdms'))[0]
        filename = ('_').join(tdms_dir.split('\\')[-1].split('_')[:3])
        
        para_cate, stim_sync, img_sync = load_tdms(tdms_dir)
        check_sync(stim_sync, img_sync, img_all, para_cate, filename)
        
        activities_cate, around_stim_cate = get_stim_resp(img_all, stim_sync, img_sync)
        
        para_all[c] = para_cate
        act_all[c] = activities_cate
        around_stim_all[c] = around_stim_cate

    #combine through all frequencies in dictionary   
    para = np.array([v for values in para_all.values() for v in values])
    activities = np.array([v for values in act_all.values() for v in values])
    around_stim = np.array([v for values in around_stim_all.values() for v in values])
    
    para_argsort = [i[0] for i in sorted(enumerate(para), key=lambda x:x[:][1][0])]
    para_sort = para[para_argsort]
    act_sort = activities[para_argsort]
    around_stim_sort = around_stim[para_argsort]
    t2=time.time()
    print(t2-t1)
    
    return para, para_sort, act_sort, around_stim_sort


def map_18(directory):
    """Mapping 1.8"""
    imgs = sorted(glob.glob(os.path.join(directory, '*.tif')), key = lambda x: int(x.split('.')[-2].split('_')[-1]))      
    img_all = load_image(imgs)
    
    tdms_dir = glob.glob(os.path.join(directory, '*[!Sound].tdms'))[0]
    filename = ('_').join(tdms_dir.split('\\')[-1].split('_')[:3])
    
    para, stim_sync, img_sync = load_tdms(tdms_dir)
    check_sync(stim_sync, img_sync, img_all, para, filename)
    activities, around_stim = get_stim_resp(img_all, stim_sync, img_sync)
        
    para_argsort = [i[0] for i in sorted(enumerate(para), key=lambda x:x[1])]
    para_sort = np.array(para)[para_argsort]
    act_sort = np.array(activities)[para_argsort]
    around_stim_sort = np.array(around_stim)[para_argsort]
    
    return para, para_sort, act_sort, around_stim_sort


def load_image_old(imgs):
    img_all = np.empty((1,512,512)).astype(np.int16)
    for img in imgs:
        _img = io.imread(img).astype(np.int16)
        img_all = np.concatenate((img_all, _img))
    img_all = img_all[1:]
    
    return img_all


def load_image(imgs):  
    #img_all = np.empty((len(imgs)*2048,512,512)).astype(np.int16)
    img_all = preallocate(len(imgs))

    idx = 0
    for img in imgs:
        im = ImageSequence.Iterator(Image.open(img))
        _im = 1
        while _im:
            _im = next(im, None)
            if _im != None:
                img_all[idx] = np.array(_im.rotate(-90), np.int16)
                idx+=1
    
    return img_all[:idx].astype(np.int16)

@jit(nopython=True, parallel=True)
def preallocate(n_imgs):
    return np.zeros((n_imgs*2048,512,512), np.int16)


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


def get_stim_resp(img_all, stim_sync, img_sync):
    activities = []
    around_stim=[]
    for stim_idx, stim_time in enumerate(stim_sync):
        _closest_frame = (np.array(img_sync) - stim_time) > 0
        img_idx = next(i for i, j in enumerate(_closest_frame) if j)
        
        """averaging 10 frame pre-stimulus and 10 frame post-stimulus for comparison"""
        pre_stim = img_all[img_idx-11:img_idx-1]
        post_stim = img_all[img_idx:img_idx+10]
        
        """crop with a window of 200 frames around stimulus start"""
        window=200
        around_stim.append(img_all[img_idx-(window//2):img_idx+(window//2)+1])
        
        diff = (np.mean(post_stim, axis=0) - np.mean(pre_stim, axis=0))
        activities.append(diff)
        
    return activities, around_stim


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


def plot_map(mapping, filename, savefig=False):
    map_plot = np.reshape(mapping, (3,4,512,512))
    vmax = np.max(mapping)/3
    vmin = np.min(mapping)/3
    
    fig, ax = plt.subplots(3,4, figsize=(15, 10))
    for x in range(3):
        for y in range(4):
            ax[x,y].imshow(map_plot[x][y], vmin=vmin, vmax=vmax, aspect='auto')
            ax[x,y].set_xticks([0,256,512])
            ax[x,y].set_yticks([0,256,512])
    cols = ['4k', '10k', '20k', '30k']
    rows = ['50dB', '60dB', '70dB']
    
    for axes, col in zip(ax[0], cols):
        axes.set_title(col, fontsize=16)
        
    for axes, row in zip(ax[:,0], rows):
        axes.set_ylabel(row, rotation=90, fontsize=16)
    
    
    _ = fig.suptitle(f'{filename}', y=0.96, fontsize=20)
    
    if savefig:
        plt.savefig(f'{filename}_mapping.png', dpi=500, bbox_inches='tight')
        plt.show()
        plt.clf()
    else:
        plt.show()
        plt.clf()


def get_xy(mapping, around_stim):
    def onclick(click):
        global coord, axes_click
        x = click.xdata
        y = click.ydata
        coord.append([x,y])
        if click.inaxes is not None:
            axes_click = click.inaxes
        
        plt.close()
            
    def plot2click(mapping):
        %matplotlib tk
        map_plot = np.reshape(mapping, (3,4,512,512))
        vmax = np.max(mapping)/2
        vmin = np.min(mapping)/2
        fig, ax = plt.subplots(3,4)
        for x in range(3):
            for y in range(4):
                ax[x,y].imshow(map_plot[x][y], vmin=vmin, vmax=vmax, aspect='auto')
                ax[x,y].set_title(f'({x},{y})')
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)      
        plt.show()
    
    global coord, axes_click
    coord = []
    axes_click = []
    plot2click(mapping)
    plt.waitforbuttonpress()
    plt.close()

    xx = int(coord[0][0])
    yy = int(coord[0][1])
    
    %matplotlib inline
    return xx, yy, axes_click


def check_xy(mapping, around_stim, para_map, filename, window = 60, saveplot=False):
    xx, yy, axes_click = get_xy(mapping, around_stim)
    axes_click = str(axes_click.title)
    subplot = int(axes_click[-6])*4+int(axes_click[-4])
    
    crop = np.swapaxes(around_stim, 0,1)
    center = len(crop)//2
    crop = np.swapaxes(crop[center-int(window/2):center+int(window/2)+1], 0,1)
    
    arst = np.reshape(crop, (12,-1,window+1,512,512))
    arst = np.moveaxis(arst, (3,4), (0,1))
    arst = arst[xx][yy]
    
    psth = np.mean(arst[subplot], axis=0)
    
    fig, [ax1,ax2] = plt.subplots(1,2,figsize = (10,12), sharex=True)
    ax1.imshow(arst[subplot], aspect='auto')
    ax1.set_yticks([0,4,9,14,19,24,29], [1,5,10,15,20,25,30])
    ax1.set_xlim(0,window)
    ax1.set_xticks([0,window//2,window], [-1*(window//2), 0, window//2])
    ax2.plot(psth)
    fig_name = f'{filename}-{para_map[subplot][0]}dB-{para_map[subplot][1]}kHz, ({xx}, {yy})'
    fig.suptitle(fig_name, y=0.92, fontsize=18)
    if saveplot:
        plt.savefig(f'{fig_name}.png', dpi=500, bbox_inches='tight')
        plt.show()
        plt.clf()
    else:
        plt.show()
        plt.clf()
    
# =============================================================================
#     for i in range(len(arst)):
#         psth = np.mean(arst[i], axis=0)
#         
#         fig, [ax1,ax2] = plt.subplots(1,2,figsize = (10,12), sharex=True)
#         ax1.imshow(arst[i], aspect='auto')
#         ax1.set_yticks([0,4,9,14,19,24,29], [1,5,10,15,20,25,30])
#         ax1.set_xlim(0,window)
#         ax1.set_xticks([0,window//2,window], [-1*(window//2), 0, window//2])
#         ax2.plot(psth)
#         fig_name = f'{filename}-{para_map[i][0]}dB-{para_map[i][1]}kHz, ({xx}, {yy})'
#         fig.suptitle(fig_name, y=0.92, fontsize=18)
#         if saveplot:
#             plt.savefig(f'{fig_name}.png', dpi=500, bbox_inches='tight')
#             plt.show()
#             plt.clf()
#         else:
#             plt.show()
#             plt.clf()
# =============================================================================
            
            

def check_window(around_stim, window, center=None):
    if not center:
        center = len(around_stim[0])//2
    
    frame_crop = np.swapaxes(around_stim, 0,1)
    frame_crop = np.swapaxes(frame_crop[center-int(window/2):center+int(window/2)+1], 0,1)
    act_window=[]
    for f in frame_crop:
        pre = f[:int(window/2)]
        post = f[int(window/2):-1]
        act = np.mean((post-pre), axis=0)
        act_window.append(act)
        
    return np.array(act_window)


if __name__ == "__main__":
    directory = r'Z:\Users\cwchiang\mapping\TP024\1'
    tdms_dir = glob.glob(os.path.join(directory, '*[!Sound].tdms'))[0]
    filename = ('_').join(tdms_dir.split('\\')[-1].split('_')[:2])
    
    with TdmsFile.open(tdms_dir) as tdms_file:
        version = float(tdms_file['Settings'].properties['Software Version'])
    if version == 1.7:
        para_all, para, act, around_stim = map_17(directory)
    elif version == 1.8:
        para_all, para, act, around_stim = map_18(directory)
    
    para_list = [tuple(i) for i in para]
    repeat = count_repeat(para_list)
    
    act_map = np.reshape(act, (-1, repeat, 512, 512))
    para_map = np.mean(np.reshape(para, (-1, repeat, 2)), axis=1)
    mapping = np.mean(act_map, axis=1)
    %matplotlib inline
    plot_map(mapping, filename, savefig=True)
