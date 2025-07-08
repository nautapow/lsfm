import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats, optimize
import TFTool
import pandas as pd
import lsfm


class Psth():
    def __init__(self, stim, resp, para, filename, version, bf, band_left, band_right, use_band=True):
        #exclude carrier less than 3kHz and puretone
        para_temp, resp_temp, stim_temp, raw_temp = [],[],[],[]
        for s,p,r in zip(stim, para, resp):
            if p[0] <= 3.0:
                pass
            elif p[2] == 0.0:
                pass
            else:
                para_temp.append(p)
                raw_temp.append(r)
                resp_temp.append((r-np.mean(r[:1250]))*100)
                stim_temp.append(s)
        
        """NOTE: resp is nocth filtered, baseline corrected and changed unit to mV"""
        
        self.resp_raw = raw_temp
        resp_temp = TFTool.prefilter(resp_temp, 25000)
        self.resp = resp_temp
        self.para = para_temp
        self.stim = stim_temp
        self.filename = filename
        self.version = version
        '''version 1 --- start: 1250, end: 38750'''
        '''version 2 --- start: 1250, end: 26250'''
        self.bf = bf
        self.band_left = band_left
        self.band_right = band_right
        self.use_band=use_band
        
        _para = np.swapaxes(np.array(self.para),0,1)
        self.mr_label = sorted(set(_para[2][:]))
        self.cf_label = sorted(set(_para[0][:]))
        self.bw_label = sorted(set(_para[1][:]))
        self.features = pd.DataFrame()
        self.psth = np.mean(self.resp, axis=0)
        
    
    def get_psth(self):
        y_all = np.mean(self.resp, axis=0)
        x_all = np.arange(0,len(y_all))
        err_all = stats.sem(self.resp, axis=0)
        
        if self.use_band:
            stim_in, stim_ex, resp_in, resp_ex, para_in, para_ex, _, _ = lsfm.resp_bf_or_not(self.stim, self.resp, self.para, self.bf, [self.band_left, self.band_right])
        else:
            stim_in, stim_ex, resp_in, resp_ex, para_in, para_ex, _, _ = lsfm.resp_bf_or_not(self.stim, self.resp, self.para, self.bf)
        
        y_in = np.mean(resp_in, axis=0)
        x_in = np.arange(0,len(y_in))
        err_in = stats.sem(resp_in, axis=0)
        
        y_ex = np.mean(resp_ex, axis=0)
        x_ex = np.arange(0,len(y_ex))
        err_ex = stats.sem(resp_ex, axis=0)
        
        return {'x_all':x_all, 'y_all':y_all, 'err_all':err_all,
                'x_in':x_in, 'y_in':y_in, 'err_in':err_in,
                'x_ex':x_ex, 'y_ex':y_ex, 'err_ex':err_ex}
    
    
    def plot_psth(self, x_in_ms=True, saveplot=False):
        """
        Plot PSTH of all lsfm responses

        Parameters
        ----------
        set_x_intime : boolean
            Ture to set x-axis in second instead of data point. The default is False.
        saveplot : boolean
            Saveplot. The default is False.

        Returns
        -------
        None.

        """
        
        resp = np.array(self.resp)
        
        y = np.mean(resp, axis=0)
        x = np.arange(0,len(y))
        err = stats.sem(resp, axis=0)
        
        if self.version == 1:
            if x_in_ms:
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,38750]]
                ax.set_xlim(0,len(x))
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                ax.set_xticks(np.linspace(0,50000,11))
                ax.set_xticklabels(label, rotation = 45)
                #ax.xticks(rotation = 45)
                ax.set_title(f'{self.filename}_PSTH-time', fontsize=14)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}_PSTH_time.png', dpi=500, bbox_inches='tight')
                    plt.savefig(f'{self.filename}_PSTH_time.pdf', dpi=500, format='pdf', bbox_inches='tight')
    
                plt.show()
                plt.clf()
                plt.close(fig)
            else:
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                [ax.axvline(x=_x, color='k', linestyle='dotted', alpha=0.3) for _x in np.arange(0,50001,2500)]
                [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in [1250,38750]]
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,50000,21))
                plt.xticks(rotation=45)
                ax.set_title(f'{self.filename}_PSTH', fontsize=14)
                ax.set_xlabel('data point (2500/100ms)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}_PSTH.png', dpi=500, bbox_inches='tight')
                    plt.savefig(f'{self.filename}_PSTH.pdf', dpi=500, format='pdf', bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
                
                
        elif self.version >= 2:
            if x_in_ms:
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                [ax.axvline(x=_x, color='k', linestyle='dotted', alpha=0.3) for _x in np.arange(0,37501,2500)]
                [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in [1250,26250]]
                ax.set_xlim(0,len(x))
                label = list(np.round(np.linspace(0, 1.5, 16), 2))
                ax.set_xticks(np.linspace(0,37500,16))
                ax.set_xticklabels(label, rotation = 45)
                #ax.xticks(rotation = 45)
                ax.set_title(f'{self.filename}_PSTH-time', fontsize=14)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('Avg Response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                if saveplot:
                    plt.savefig(f'{self.filename}_PSTH_time.png', dpi=500, bbox_inches='tight')
                    plt.savefig(f'{self.filename}_PSTH_time.pdf', dpi=500, format='pdf', bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
            
            else:  
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                [ax.axvline(x=_x, color='k', linestyle='dotted', alpha=0.3) for _x in np.arange(0,37501,2500)]
                [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in [1250,26250]]
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,37500,16))
                plt.xticks(rotation=45)
                ax.set_title(f'{self.filename}_PSTH', fontsize=14)
                ax.set_xlabel('data point (2500/100ms)', fontsize=16)
                ax.set_ylabel('Avg Response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                if saveplot:
                    plt.savefig(f'{self.filename}_PSTH.png', dpi=500, bbox_inches='tight')
                    plt.savefig(f'{self.filename}_PSTH.pdf', dpi=500, format='pdf', bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
        
# =============================================================================
#         plt.plot(x,y)
#         plt.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
#         plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
#         plt.axvline(x=26250, color='k', linestyle='--', alpha=0.5)
#         label = list(np.round(np.linspace(0, 2.0, 16), 2))
#         
#         if set_x_intime:
#             plt.xticks(np.linspace(0,37500,16),label)
#         else:
#             plt.xticks(np.linspace(0,37500,16))
#             plt.xticks(rotation = 45)
#         
#         ax = plt.subplot()
#         txt = (f'{self.filename}-PSTH')
#         ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
#         
#         
#         if saveplot:
#             plt.savefig(f'{self.filename}-PSTH.png', dpi=500)
#             plt.clf()
#         else:
#             plt.show()
# =============================================================================
    
    def psth_seperateByPara(self, plot=True, saveplot=False) -> dict:
        """
        Ploting PSTH seperate with each parameters.

        Parameters
        ----------
        plot : boolean
            Show plots. The default is False.
        saveplot : boolean
            Save plots. The default is False.

        Returns
        -------
        dict
            Return responses seperated by parameters: 
            'modrate', 'centerfreq', 'bandwidth'.

        """
        
        resp = np.array(self.resp)
        _para = np.swapaxes(np.array(self.para),0,1)
        para_mod = _para[2][:]
        para_cf = _para[0][:]
        para_band = _para[1][:]
        
        resp_mod, resp_cf, resp_band=[],[],[]
        
        for mod in self.mod_label:
            temp = []
            for p, r in zip(para_mod, resp):
                if p == mod:
                    temp.append(r)     #resp with same mod_rate
            resp_mod.append(temp)       #resp seperated by mod_rate
    
        for cf in  self.cf_label:
            temp = []
            for p, r in zip(para_cf, resp):
                if p == cf:
                    temp.append(r)
            resp_cf.append(temp)
    
        for band in self.bw_label:
            temp = []
            for p, r in zip(para_band, resp):
                if p == band:
                    temp.append(r)
            resp_band.append(temp)
        
        if self.version == 1:
            for i in range(len(self.mod_label)):
                y = np.mean(resp_mod[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_mod[i], axis=0)
                
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                ax.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,50000,11),label)
                ax.set_title(f'{self.filename}-mod {self.mod_label[i]} Hz', fontsize=16)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-mod {self.mod_label[i]} Hz.png', dpi=500, bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
                        
                
            for i in range(len(self.cf_label)):
                y = np.mean(resp_cf[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_cf[i], axis=0)
                
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                ax.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,50000,11),label)
                ax.set_title(f'{self.filename}-cf {self.cf_label[i]} Hz', fontsize=16)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-cf {self.cf_label[i]} kHz.png', dpi=500, bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
                    
                    
            for i in range(len(self.bw_label)):
                y = np.mean(resp_band[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_band[i], axis=0)
                
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                ax.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,50000,11),label)
                ax.set_title(f'{self.filename}-bdwidth {self.bw_label[i]} kHz', fontsize=16)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-bdwidth {self.bw_label[i]} kHz.png', dpi=500, bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
        
        
        elif self.version == 2:
            for i in range(len(self.mod_label)):
                y = np.mean(resp_mod[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_mod[i], axis=0)
                
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                ax.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=26250, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 1.5, 11), 2))
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,37500,11),label)
                ax.set_title(f'{self.filename}-mod {self.mod_label[i]} Hz', fontsize=16)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-mod {self.mod_label[i]} Hz.png', dpi=500, bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
                
            
            for i in range(len(self.cf_label)):
                y = np.mean(resp_cf[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_cf[i], axis=0)
                
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                ax.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=26250, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 1.5, 11), 2))
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,37500,11),label)
                ax.set_title(f'{self.filename}-cf {self.cf_label[i]} Hz', fontsize=16)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-cf {self.cf_label[i]} kHz.png', dpi=500, bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
                    
                    
            for i in range(len(self.bw_label)):
                y = np.mean(resp_band[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_band[i], axis=0)
                
                fig, ax = plt.subplots()
                ax.plot(x,y)
                ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                ax.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=26250, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 1.5, 11), 2))
                ax.set_xlim(0,len(x))
                ax.set_xticks(np.linspace(0,37500,11),label)
                ax.set_title(f'{self.filename}-bdwidth {self.bw_label[i]} kHz', fontsize=16)
                ax.set_xlabel('time (sec)', fontsize=16)
                ax.set_ylabel('average response (mV)', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-bdwidth {self.bw_label[i]} kHz.png', dpi=500, bbox_inches='tight')
                    
                plt.show()
                plt.clf()
                plt.close(fig)
                
            
        #return resp grouped by parameters
        return {'modrate':resp_mod, 'centerfreq':resp_cf,
                'bandwidth':resp_band}
    
    
    def plot_psth_wwobf(self, saveplot=False):
        """
        Generate PSTH seperate by stimulus crossed best frequency or not.
        If bandwidth is specified, stimulus ever entered the tuning area would be included.
        Requires lsfm.resp_bf_or_not or lsfm.resp_bfband_or_not

        Parameters
        ----------
        resp : 2d-array
            DESCRIPTION.
        para : list of tuple
            Tuple need to be (frequency, bandwidth, []).
        bf : float
            Best frequency.
        version : int
            For x-axis range in plot.
        filename : str
            Filename to show in plot.
        bandwidth : list of floats, optional
            List of responsive area edges in the form of [left_edge, right_edge], unit should be Hz. The default is None.
        plot : booln, optional
            To show inline plot. The default is False.
        saveplot : booln, optional
            To save plot to png. The default is False.

        Returns
        -------
        list
            ([psth_x_inbf, psth_y_inbf, err_inbf], [psth_x_exbf, psth_y_exbf, err_exbf]).

        """
        data = self.get_psth()
        x1 = data['x_in']
        y1 = data['y_in']
        err1 = data['err_in']
        x2 = data['x_ex']
        y2 = data['y_ex']
        err2 = data['err_ex']
        
        if self.use_band:
            condition='band'
        else:
            condition='bf'
        
        if self.version == 1:

            fig, ax = plt.subplots()
            ax.plot(x1,y1,color='midnightblue', label='w/_bf')
            ax.fill_between(x1, y1+err1, y1-err1, color='cornflowerblue', alpha=0.6)
            ax.plot(x2,y2,color='firebrick', label='w/o_bf')
            ax.fill_between(x2, y2+err2, y2-err2, color='salmon', alpha=0.6)
            
            [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,38750]]
            ax.set_xlim(0,len(x1))
            label = list(np.round(np.linspace(0, 2, 11), 2))
            ax.set_xticks(np.linspace(0,50000,11))
            ax.set_xticklabels(label, rotation = 45)
            #ax.xticks(rotation = 45)
            #ax.set_title(f'{filename}_PSTH_BF', fontsize=14)
            ax.set_xlabel('time (sec)', fontsize=16)
            ax.set_ylabel('Membrane Potential (mV)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(loc='upper left', fontsize=14)
            
            if saveplot:
                plt.savefig(f'{self.filename}_PSTH_{condition}.png', dpi=500, bbox_inches='tight')
            
            plt.show()
            plt.clf()
            plt.close(fig)
        
        elif self.version >= 2:
              
            fig, ax = plt.subplots()
            ax.plot(x1,y1,color='midnightblue', label='In RF')
            ax.fill_between(x1, y1+err1, y1-err1, color='cornflowerblue', alpha=0.6)
            ax.plot(x2,y2,color='firebrick', label='Out RF')
            ax.fill_between(x2, y2+err2, y2-err2, color='salmon', alpha=0.6)
            ax.legend()
            
            [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,26250]]
            ax.set_xlim(0,len(x1))
            label = list(np.round(np.linspace(0, 1.5, 16), 2))
            ax.set_xticks(np.linspace(0,37500,16))
            ax.set_xticklabels(label, rotation = 45)
            #ax.xticks(rotation = 45)
            #ax.set_title(f'{filename}_PSTH_BfBand', fontsize=14)
            ax.set_title(f'{self.filename} psth wwo {condition}', fontsize=18)
            ax.set_xlabel('time (sec)', fontsize=16)
            ax.set_ylabel('Membrane Potential (mV)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            if saveplot:
                plt.savefig(f'{self.filename}_PSTH_{condition}.png', dpi=500, bbox_inches='tight')
                plt.savefig(f'{self.filename}_PSTH_{condition}.pdf', dpi=500, format='pdf', bbox_inches='tight')
            
            plt.show()
            plt.clf()
            plt.close(fig)
    
        
    def psth_trend(self, tuning=None, plot=True, saveplot=False, **kwargs) -> None:
        """
        Generate average potential vs base. Traces seperated by group.

        Parameters
        ----------
        saveplot : boolean
            Set ture to save plot. The default is False.
        tuning : tuple, optional
            Use tuning=(start,end) to specify the frequency range in kHz to include
            when averaging carrier frequency.
        **kwargs : str
            window = (start,end) to specify the resposne range in datapoint
            locaiton = 'onset', 'second, 'plateau', 'offset'
            onset: 0-0.4sec, second: 0.4-0.8sec, plateau: 0.8-1.6sec, offset: 1.6-2.0sec
           
        
        Returns
        -------
        None.

        """
        
        _para = np.swapaxes(np.array(self.para),0,1)
        para_cf = _para[0][:]
        para_band = _para[1][:]
        para_mod = _para[2][:]
        para_name = ['Center Freq', 'Band Width', 'Mod Rate']
        label_list = [self.cf_label, self.bw_label, self.mod_label]
        para_list = [para_cf, para_band, para_mod]
        
        from itertools import permutations
        aranges = []
        
        #generate combinations
        for i in permutations(range(3),3):
            aranges.append(i)
        
        #e.g. arange=[0,2,1]: grouped by cf, plot mV versus mod rate(base),
        #average over bandwidth
        for arange in aranges:
            group = para_name[arange[0]]
            base = para_name[arange[1]]
            volt = para_name[arange[2]]
            
            samegroup=[]    #reset in different arangement
            for g in label_list[arange[0]]:               
                samebase=[]     #reset in each group
                for b in label_list[arange[1]]:
                    resp_incategory=[]      #reset in every base
                    for i,p in enumerate(self.para):                                                                 
                        if p[arange[0]] == g and p[arange[1]] == b:

                            _resp = Psth.baseline(self.resp[i])
                            
                            set_window = kwargs.get('window')
                            if set_window:
                                if(set_window[0]>set_window[1]):
                                    set_window[0], set_window[1] = set_window[1], set_window[0]
                                if(min(set_window) < 0):
                                    raise ValueError('Cannot start before zero')
                                if(max(set_window) > len(_resp)):
                                    raise ValueError('Exceed data range')
                            
                                _resp = _resp[set_window[0]:set_window[1]]
                           
                            #exclude resp if not in tuning range
                            if arange[2] == 0 and tuning != None:
                                if p[arange[2]] < float(tuning[0]) or p[arange[2]] > float(tuning[1]):
                                    pass
                                else:
                                    resp_incategory.append(_resp)
                            else:
                                resp_incategory.append(_resp)

                    if resp_incategory:
                        v_mean = np.mean(resp_incategory, axis=1)
                        samebase.append([g,b,np.mean(v_mean),np.std(v_mean)])
                samegroup.append(samebase)
                
            colors = plt.cm.rainbow(np.linspace(0.3,1,len(samegroup)))
            fig, ax = plt.subplots()
            
            for i,gp in enumerate(samegroup):              
                x,y,err=[],[],[]
                for ii in gp:
                    x.append(ii[1])
                    y.append(ii[2])
                    err.append(ii[3])
                try:                                            
                    ax.errorbar(x,y,yerr=err, color=colors[i], capsize=(4), marker='o', elinewidth= 2, linewidth = 3, label=f'{group}-{gp[0][0]}')
                except IndexError:
                    pass
                
                if set_window:
                    txt = f'{self.filename}_{base}_{group}_window'+str(set_window)
                else:
                    txt=f'{self.filename}_{base}_{group}_all'
                
                if tuning:
                    txt = f'{txt}_{tuning}'
                
                ax.set_title(txt)
                if base == 'Center Freq':
                    ax.set_xscale('log')
                    ax.set_xticks(self.cf_label)
                    ax.set_xticklabels(self.cf_label, rotation=30)
                elif base == 'Band Width':
                    label = [round(i,2) for i in self.bw_label]
                    ax.set_xscale('log')
                    ax.set_xticks(self.bw_label)
                    ax.set_xticklabels(label, rotation=30)
                elif base == 'Mod Rate':
                    ax.set_xscale('log')
                    ax.set_xticks(self.mod_label)
                    ax.set_xticklabels(self.mod_label, rotation=30)
                ax.minorticks_off()
                ax.set_xlabel(f'{base}', fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=14)
                plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            
            if saveplot:
                plt.savefig(f'{txt}.png', dpi=500, bbox_inches='tight')
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
                
            if plot:
                plt.show()
                plt.clf()
                plt.close(fig)
                      

    def psth_window(self, window, featname, tuning=None, saveplot=False, savenotes=False):
        """
        Generate PSTH for every parameters within the range of interest.

        Parameters
        ----------
        window : Tuple
            Use (start,end) to specify the range in data point.
        featname : str
            The name of the feature of interest.
        saveplot : TYPE
            Save plot. The default is False.
        savenotes : TYPE
            Save Pandas dataframe to csv file. The default is False.

        Returns
        -------
        None.

        """
        
        try: self.features
        except NameError:
            self.features = pd.DataFrame()
        
        #be aware the resp in resp_list is returned from psth_para which went through
        #baseline correction thus already scaled to real value.
        para_dict = Psth.psth_para(self)
        resp_list = [para_dict['centerfreq'],para_dict['bandwidth'],para_dict['modrate']]
        label_list = [self.cf_label, self.bw_label, self.mod_label]
        para_name = ['center_freq (KHz)', 'bandwidth (octave)', 'mod_rate (Hz)']
        
        for par in range(3): #parameter
            x,y,err = [],[],[] 
            for con in range(len(label_list[par])): #condition
                def slicing(arr, start, end):
                    return arr[start:end]
                resp_window = np.apply_along_axis(slicing, 1, \
                        resp_list[par][con], start=window[0], end=window[1])  
                resp_incondition = np.mean(resp_window, axis=1)
                _x = label_list[par][con]
                _y = np.mean(resp_incondition, axis=0)
                _err = stats.sem(resp_incondition, axis=0)
                x.append(_x)
                y.append(_y)
                err.append(_err)
                temp = pd.DataFrame({'x':[_x], 'y':[_y], 'error':[_err], 'base':[f'{para_name[par]}'], \
                    'feature':[f'{featname}'], 'start':[window[0]], 'end':[window[1]]})
                self.features = pd.concat((self.features, temp), axis=0)            
            
                        
            #plt.plot(x,y, marker='o')
            plt.errorbar(x,y,yerr=err, color='k', capsize=(4), marker='o')
            plt.xlabel(f'{para_name[par]}')
            plt.ylabel('mV')
            if par == 0 or par == 2:
                plt.xscale('log')
            ax = plt.subplot()
            txt = (f'{self.filename}_{featname}, range:{window[0]} to {window[1]}')
            ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
            
            if saveplot:
                plt.savefig(f'{self.filename}_p_{para_name[par]}_Feature_{featname}.png', dpi=500, bbox_inches='tight')
                plt.clf()
                plt.close()
            else:
                plt.show()
        if savenotes:
            self.features.to_csv(f'{self.filename}--feature_notes.csv', index=False)
    
    
    def get_psth_cat(self, inRF=True):
        from IPython import get_ipython
        from matplotlib.lines import Line2D
        from matplotlib.widgets import Button, CheckButtons
        get_ipython().run_line_magic('matplotlib', 'qt5')
        data = self.get_psth()
        
        if not inRF:
            x = data['x_ex']
            y = data['y_ex']
        else:
            x = data['x_in']
            y = data['y_in']
        
        # Initialize figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        line, = ax.plot(x, y, label="Signal")
        
        # === CUSTOM CURSOR NAMES ===
        cursor_names = ["onpeak_start", "onpeak", "onpeak_stop", "offpeak_start", "offpeak", 'offpeak_stop']
        NUM_CURSORS = len(cursor_names)
    
        cursor_lines = []
        cursor_labels = []
        cursor_positions = np.linspace(x[0], x[-1], NUM_CURSORS + 2)[1:-1]
        dragging = [None]
        active_cursor = [None]
        cursor_result = []
        done_flag = [False]
    
        ymin, ymax = ax.get_ylim()
        for i, xpos in enumerate(cursor_positions):
            # Draw vertical line
            line = Line2D([xpos, xpos], [ymin, ymax], color='r', linestyle='--', lw=1.5)
            ax.add_line(line)
            cursor_lines.append(line)
    
            # Label at middle y
            mid_y = (ymin + ymax) / 2
            label = ax.text(xpos, mid_y, cursor_names[i],
                            rotation=90, fontsize=20,
                            color='blue', ha='left', va='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            cursor_labels.append(label)
    
        # === Checkboxes as one horizontal row ===
        ax_check = plt.axes([0.05, 0.9, 0.9, 0.08])  # across the top
        checkbox_status = [False] * NUM_CURSORS
        check = CheckButtons(ax_check, cursor_names, checkbox_status)
        [lbl.set_fontsize(16) for lbl in check.labels]
    
        changing_checkbox = [False]
    
        def on_check(label):
            if changing_checkbox[0]:
                return
            idx = cursor_names.index(label)
            changing_checkbox[0] = True
            current_status = check.get_status()
            for i in range(NUM_CURSORS):
                if current_status[i] and i != idx:
                    check.set_active(i)  # toggle off
            # Re-read status after toggles
            new_status = check.get_status()
            active_cursor[0] = idx if new_status[idx] else None
            changing_checkbox[0] = False
    
        check.on_clicked(on_check)
    
        def on_press(event):
            if event.inaxes != ax or event.xdata is None:
                return
            if active_cursor[0] is not None:
                dragging[0] = active_cursor[0]
    
        def on_motion(event):
            if dragging[0] is None or event.inaxes != ax or event.xdata is None:
                return
            idx = dragging[0]
            x_new = event.xdata
            cursor_lines[idx].set_xdata([x_new, x_new])
            # Update label position
            cursor_labels[idx].set_position((x_new, (ymin + ymax) / 2))
            fig.canvas.draw_idle()
    
        def on_release(event):
            dragging[0] = None
    
        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)
    
        def output_and_close(event):
            for i, line in enumerate(cursor_lines):
                x_pos = line.get_xdata()[0]
                y_pos = np.interp(x_pos, x, y)
                cursor_result.append((cursor_names[i], x_pos, y_pos))
            print("Cursor values:")
            for name, xv, yv in cursor_result:
                print(f"{name}: x = {xv:.2f}, y = {yv:.2f}")
            done_flag[0] = True
            plt.close('all')
            get_ipython().run_line_magic('matplotlib', 'inline')
    
        ax_button = plt.axes([0.8, 0.05, 0.12, 0.075])
        button = Button(ax_button, "Output")
        button.on_clicked(output_and_close)
    
        plt.legend()
        plt.show(block=False)
    
        # Wait loop until output is clicked
        while not done_flag[0] and plt.fignum_exists(fig.number):
            plt.pause(0.1)
    
        return cursor_result


def psth_wwo_bf(stim, resp, para, bf, version, filename, bandwidth=None, saveplot=False):
    """
    Generate PSTH seperate by stimulus crossed best frequency or not.
    If bandwidth is specified, stimulus ever entered the tuning area would be included.
    Requires lsfm.resp_bf_or_not or lsfm.resp_bfband_or_not

    Parameters
    ----------
    resp : 2d-array
        DESCRIPTION.
    para : list of tuple
        Tuple need to be (frequency, bandwidth, []).
    bf : float
        Best frequency.
    version : int
        For x-axis range in plot.
    filename : str
        Filename to show in plot.
    bandwidth : float, optional
        Bandwidth in octave. The default is None.
    plot : booln, optional
        To show inline plot. The default is False.
    saveplot : booln, optional
        To save plot to png. The default is False.

    Returns
    -------
    list
        ([psth_x_inbf, psth_y_inbf, err_inbf], [psth_x_exbf, psth_y_exbf, err_exbf]).

    """
    
    if bandwidth:
        resp_in, resp_ex, para_in, para_ex, _, _ = lsfm.resp_bfband_or_not(stim, resp, para, bf, bandwidth)
    else:
        resp_in, resp_ex, para_in, para_ex, _, _ = lsfm.resp_bf_or_not(stim, resp, para, bf)
    
    p1 = Psth(resp_in, para_in, filename, version)
    p2 = Psth(resp_ex, para_ex, filename, version)
    x1,y1,err1 = p1.get_psth()
    x2,y2,err2 = p2.get_psth()
    
    if version == 1:
    
        fig, ax = plt.subplots()
        ax.plot(x1,y1,color='midnightblue', label='w/_bf')
        ax.fill_between(x1, y1+err1, y1-err1, color='cornflowerblue', alpha=0.6)
        ax.plot(x2,y2,color='firebrick', label='w/o_bf')
        ax.fill_between(x2, y2+err2, y2-err2, color='salmon', alpha=0.6)
        
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,38750]]
        ax.set_xlim(0,len(x1))
        label = list(np.round(np.linspace(0, 2, 11), 2))
        ax.set_xticks(np.linspace(0,50000,11))
        ax.set_xticklabels(label, rotation = 45)
        #ax.xticks(rotation = 45)
        #ax.set_title(f'{filename}_PSTH_BF', fontsize=14)
        ax.set_xlabel('time (sec)', fontsize=16)
        ax.set_ylabel('Membrane Potential (mV)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper left', fontsize=14)
        
        if saveplot:
            plt.savefig(f'{filename}_PSTH_bfBand.png', dpi=500, bbox_inches='tight')
            #plt.savefig(f'{filename}_PSTH_bfBand.pdf', dpi=500, format='pdf', bbox_inches='tight')
            
        plt.show()
        plt.clf()
        plt.close(fig)

    
    elif version >= 2:
    
        #p1.plot_psth(saveplot=False)
        #p2.plot_psth(saveplot=False)
        #resp_by_para = p.psth_para(plot=True, saveplot=False)
        #p.psth_trend(saveplot=False)
        
        fig, ax = plt.subplots()
        ax.plot(x1,y1,color='midnightblue', label='In RF')
        ax.fill_between(x1, y1+err1, y1-err1, color='cornflowerblue', alpha=0.6)
        ax.plot(x2,y2,color='firebrick', label='Out RF')
        ax.fill_between(x2, y2+err2, y2-err2, color='salmon', alpha=0.6)
        ax.legend()
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,26250]]
        ax.set_xlim(0,len(x1))
        label = list(np.round(np.linspace(0, 1.5, 16), 2))
        ax.set_xticks(np.linspace(0,37500,16))
        ax.set_xticklabels(label, rotation = 45)
        #ax.xticks(rotation = 45)
        #ax.set_title(f'{filename}_PSTH_BfBand', fontsize=14)
        ax.set_xlabel('time (sec)', fontsize=16)
        ax.set_ylabel('Membrane Potential (mV)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if saveplot:
            plt.savefig(f'{filename}_PSTH_BfBand.png', dpi=500, bbox_inches='tight')
            plt.savefig(f'{filename}_PSTH_BfBand.pdf', dpi=500, format='pdf', bbox_inches='tight')
        
        plt.show()
        plt.clf()
        plt.close(fig)


def psth_para_sepearte(stim, resp, para, which_parameter, bf, filename, *suffix : str, plot=True, saveplot=False):
    para_type = {0: 'cf', 1: 'bw', 2: 'mf'}
    unit = {0: 'kHz', 1: 'octave', 2: 'Hz'}
    
    data = lsfm.seperate_by_para(stim, resp, para, which_parameter)
    para_set = list(data['parameter'])
    para_set.sort()
    
    psth_type=[]
    parameter_type=[]
    for i,p in enumerate(para_set):
        resp_set = np.array([r - np.mean(r[750:1250]) for r in data['resp'][i]])
        #resp_set = np.array(data['resp'][i])
        psth = np.mean(resp_set, axis=0)
        psth_type.append(psth)
        parameter_type.append((p,len(resp_set)))
        
        if plot or saveplot:
            plt.plot(psth, label=f'{p} ({len(resp_set)})')
            plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.title(f'{filename}_{para_type[which_parameter]}, {suffix}, bf:{bf/1000:.1f}kHz')
    
    if saveplot:
        plt.savefig(f'{filename}_PSTH_{para_type[which_parameter]}_{suffix}.png', dpi=500, bbox_inches='tight')
    
    if plot:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.clf()
        
    return psth_type, parameter_type

        

def get_section_old(psth_sep, para_sep, para_type: int):
    """
    get numbers from interested section of a psth with specific parameter
    
    lsfm stimulus period (1250, 26250)
    section:    1. onset peak - highest potential in (1250,7500)
                2. onset rise - 10-90% of onset peak
                3. onset decay - either 1 exponential decay or drop to 37% of peak amplitude
                4. onset charge - area between (rise10,decay)
                5. offset peak - highest potential in (26250,35000)
                6. offset rise - 10-90% of offset peak
                7. offset decay - either 1 exponential decay or drop to 37% of peak amplitude
                8. offset charge - area between (rise10,decay)
                9. sustain potetial - average potential in (15000,25000)
    
    Parameters
    ----------
    psth_sep : 3d-array
        array of appending 3 (each parameter) psth from psth_para_sepearte function
    para_sep : 2d-array
        array of appending 3 para from psth_para_sepearte function
    para_type : int
        0. cf
        1. bw
        2. mr

    Returns
    -------
    dictionary
    
    item structure: (parameters (eg. 3 if bandwidth parameters are 0.3,1.5,3 octave), numbers of each category)
    numbers of each categories are list of list. On, offset rise is list of tuple. 

    """
    psths = np.array(psth_sep[para_type])
    paras = np.array(para_sep[para_type])
    current_type = ['cf', 'bw', 'mr'][para_type]
    fs = 25000
    
    data = []
    #loop through individual parameter (e.g. mod = 8Hz)
    for idx, psth in enumerate(psths):            
        def exponential_func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        def alpha_function(t, A, t0, tau, C):
            return A * (t - t0) * np.exp(-(t - t0) / tau) * (t >= t0) + C
        
        def decay37(psth, peak, peak_loc, switch: str):
            if switch=='on':
                if all(psth[peak_loc:15000] - 0.37*peak>0):
                    decay = np.argmin(psth[peak_loc:15000])
                else:
                    decay = np.diff(np.sign(psth[peak_loc:15000] - 0.37*peak))<0
                    decay = [i for i,d in enumerate(decay) if d][0]
                decay_loc = decay + peak_loc
                decay_time = decay / (fs/1000)    #change unit to ms
            
            elif switch=='off':
                off_base = np.mean(psth[25250:26250])
                if all(psth[peak_loc:35000] - 0.37*peak < off_base):
                    off_base = np.min(psth[25250:peak_loc])
                    decay = np.diff(np.sign(psth[peak_loc:35000] - off_base - 0.37*peak))<0
                else:
                    decay = np.diff(np.sign(psth[peak_loc:35000] - off_base - 0.37*peak))<0
                
                if all(decay == False):
                    decay = 0
                else:
                    decay = [i for i,d in enumerate(decay) if d][0]
                decay_time = decay / (fs/1000)        #change unit to ms
                decay_loc = decay + peak_loc
            
            else:
                raise ValueError('switch can only be on or off')
            
            return decay_time, decay_loc
        
        print(f'current: type {current_type}, parameter {paras[idx]}, list index {idx}')
        
        para_spec = paras[idx][0]
        repeat = paras[idx][1]
        psth = TFTool.butter(psth, 6, 50, 'low', fs)
        
        """onset"""
        on_base = np.mean(psth[250:1250])
        on_peak_range = psth[1250:7500]
        
        #alpha function fit to check if peak polarity
        charge_plus = sum([i for i in on_peak_range if i-on_base > 0])
        charge_minus = sum([i for i in on_peak_range if i-on_base < 0])
        if charge_plus - abs(charge_minus) >= 0:
            polar = 1
        else:
            polar = -1
        
        on_peak_range = on_peak_range*polar
        
        on_peak = (np.max(on_peak_range) - on_base)*polar
        onpeak_loc = np.argmax(on_peak_range)+1250
        if on_peak*polar < 0:
            on_base = np.min(psth[1250:onpeak_loc]*polar)
            on_peak = np.max(on_peak_range) - on_base
        
        on_rise10 = np.diff(np.sign(psth[1250:onpeak_loc]*polar - on_base - 0.1*on_peak*polar))>0
        if all(on_rise10==False):
            on_rise10 = 1251
        else:
            on_rise10 = [i for i,r in enumerate(on_rise10) if r][0] + 1250
        
        on_rise90 = np.diff(np.sign(psth[1250:onpeak_loc]*polar - on_base - 0.9*on_peak*polar))>0
        on_rise90 = [i for i,r in enumerate(on_rise90) if r][-1] + 1250
        on_rise = (on_rise90 - on_rise10) / (fs/1000) #in milisecond
        
        pivot = int(np.argmin(psth[onpeak_loc:15000]*polar)+onpeak_loc)
        if pivot == onpeak_loc:
            on_decay_loc = onpeak_loc
            on_decay = 0
        
        else:
            try:
                popt, pcov = optimize.curve_fit(exponential_func, range(len(psth[onpeak_loc:pivot])), psth[onpeak_loc:pivot])
                
                y_pred = exponential_func(range(len(psth[onpeak_loc:15000])), *popt)
                on_decay_loc = 1/popt[1]+onpeak_loc
                on_decay = 1/popt[1]/(fs/1000)
                
                if (on_decay_loc > 15000):
                    #if fit location larger than boundary, use 37% drop instead
                    on_decay, on_decay_loc = decay37(psth*polar, on_peak*polar, onpeak_loc, switch='on')
            
            
            except RuntimeError:
                #if cannot fit, use 37% drop instead
                on_decay, on_decay_loc = decay37(psth*polar, on_peak*polar, onpeak_loc, switch='on')
    
        onset_charge = np.sum(psth[int(on_rise10):int(on_decay_loc)]-on_base)
        
        
        """offset"""
        offpeak_loc = np.argmax(psth[26251:35000])+26251
        off_base = np.mean(psth[25250:26250])
        off_peak = np.max(psth[26251:35000]) - off_base
        if off_peak < 0:
            off_base = np.min(psth[25250:offpeak_loc])
            off_peak = np.max(psth[26251:35000]) - off_base
        
        off_rise10 = np.diff(np.sign(psth[26251:offpeak_loc] - off_base - 0.1*off_peak))>0
        if all(off_rise10==False) or off_rise10.size==0:
            off_rise10 = 26251
        else:
            off_rise10 = [i for i,r in enumerate(off_rise10) if r][0] + 26251
        off_rise90 = np.diff(np.sign(psth[26251:offpeak_loc] - off_base - 0.9*off_peak))>0
        if all(off_rise90==False) or off_rise90.size == 0:
            off_rise90 = 26251
        else:
            off_rise90 = [i for i,r in enumerate(off_rise90) if r][0] + 26251
        off_rise = (off_rise90 - off_rise10) / (fs/1000) #in milisecond
        
        try:
            pivot = int(np.argmin(psth[offpeak_loc:37500])+offpeak_loc)
            popt, pcov = optimize.curve_fit(exponential_func, range(len(psth[offpeak_loc:pivot])), psth[offpeak_loc:pivot])
            y_pred = exponential_func(range(len(psth[offpeak_loc:35000])), *popt)
            off_decay_loc = 1/popt[1]+offpeak_loc
            off_decay = 1/popt[1]/(fs/1000)     #change unit to ms
            
            if off_decay_loc>35000:
                #out of boundary, use 37% drop instead
                off_decay, off_decay_loc = decay37(psth, off_peak, offpeak_loc, switch='off')

        except RuntimeError:
            #use 37% drop to get decay time instead
            off_decay, off_decay_loc = decay37(psth, off_peak, offpeak_loc, switch='off')
        
        if off_decay_loc == offpeak_loc:
            offset_charge = 0
        else:
            offset_charge = np.sum(psth[int(off_rise10):int(off_decay_loc)]-off_base)
        
        
        """sustain"""
        sustain_v = np.mean(psth[15000:25000])
        
        
        #change unit
        on_peak = on_peak*100       #mV
        off_peak = off_peak*100     #mV
        onpeak_loc = onpeak_loc/(fs/1000)       #ms
        offpeak_loc = offpeak_loc/(fs/1000)     #ms
        sustain_v  = sustain_v*100
        
        
        """save data from individual parameter to list"""
        data.append([para_spec, repeat, on_peak, onpeak_loc, on_rise, on_rise10, on_rise90, on_decay, on_decay_loc, onset_charge,
                     off_peak, offpeak_loc, off_rise, off_rise10, off_rise90, off_decay, off_decay_loc, offset_charge, sustain_v])
    
    
    data = np.swapaxes(np.array(data), 0, 1)
    file = {'parameter':data[0], 'repeat':data[1],
            'on_peak':data[2], 'on_loc':data[3], 'on_rise':data[4], 'on_rise10_loc':data[5], 
            'on_rise90_loc':data[6], 'on_decay':data[7], 'on_decay_loc':data[8], 'on_charge':data[9],
            'off_peak':data[10], 'off_loc':data[11], 'off_rise':data[12], 'off_rise10_loc':data[13], 
            'off_rise90_loc':data[14], 'off_decay':data[15], 'off_decay_loc':data[16], 'off_charge':data[17],
            'sustain':data[18]}
    
    return file    


def get_section(filename, mouseID, site, psth_para_sep, para_sep, para_type: int):
    psths = np.array(psth_para_sep[para_type])
    paras = np.array(para_sep[para_type])
    current_type = ['cf', 'bw', 'mr'][para_type]
    fs = 25000
    df_cursor = pd.read_excel('lsfm_PSTH_cursors.xlsx')
    cursor = df_cursor[(df_cursor['mouseID']==mouseID)&(df_cursor['filename']==filename)]
    
    data = []
    #loop through individual parameter (e.g. mod = 8Hz)
    for idx, psth in enumerate(psths):     
        print(f'current: type {current_type}, parameter {paras[idx]}, list index {idx}')
        
        para_spec = paras[idx][0]
        repeat = paras[idx][1]
        psth = TFTool.butter(psth, 6, 50, 'low', fs)
        baseline = np.mean(psth[1000:1250])
        psth = psth-baseline
        
        """onset"""
        on_peak_range = psth[1250:3750]
        peak_plus = np.max(on_peak_range)
        peak_minus = np.min(on_peak_range)
        
        if peak_plus >= abs(peak_minus):
            on_peak_amp = peak_plus
        else:
            on_peak_amp = peak_minus
        
        """onset_charge"""
        on_charge = np.sum(psth[int(cursor['onpeak_start_x'].item()):int(cursor['onpeak_stop_x'].item())])
        
        
        """offset"""
        plateau = np.mean(psth[26000:26250])
        peak_x = int(cursor['offpeak_x'].item())
        off_peak_range = psth[26250:(peak_x - 26250)*2+26250]
        peak_plus = np.max(off_peak_range)
        peak_minus = np.min(off_peak_range)
        
        if peak_plus - plateau >= abs(peak_minus - plateau):
            off_peak_amp = peak_plus - plateau
        else:
            off_peak_amp = peak_minus - plateau
        
        """offset -baseline"""
        if peak_plus >= abs(peak_minus):
            off_peak_amp_base = peak_plus
        else:
            off_peak_amp_base = peak_minus
        
        """offset_charge"""
        off_charge = np.sum(psth[int(cursor['offpeak_start_x'].item()):int(cursor['offpeak_stop_x'].item())])
        
        
        """sustain"""
        sustain_v = np.mean(psth[16250:26250])
        
        
        #change unit
        on_peak_amp = on_peak_amp*100                   #mV
        off_peak_amp = off_peak_amp*100                 #mV
        off_peak_amp_base = off_peak_amp_base*100       #mV
        sustain_v  = sustain_v*100                      #mV
        on_charge = on_charge*100000/fs               #mV*ms
        off_charge = off_charge*100000/fs             #mV*ms
        
        
        """save data from individual parameter to list"""
        data.append([para_spec, repeat, on_peak_amp, on_charge, off_peak_amp, off_peak_amp_base,
                     off_charge, sustain_v])
    
    data = np.swapaxes(np.array(data), 0, 1)
    file = {'filename':filename, 'mouseID':mouseID, 'patch_site':site,
            'parameter':list(data[0]), 'repeat':list(data[1]),
            'on_peak':list(data[2]), 'on_charge':list(data[3]),
            'off_peak':list(data[4]), 'off_peak_base':list(data[5]), 'off_charge':list(data[6]),
            'sustain_potential':list(data[7])}
    
    return file
    

def plot_para_category(filename, bf, data, para_type: int, saveplot=False, **kwargs):
    """
    plot each parameter category (peak amp, kinetics...etc) seperated by paramenter type(cf, bw, mr),
    along values in each parameter type.

    Parameters
    ----------
    filename : str
        filename.
    bf : float
        best frequency in Hz.
    data : dict
        data returned from get_section function.
    para_type : int
        0. cf, 1. bw, 2. mr
    saveplot : Boolean, optional
        save plot output. The default is False.
    **kwargs : TYPE
        extra information used for plotting.

    Returns
    -------
    None.

    """
    p_type = ['cf', 'bw', 'mr'][para_type]
    titles = ['center frequency', 'bandwidth', 'modulation rate'][para_type]
    x_labels = ['distance from bf (kHz)', 'octave', 'modulation frequency (Hz)']
    y_labels = ['','','onset peak amp (mV)', 'onset peak latency (ms)', 'onset peak rise time (ms)', '',
                '','onset peak decay time (ms)', '', 'onset charge (ms*mV)',
                'offset peak amp (mV)', 'offset peak latency (ms)', 'offset peak rise time (ms)', '',
                '', 'offset peak decay time (ms)', '', 'offset charge (ms*mV)',
                'avg sustain potential (mV)']
                
    
    if para_type == 0:
        xx = np.array(data['parameter'])*1000   #transform from kHz to Hz
        xx = np.abs(xx-bf)/1000
    else:
        xx = np.array(data['parameter'])
        
    for idx, (key, value) in enumerate(data.items()):
        if key != 'parameter' or 'repeat':
            yy = value            
            
            fig, ax = plt.subplots()
            ax.scatter(xx,yy, c='r', s=48)
            ax.set_title(f'{key}-{titles}', fontsize=16)
            ax.set_xlabel(f'{x_labels[para_type]}', fontsize=14)
            ax.set_ylabel(f'{y_labels[idx]}', fontsize=14)
            
            if saveplot:
                plt.savefig(f'{filename}_{p_type}_{key}.png', dpi=500, bbox_inches='tight')
            plt.show()
            plt.close()


def plot_single(filename, bf, psth_sep, psth_section, para_type: int, saveplot=False, **kwargs):
    p_type = ['cf', 'bw', 'mr'][para_type]
    titles = ['center frequency', 'bandwidth', 'modulation rate'][para_type]
    x_labels = ['distance from bf (kHz)', 'octave', 'modulation frequency (Hz)'][para_type]
    y_labels = ['','','onset peak amp (mV)', 'onset peak latency (ms)', 'onset peak rise time (ms)', '',
                '','onset peak decay time (ms)', '', 'onset charge (ms*mV)',
                'offset peak amp (mV)', 'offset peak latency (ms)', 'offset peak rise time (ms)', '',
                '', 'offset peak decay time (ms)', '', 'offset charge (ms*mV)',
                'avg sustain potential (mV)']
    fs=25000
    paras = psth_section['parameter']
    
    for idx, para in enumerate(paras):
        if para_type == 0:
            para_adj = np.round(abs(para-bf/1000), 2)
        else:
            para_adj = para
        
        print(paras)
        fig, ax = plt.subplots()
        yy = psth_sep[para_type][idx]*100
        
        on_rise10_loc = int(psth_section['on_rise10_loc'][idx])
        on_rise90_loc = int(psth_section['on_rise90_loc'][idx])
        on_decay_loc = int(psth_section['on_decay_loc'][idx])
        off_rise10_loc = int(psth_section['off_rise10_loc'][idx])
        off_rise90_loc = int(psth_section['off_rise90_loc'][idx])
        off_decay_loc = int(psth_section['off_decay_loc'][idx])
        
        on_area = np.arange(on_rise10_loc, on_decay_loc, 125)
        off_area = np.arange(off_rise10_loc, off_decay_loc, 125)
        
        ax.plot(yy, zorder=10)
        
        #on/off peak location
        on_x = int(psth_section['on_loc'][idx]*(fs/1000))
        ax.scatter(on_x, yy[on_x]+0.2, color='k', alpha=0.8, marker=7, zorder=15)
        off_x = int(psth_section['off_loc'][idx]*(fs/1000))
        ax.scatter(off_x, yy[off_x]+0.2, color='k', alpha=0.8, marker=7, zorder=15)
        
        #on/off kinetics
        ax.scatter(on_rise10_loc-500, yy[on_rise10_loc], color='r', marker='>', alpha=0.8, zorder=15)
        ax.scatter(on_rise90_loc-500, yy[on_rise90_loc], color='orangered', marker='>', alpha=0.8, zorder=15)
        ax.scatter(on_decay_loc+500, yy[on_decay_loc], color='deeppink', marker='<', alpha=0.8, zorder=15)
        
        ax.scatter(off_rise10_loc-500, yy[off_rise10_loc], color='r', marker='>', alpha=0.8, zorder=15)
        ax.scatter(off_rise90_loc-500, yy[off_rise90_loc], color='orangered', marker='>', alpha=0.8, zorder=15)
        ax.scatter(off_decay_loc+500, yy[off_decay_loc], color='deeppink', marker='<', alpha=0.8, zorder=15)
        
        #on/off charge
        for i in on_area:
            i = int(i)
            ax.plot([i,i], [np.mean(yy[250:1250]),yy[i]], color='skyblue', alpha=0.6, zorder=0)
        for i in off_area:
            i = int(i)
            ax.plot([i,i], [np.mean(yy[25250:26250]),yy[i]], color='skyblue', alpha=0.6, zorder=0)
            
        sustain = psth_section['sustain'][idx]
        ax.hlines(y=sustain, xmin=15000, xmax=25000, color='r', linestyle='--', zorder=15)
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,26250]]
        ax.set_title(f'{filename}  {p_type} {para_adj}', fontsize=16)
        if saveplot:
            plt.savefig(f'{filename}_{p_type}_{para}.png', dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()
            
            
def plot_group_category(df, coordinate, parameter_type:int, category:str):
    para_type = ['center frequency', 'bandwidth', 'modulation rate'][parameter_type]
    cate = category
    
    sessions = set(df['filename'])
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    ortho_all = list(coordinate.ortho_A12)
    norm = colors.Normalize(vmin=np.nanmin(ortho_all), vmax=np.nanmax(ortho_all))
    
    fig, ax = plt.subplots(figsize=(10,6))
    for session in sessions:
        xx = np.array(df[df['filename']==session]['parameter'].item())
        yy = np.array(df[df['filename']==session][cate].item())
        mouse = list(df[df['filename']==session]['mouseID'])[0]
        site = list(df[df['filename']==session]['patch_site'])[0]
        ortho_x = coordinate[(coordinate['mouseid']==mouse)&(coordinate['regions']==f'Patch_{site}')].ortho_A12.item()
        
        color = cmap(norm(ortho_x))
        im = ax.plot(xx, yy, c=color, label=mouse)
    
    if parameter_type == 2:
        ax.set_xscale('log')
    
    if 'peak' in cate or 'potential' in cate:
        y_label = 'potential (mV)'
    elif 'charge' in cate:
        y_label = 'charge (mV*ms)'
    
    x_label = ['frequency from bf (octave)', 'bandwidth (octave)', 'mod rate (Hz)'][parameter_type]
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xlabel(x_label, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    
    plt.title(f'Type: {para_type}, Category: {cate}', fontsize=22)
    plt.savefig(f'{para_type}_{cate}.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    
    
    
