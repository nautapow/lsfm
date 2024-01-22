import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
import TFTool
import pandas as pd
import lsfm


class Psth():
    def __init__(self, resp, para, filename, version):
        #exclude carrier less than 3kHz and puretone
        p_t, r_t = [],[]
        for i,p in enumerate(para):
            if p[0] <= 3.0:
                pass
            elif p[2] == 0.0:
                pass
            else:
                p_t.append(p)
                r_t.append(resp[i])
        
        self.resp = r_t
        self.para = p_t
        self.filename = filename
        self.version = version
        '''version 1 --- start: 1250, end: 38750'''
        '''version 2 --- start: 1250, end: 26250'''
        
        _para = np.swapaxes(np.array(self.para),0,1)
        self.mod_label = sorted(set(_para[2][:]))
        self.cf_label = sorted(set(_para[0][:]))
        self.bw_label = sorted(set(_para[1][:]))
        self.features = pd.DataFrame()


        
        
    """reutrn *100 to switch from LabView volt to real mV scale"""
    def baseline(resp_iter):    #correct baseline
        return (resp_iter - np.mean(resp_iter[:50*25]))*100
    
    def baseline_zero(resp_iter):   #fix sound onset to zero
        return (resp_iter - resp_iter[50*25])*100
    
    def psth_all(self, plot=True, saveplot=False):
        """
        Generates PSTH using all lsfm responses.

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
        resp_base = np.apply_along_axis(Psth.baseline, 1, resp)
        
        y = np.mean(resp_base, axis=0)
        x = np.arange(0,len(y))
        err = stats.sem(resp_base, axis=0)
        
        if self.version == 1:
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
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
            
            if plot:
                plt.show()
                plt.clf()
                plt.close(fig)
                
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
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
            
            if plot:
                plt.show()
                plt.clf()
                plt.close(fig)
                
        elif self.version >= 2:
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
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
            
            if plot:
                plt.show()
                plt.clf()
                plt.close(fig)
                
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
            
            if plot:
                plt.show()
                plt.clf()
            else:
                plt.clf()
                
            plt.close(fig)
        
        return x,y,err
        
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
    
    def psth_para(self, plot=True, saveplot=False) -> dict:
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
            for i, p in enumerate(para_mod):
                if p == mod:
                    temp.append(Psth.baseline(resp[i]))     #resp with same mod_rate
            resp_mod.append(temp)       #resp seperated by mod_rate
    
        for cf in  self.cf_label:
            temp = []
            for i, p in enumerate(para_cf):
                if p == cf:
                    temp.append(Psth.baseline(resp[i]))
            resp_cf.append(temp)
    
        for band in self.bw_label:
            temp = []
            for i, p in enumerate(para_band):
                if p == band:
                    temp.append(Psth.baseline(resp[i]))
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
                    if plot:
                        plt.show()
                    plt.clf()
                    plt.close(fig)
                    
                if plot:
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
                    if plot:
                        plt.show()
                    plt.clf()
                    plt.close(fig)
                    
                if plot:
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
                    if plot:
                        plt.show()
                    plt.clf()
                    plt.close(fig)
                    
                if plot:
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
                    if plot:
                        plt.show()
                    plt.clf()
                    plt.close(fig)
                    
                if plot:
                    plt.show()
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
                    if plot:
                        plt.show()
                    plt.clf()
                    plt.close(fig)
                    
                if plot:
                    plt.show()
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
                    if plot:
                        plt.show()
                    plt.clf()
                    plt.close(fig)
                    
                if plot:
                    plt.show()
                    plt.close(fig)
                
            
        #return resp grouped by parameters
        return {'modrate':resp_mod, 'centerfreq':resp_cf,
                'bandwidth':resp_band}
    
    
# =============================================================================
#     def psth_correlation(self, saveplot=False):
#         psth_a = Psth.psth_all(self)
#         psth_p = Psth.psth_para(self)
#         
#         '''coeff'''
#         mod_coeff, cf_coeff, bw_coeff = [],[],[]
#      
#         mod = psth_p['modrate']
#         for i in range(len(mod)):
#             psth_mod = np.mean(mod[i], axis=0)    
#             mod_coeff.append(stats.pearsonr(psth_a, psth_mod)[0])
#         plt.plot(mod_coeff)
#         plt.xticks(list(range(len(mod))), self.mod_label)
#         ax = plt.subplot()
#         txt = (f'{self.filename}_Modrate_coeff')
#         ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
#         if saveplot:
#             plt.savefig(f'{self.filename}_PSTH_ModRate_Coeff.png', dpi=500)
#             plt.clf()
#         else:
#             plt.show()
#         
#         cf = psth_p['centerfreq']
#         for i in range(len(cf)):
#             psth_cf = np.mean(cf[i], axis=0)
#             cf_coeff.append(stats.pearsonr(psth_a, psth_cf)[0])
#         plt.plot(cf_coeff)
#         plt.xticks(list(range(len(cf))), self.cf_label)
#         plt.xticks(rotation = 45)
#         ax = plt.subplot()
#         txt = (f'{self.filename}_centerfreq_coeff')
#         ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
#         if saveplot:
#             plt.savefig(f'{self.filename}_PSTH_CenterFreq_Coeff.png', dpi=500)
#             plt.clf()
#         else:
#             plt.show()
#         
#         bw = psth_p['bandwidth']
#         for i in range(len(bw)):
#             psth_bw = np.mean(bw[i], axis=0)
#             bw_coeff.append(stats.pearsonr(psth_a, psth_bw)[0])
#             plt.show()
#         plt.plot(bw_coeff)
#         plt.xticks(list(range(len(bw))), self.bw_label)
#         plt.xticks(rotation = 45)
#         ax = plt.subplot()
#         txt = (f'{self.filename}_bandwidth_coeff')
#         ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
#         if saveplot:
#             plt.savefig(f'{self.filename}_PSTH_BandWidth_Coeff.png', dpi=500)
#             plt.clf()
#         else:
#             plt.show()
# =============================================================================
            
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
                
# =============================================================================
#             if arange==(1,0,2):
#                 return samegroup
#             else:
#                 pass    
# =============================================================================
        

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




class Psth_():
    def __init__(self, resp, para, filename):
        #exclude carrier less than 3kHz and puretone
        p_t, r_t = [],[]
        for i,p in enumerate(para):
            if p[0] <= 3.0:
                pass
            elif p[2] == 0.0:
                pass
            else:
                p_t.append(p)
                r_t.append(resp[i])
        
        self.resp = r_t
        self.para = p_t
        self.filename = filename
        _para = np.swapaxes(np.array(self.para),0,1)
        self.mod_label = sorted(set(_para[2][:]))
        self.cf_label = sorted(set(_para[0][:]))
        self.bw_label = sorted(set(_para[1][:]))
        self.features = pd.DataFrame()
        
    """reutrn *100 to switch from LabView volt to real mV scale"""
    def baseline(resp_iter):    #correct baseline
        return (resp_iter - np.mean(resp_iter[:50*25]))*100
    
    def baseline_zero(resp_iter):   #fix sound onset to zero
        return (resp_iter - resp_iter[50*25])*100
    
    def psth_all(self, plot=True, saveplot=False):
        """
        Generates PSTH using all lsfm response.

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
        resp_base = np.apply_along_axis(Psth.baseline, 1, resp)
        
        y = np.mean(resp_base, axis=0)
        x = np.arange(0,len(y))
        err = stats.sem(resp_base, axis=0)
        
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
            if plot:
                plt.show()
            plt.clf()
            plt.close(fig)
        
        if plot:
            plt.show()
            plt.clf()
            plt.close(fig)
            
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
            if plot:
                plt.show()
            plt.clf()
            plt.close(fig)
        
        if plot:
            plt.show()
            plt.clf()
            plt.close(fig)
                
        return x,y,err   
    
    def psth_para(self, plot=False, saveplot=False) -> dict:
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
            for i, p in enumerate(para_mod):
                if p == mod:
                    temp.append(Psth.baseline(resp[i]))     #resp with same mod_rate
            resp_mod.append(temp)       #resp seperated by mod_rate
    
        for cf in  self.cf_label:
            temp = []
            for i, p in enumerate(para_cf):
                if p == cf:
                    temp.append(Psth.baseline(resp[i]))
            resp_cf.append(temp)
    
        for band in self.bw_label:
            temp = []
            for i, p in enumerate(para_band):
                if p == band:
                    temp.append(Psth.baseline(resp[i]))
            resp_band.append(temp)
            
        
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
            ax.set_title(f'{self.filename}-mod {self.mod_label[i]} Hz')
            ax.set_xlabel('time (sec)')
            ax.set_ylabel('average response (mV)')
            
            if saveplot:
                plt.savefig(f'{self.filename}-mod {self.mod_label[i]} Hz.png', dpi=500, bbox_inches='tight')
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
                
            if plot:
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
            ax.set_title(f'{self.filename}-cf {self.cf_label[i]} Hz')
            ax.set_xlabel('time (sec)')
            ax.set_ylabel('average response (mV)')
            
            if saveplot:
                plt.savefig(f'{self.filename}-cf {self.cf_label[i]} kHz.png', dpi=500, bbox_inches='tight')
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
                
            if plot:
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
            ax.set_title(f'{self.filename}-bdwidth {self.bw_label[i]} kHz')
            ax.set_xlabel('time (sec)')
            ax.set_ylabel('average response (mV)')
            
            
            if saveplot:
                plt.savefig(f'{self.filename}-bdwidth {self.bw_label[i]} kHz.png', dpi=500, bbox_inches='tight')
                if plot:
                    plt.show()
                plt.clf()
                plt.close(fig)
                
            if plot:
                plt.show()
                plt.clf()
                plt.close(fig)
            
        #return resp grouped by parameters
        return {'modrate':resp_mod, 'centerfreq':resp_cf,
                'bandwidth':resp_band}
    
    
    def psth_correlation(self, saveplot=False):
        psth_a = Psth.psth_all(self)
        psth_p = Psth.psth_para(self)
        
        '''coeff'''
        mod_coeff, cf_coeff, bw_coeff = [],[],[]
     
        mod = psth_p['modrate']
        for i in range(len(mod)):
            psth_mod = np.mean(mod[i], axis=0)    
            mod_coeff.append(stats.pearsonr(psth_a, psth_mod)[0])
        plt.plot(mod_coeff)
        plt.xticks(list(range(len(mod))), self.mod_label)
        ax = plt.subplot()
        txt = (f'{self.filename}_Modrate_coeff')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}_PSTH_ModRate_Coeff.png', dpi=500)
            plt.clf()
        else:
            plt.show()
        
        cf = psth_p['centerfreq']
        for i in range(len(cf)):
            psth_cf = np.mean(cf[i], axis=0)
            cf_coeff.append(stats.pearsonr(psth_a, psth_cf)[0])
        plt.plot(cf_coeff)
        plt.xticks(list(range(len(cf))), self.cf_label)
        plt.xticks(rotation = 45)
        ax = plt.subplot()
        txt = (f'{self.filename}_centerfreq_coeff')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}_PSTH_CenterFreq_Coeff.png', dpi=500)
            plt.clf()
        else:
            plt.show()
        
        bw = psth_p['bandwidth']
        for i in range(len(bw)):
            psth_bw = np.mean(bw[i], axis=0)
            bw_coeff.append(stats.pearsonr(psth_a, psth_bw)[0])
            plt.show()
        plt.plot(bw_coeff)
        plt.xticks(list(range(len(bw))), self.bw_label)
        plt.xticks(rotation = 45)
        ax = plt.subplot()
        txt = (f'{self.filename}_bandwidth_coeff')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}_PSTH_BandWidth_Coeff.png', dpi=500)
            plt.clf()
        else:
            plt.show()
            
    def psth_trend(self, tuning=None, saveplot=False, **kwargs) -> None:
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
                                
                            set_location = kwargs.get('location')
                            if(set_location == 'onset'):
                                _resp = _resp[0:10000]
                            elif(set_location == 'second'):
                                _resp = _resp[10000:20000]
                            elif(set_location == 'plateau'):
                                _resp = _resp[20000:40000]
                            elif(set_location == 'offset'):
                                _resp = _resp[40000:]
                           
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
            
            for i,gp in enumerate(samegroup):          
                x,y,err=[],[],[]
                for ii in gp:
                    x.append(ii[1])
                    y.append(ii[2])
                    err.append(ii[3])
                try:                        
                    plt.errorbar(x,y,yerr=err, color=colors[i], capsize=(4), marker='o', label=f'{group}-{gp[0][0]}')
                except IndexError:
                    pass
                
                if set_window:
                    txt = '_window'+str(set_window)
                else:
                    txt='_all'
                
                ax = plt.subplot()
                ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
                plt.xscale('symlog')                
                plt.xlabel(f'{base}')
                plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            
            if saveplot:
                plt.savefig(f'{self.filename}_{group}-{base}{txt}.png', \
                            dpi=500, bbox_inches='tight')
                plt.clf()
                plt.close()
            else:
                plt.show()
                
# =============================================================================
#             if arange==(1,0,2):
#                 return samegroup
#             else:
#                 pass
# =============================================================================
        

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
            
            
            
def psth_wwo_bf(resp, para, bf, version, filename, plot=True, saveplot=False):
    '''
    plot PSTH using stim with any crossing with bf and without. 

    Parameters
    ----------
    resp : TYPE
        DESCRIPTION.
    para : TYPE
        DESCRIPTION.
    bf : TYPE
        DESCRIPTION.
    version : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    saveplot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    
    resp_in, resp_ex, para_in, para_ex = lsfm.resp_bf_or_not(resp, para, bf)
    p1 = Psth(resp_in, para_in, filename)
    p2 = Psth(resp_ex, para_ex, filename)
    
    if version == 1:
        
        x1,y1,err1 = p1.psth_all(plot=False, saveplot=False)
        x2,y2,err2 = p2.psth_all(plot=False, saveplot=False)
        
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
        ax.set_title(f'{filename}_PSTH_BF', fontsize=14)
        ax.set_xlabel('time (sec)', fontsize=16)
        ax.set_ylabel('average response (mV)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper left', fontsize=14)
        
        if saveplot:
            plt.savefig(f'{filename}_PSTH_BF.png', dpi=500, bbox_inches='tight')
            plt.savefig(f'{filename}_PSTH_BF.pdf', dpi=500, format='pdf', bbox_inches='tight')
            if plot:
                plt.show()
            plt.clf()
            plt.close(fig)
        if plot:
            plt.show()
            plt.clf()
            plt.close(fig)
    
    elif version == 2:
    
        x1,y1,err1 = p1.psth_all(plot=False, saveplot=False)
        x2,y2,err2 = p2.psth_all(plot=False, saveplot=False)
        #resp_by_para = p.psth_para(plot=True, saveplot=False)
        #p.psth_trend(saveplot=False)
        
        fig, ax = plt.subplots()
        ax.plot(x1,y1,color='midnightblue', label='w/_bf')
        ax.fill_between(x1, y1+err1, y1-err1, color='cornflowerblue', alpha=0.6)
        ax.plot(x2,y2,color='firebrick', label='w/o_bf')
        ax.fill_between(x2, y2+err2, y2-err2, color='salmon', alpha=0.6)
        
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [1250,26250]]
        ax.set_xlim(0,len(x1))
        label = list(np.round(np.linspace(0, 1.5, 16), 2))
        ax.set_xticks(np.linspace(0,37500,16))
        ax.set_xticklabels(label, rotation = 45)
        #ax.xticks(rotation = 45)
        ax.set_title(f'{filename}_PSTH_BF', fontsize=14)
        ax.set_xlabel('time (sec)', fontsize=16)
        ax.set_ylabel('average response (mV)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if saveplot:
            plt.savefig(f'{filename}_PSTH_BF.png', dpi=500, bbox_inches='tight')
            plt.savefig(f'{filename}_PSTH_BF.pdf', dpi=500, format='pdf', bbox_inches='tight')
            if plot:
                plt.show()
            plt.clf()
            plt.close(fig)
        if plot:
            plt.show()
            plt.clf()
            plt.close(fig)