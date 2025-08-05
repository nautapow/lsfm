# =============================================================================
#         """reverse FIR"""
#         target_FIR = f'E:\in-vivo_patch\FIR_list\FIR_{df["FIR"][df_loc]}.txt'
#         
#         with open(target_FIR, 'r') as file:
#                  fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#         sound_re = lsfm.inv_fir(sound, fir)
#         sound_re = t.cut(sound_re)
#         scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
# =============================================================================
        

        

# =============================================================================
#     ##invert fir to correct stimuli
#     with open('FIR_07_27_2021.txt', 'r') as file:
#         fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#     sound, _ = t.get_raw()
#     sound_re = lsfm.inv_fir(sound, fir)
#     sound_re = t.cut(sound_re)
#     scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
# =============================================================================