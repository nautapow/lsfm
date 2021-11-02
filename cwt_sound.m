stim = stim(~cellfun('isempty', stim));

[~, f] = cwt(stim{1,1},200000, 'WaveletParameters',[32,1280]);

for x=1:length(stim)
    wt{x} = cwt(stim{1,x},200000, 'WaveletParameters',[32,1280]);
    wt{x} = abs(wt{x});
    
    for i = 1:length(f)
        wt_t(i,:) = decimate(wt{x}(i,:), 800);
    end
    
    wt{x}=wt_t
end

save('/Users/POW/Desktop/python_learning/sound_cwt_date.mat', 'f', 'wt')