%stim = stim(~cellfun('isempty', stim));
[~, f] = cwt(stim(1,:),200000, 'WaveletParameters',[32,1280]);

parfor x=1:length(stim(:,1))
    wt{x} = abs(cwt(stim(x,:),200000, 'WaveletParameters',[32,1280]));
    
    wt_t = zeros(length(f),500)
    
    for y = 1:length(f)
        wt_t(y,:) = decimate(wt{x}(y,:), 800);
    end

    wt{x}=wt_t;
    x
end



save('cwt_sound.mat', 'f', 'wt')