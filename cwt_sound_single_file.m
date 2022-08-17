%stim = stim(~cellfun('isempty', stim));
[~, f] = cwt(stim(1,:), 200000, 'WaveletParameters',[24,960], 'VoicesPerOctave',24,'FrequencyLimits',[3000 96000]);

for x=1:length(stim(:,1))
    wt{x} = cwt(stim(x,:), 200000, 'WaveletParameters',[24,960], 'VoicesPerOctave',24,'FrequencyLimits',[3000 96000]);
    wt{x} = abs(wt{x});

    for y = 1:length(wt{1}(:,1))
        %wt_t(y,:) = decimate(wt{x}(y,:), 8);
        wt_t(y,:) = resample(wt{x}(y,:),1,8);
    end

    wt{x}=wt_t;
    x
end

save('cwt_sound.mat', 'f', 'wt'. '-v7.3');