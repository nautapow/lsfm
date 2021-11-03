wdir = pwd;
cd ..
mdir = pwd;
cd (wdir)

list = dir;
names = {list.name};
names = names(~ismember(names,{'.','..'}));
for i = 1:length(names)
    load(names{i});
    names{i}
    stim = stim(~cellfun('isempty', stim));

    [~, f] = cwt(stim{1,1},200000, 'WaveletParameters',[32,1280]);

    for x=1:length(stim)
        wt{x} = cwt(stim{1,x},200000, 'WaveletParameters',[32,1280]);
        wt{x} = abs(wt{x});

        for y = 1:length(f)
            wt_t(y,:) = decimate(wt{x}(y,:), 800);
        end

        wt{x}=wt_t;
        x
    end

    str = append(names{i}(1:12), '_cwt.mat');
    sdir = append(mdir,'/cwt_sound/');
    save([(sdir), str], 'f', 'wt')
    
    clearvars -except list names wdir mdir
    
end