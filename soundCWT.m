tone = false;
lsFM = true;

if tone
    filename='Z:\Users\cwchiang\Sound_woFIR\puretone_Sound.tdms'
    data = tdmsread(filename);
    dataCell = data{1};
    signal_tone = double(dataCell.SoundO);
    Fs=200000;
    [wt,f] = cwt(signal_tone(5e6:7.5e6),200000,'FrequencyLimits',[3000 96000],'WaveletParameters',[32,32*40],'VoicesPerOctave',48);

    save('tone_noFIR.mat', 'f', 'wt', '-v7.3')
    
    % Time vector for the segment
    t = (0:length(signal_tone(5e6:7.5e6))-1) / 200000;  % in seconds
    
    figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.08], 'Color', 'w');
    imagesc(t, f/1000, abs(wt))
    axis xy
    colormap(flipud(gray))
    set(gca, 'YScale', 'log')
    set(gcf, 'Renderer', 'painters')

    % Adjust contrast to avoid clipping (based on 1st to 99th percentile)
    m = abs(wt);
    clim = prctile(m(:), [1 99]);
    caxis(clim)
    
    xlabel('Time (s)')
    ylabel('Frequency (kHz)')
    title('CWT Magnitude (Inverted Greyscale)')
    yticks([3, 6, 12, 24, 48, 96])
    colorbar off
    
    print('tone_cwt', '-depsc2', '-r500')
    
    
    
    % % Step 2: Downsample the CWT in time (columns)
    % ds_factor = 10;
    % wt_ds = wt(:, 1:ds_factor:end);  % keep every 8th column
    % 
    % % Step 3: Adjust time vector accordingly
    % t_full = (0:length(signal(7.05e6:9.2e6))-1) / 200000;
    % t = t_full(1:ds_factor:end);
    % 
    % % Step 4: Plot
    % figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.08], 'Color', 'w');
    % imagesc(t, f/1000, abs(wt_ds))
    % axis xy
    % colormap(flipud(gray))
    % 
    % % Step 5: Adjust contrast
    % m = abs(wt_ds);
    % clim = prctile(m(:), [2 98]);  % Adjust percentile if needed
    % caxis(clim)
    % 
    % xlabel('Time (s)')
    % ylabel('Frequency (kHz)')
    % title('CWT Magnitude (Downsampled in Time ×8)')
    % yticks([3, 12, 24, 48, 96])
    % colorbar off
    % 
    % % Step 6: Save figure at 500 DPI
    % print('cwt_timeslice_downsampled', '-dpng', '-r500')
end


if lsFM
    filename='Z:\Users\cwchiang\Sound_woFIR\lsFM_Sound.tdms'
    data = tdmsread(filename);
    dataCell = data{1};
    signal = double(dataCell.SoundO);
    Fs=200000;
    
    % fir = load('FIR_20230907.txt');
    % %target = [1; zeros(length(fir)-1, 1)];
    % %h_inv = deconv(target, fir);  % Find inverse FIR
    % %signal_unfiltered = filter(h_inv, 1, signal);
    % 
    % N = length(signal) + length(fir) - 1;
    % B = fft(fir, N);
    % S = fft(signal, N);
    % 
    % % Avoid dividing by very small values (numerical stability)
    % B_inv = 1 ./ B;
    % B_inv(abs(B) < 1e-6) = 0;
    % 
    % S_inv = S .* B_inv;
    % 
    % signal_unfiltered = real(ifft(S_inv));
    % signal_unfiltered = signal_unfiltered(1:length(signal));  % Crop to original length
    
    
    %cwt(signal(10e6:18e6),200000,'FrequencyLimits',[3000 96000],'WaveletParameters',[32,32*40],'VoicesPerOctave',48);
    [wt,f] = cwt(signal(7.35e6:9.2e6),200000,'FrequencyLimits',[3000 96000],'WaveletParameters',[32,32*40],'VoicesPerOctave',48);
    
    
    save('lsfm_noFIR.mat', 'f', 'wt', '-v7.3')
    
    
    % Time vector for the segment
    t = (0:length(signal(7.35e6:9.2e6))-1) / 200000;  % in seconds
    
    figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.08], 'Color', 'w');
    imagesc(t, f/1000, abs(wt))
    axis xy
    colormap(flipud(gray))

    % Adjust contrast to avoid clipping (based on 1st to 99th percentile)
    m = abs(wt);
    clim = prctile(m(:), [1 99]);
    caxis(clim)
    
    xlabel('Time (s)')
    ylabel('Frequency (kHz)')
    title('CWT Magnitude (Inverted Greyscale)')
    yticks([3, 12, 24, 48, 96])
    colorbar off
    
    print('cwt_inverted_greyscale_contrast', '-dpng', '-r300')
    
    
    
    % Step 2: Downsample the CWT in time (columns)
    ds_factor = 2000;
    wt_ds = wt(:, 1:ds_factor:end);  % keep every 8th column
    
    % Step 3: Adjust time vector accordingly
    t_full = (0:length(signal(7.35e6:9.2e6))-1) / 200000;
    t = t_full(1:ds_factor:end);
    
    % Step 4: Plot
    figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.08], 'Color', 'w');
    imagesc(t, f/1000, abs(wt_ds))
    axis xy
    colormap(flipud(gray))
    set(gca, 'YScale', 'log')
    %set(gcf, 'Renderer', 'painters')


    % Step 5: Adjust contrast
    m = abs(wt_ds);
    clim = prctile(m(:), [2 98]);  % Adjust percentile if needed
    caxis(clim)
    
    xlabel('Time (s)')
    ylabel('Frequency (kHz)')
    title('CWT Magnitude (Downsampled in Time ×10)')
    yticks([3, 6, 12, 24, 48, 96])
    colorbar off
    
    % Step 6: Save figure at 500 DPI
    print('cwt_timeslice_downsampled', '-dpng', '-r300')
end

