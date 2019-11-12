%Author: James W. Ryland
%June 14, 2012

function [ closeWindowHandle ] = ShellLayerWindow(pos, externalUpdateHandle, settings)
%VOLUMELAYERWINDOW allows the creation of a hollow shell around a region of
%volume.
%   pos is the position the ShellLayerWindow will occupy on the desktop.
%   ExternalUpdateHandle is a function handle that updates the parent Layer
%   Box with the CAV representation of the shell that is created.
%   ShellLayerWindow also makes use of SourceBox, ColorBox, and PreviewBox.

    if isempty(pos)
        pos = [0 0];
    end

    previewBoxCAV = [];
    colorBoxCA = [];
    sourceSettings = {};
    
    if ~isempty(settings)
        previewBoxCAV = settings.previewBoxCAV;
        colorBoxCA = settings.colorBoxCA;
        sourceSettings = settings.sourceSettings;
    end
    
    volume = [];
    volumeCA = [];
    
    fig = figure('Name', 'Shell Layer Properties', 'Resize', 'off', 'NumberTitle', 'off','Position', [pos(1) pos(2) 280 570],...
        'CloseRequestFcn', @fig_CloseRequestFcn);
    
    figureAdjust(fig);
    
    % defined boxes in reverse order
    
    volumeCAPrepareHandle = @volumeCAPrepare;
    
    [updateVolumeCAHandle getCAV] = PreviewBox(fig, [0 0],'Preview', volumeCAPrepareHandle, previewBoxCAV);
    
    [colorBoxUpdate getCA] = ColorBox(fig, 'Color', [0 280], [], colorBoxCA);
    
    %filterBoxUpdate = FilterBox(fig, [0 430], 'FilterBox', @filter2color);
    
    [getSourceSettings] = SourceBox(fig, 'Source', [0 430], @source2color, sourceSettings);
    
    closeWindowHandle = @closeWindow;
    
    function volumeCAPrepare()
        volumeCA = [];
        if ~isempty(volume)
            %[selectedVol dum1 dum2 dum3] = Crop3(volume);
            selectedVol = uint8(volume);
            SCVol = smooth3(single(selectedVol), 'box', 5);
            SCShell = uint8(4*(SCVol).*(1-SCVol)*255);
            clear('SCVol');
            CA = getCA();
            
            max(max(max(selectedVol)))
            
            max(max(max(SCShell)))
            
            %all = ones(size(SCShell));
            volumeCA(:,:,:,1) = uint8(SCShell*CA(1));
            volumeCA(:,:,:,2) = uint8(SCShell*CA(2));
            volumeCA(:,:,:,3) = uint8(SCShell*CA(3));
            volumeCA(:,:,:,4) = uint8(power(double(SCShell)/255,4)*CA(4)*255);
            
            clear('SCShell');
        end
        updateVolumeCAHandle(uint8(volumeCA));
        disp('volumeCAPrepare');
        externalUpdate();
    end

    function source2color(selectionVolume, sourceVolume)
        volume = selectionVolume;
        colorBoxUpdate(volume);
        disp('filter2Color');
    end

    %function filter2color(newVolume)
    %    volume = newVolume;
    %    colorBoxUpdate(newVolume);
    %    disp('filter2color');
    %end

    %function source2filter(selectionVolume, sourceVolume)
    %    filterBoxUpdate(selectionVolume);
    %end
    
    function externalUpdate()
        if ~isempty(volumeCA)
            if ~isempty(externalUpdateHandle)
                updateSettings();
                externalUpdateHandle(volumeCA, settings);
            end
        end
    end
    
    function closeWindow()
        fig_CloseRequestFcn([],[]);
    end

    %call backs
    function fig_CloseRequestFcn(h, EventData)
        if ~isempty(fig)
            delete(fig);
            fig = [];
        end
        clear('volumeCA');
        clear('volume');
        clear('settings');
        clear('previewBoxCAV');
        clear('colorBoxCA');
        clear('sourceSettings');
        
    end

    % settings related functions
    function updateSettings()
        settings.previewBoxCAV = getCAV(); 
        settings.colorBoxCA = getCA();
        settings.sourceSettings = getSourceSettings();
    end

    function [set] = getSettings()
        set = settings;
    end
end
