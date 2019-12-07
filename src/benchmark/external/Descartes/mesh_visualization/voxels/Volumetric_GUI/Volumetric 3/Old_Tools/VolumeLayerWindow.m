%Author: James W. Ryland
%June 14, 2012

function [ closeWindowHandle] = VolumeLayerWindow(pos, externalUpdateHandle, settings)
%VOLUMELAYERWINDOW creates a window used to create filled and colored
%volumes.
%  This window uses, sourceBox, colorBox, and previewBox. Using the source
%  box the user will select a bianary volume to be edited using the color
%  box. Position sets the position the window will attemp to place itself.
%  the externalUdateHandle is a function that updates the layerBox that
%  created this VolumeLayerWindow with a new ColorAlphaVolume (CAV or CA)
%  to be merged with the other CAV's to form the final rendering. 
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
    
    fig = figure('Name', 'Volume Layer Properties', 'Resize', 'off', 'NumberTitle', 'off','Position', [pos(1) pos(2) 280 570],...
        'CloseRequestFcn', @fig_CloseRequestFcn);
    
    figureAdjust(fig);
    
    % defined in reverse order
    
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
            selectedVol = volume*255;
            SC = uint8(smooth3(double(selectedVol), 'box', 3));
            %filt = fspecial3('average', 3);
            %SC = convn(double(selectedVol), filt, 'same');
            
            CA = getCA();
            volumeCA(:,:,:,1) = uint8(SC*CA(1));
            volumeCA(:,:,:,2) = uint8(SC*CA(2));
            volumeCA(:,:,:,3) = uint8(SC*CA(3));
            volumeCA(:,:,:,4) = uint8(power(int16(SC),2)/power(255,1)*CA(4));
            
            clear('SC');
        end
        
        updateVolumeCAHandle(uint8(volumeCA));
        disp('volumeCAPrepare');
        externalUpdate();
    end

    function source2color(selectionVolume, sourceVolume)
        volume = uint8(selectionVolume);
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
