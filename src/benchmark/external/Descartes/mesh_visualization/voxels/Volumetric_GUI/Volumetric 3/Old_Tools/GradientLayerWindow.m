%Author: James W. Ryland
%June 14, 2012

function [ closeWindowHandle ] = GradientLayerWindow(pos, externalUpdateHandle, settings)
%VOLUMELAYERWINDOW allows a user to map a graded volume to RGBA space and
%output it to a CAV representation.
%   Pos is the position on the screen that GradientLayerWindow will occupy.
%   ExternalUpdateHandle updates the LayerBox that called
%   GradientLayerWindow. Settings is a structure containing initialization
%   parameters, but its fine to call this function with setting empty [].
    
    if isempty(pos)
        pos = [0 0];
    end
    
    previewBoxCAV = [];
    mapBoxSettings = [];
    sourceSettings = {};
    
    if ~isempty(settings)
        previewBoxCAV = settings.previewBoxCAV;
        mapBoxSettings = settings.mapBoxSettings;
        sourceSettings = settings.sourceSettings;
    end

    volume = [];
    volumeCA = [];
    
    
    fig = figure('Name', 'Gradient Layer Properties', 'Resize', 'off', 'NumberTitle', 'off','Position', [pos(1) pos(2) 280 570],...
        'CloseRequestFcn', @fig_CloseRequestFcn);
    
    figureAdjust(fig);
    
    % defined in reverse order
    
    volumeCAPrepareHandle = @volumeCAPrepare;
    
    mapBoxUpdate = [];
    
    [updateVolumeCAHandle getCAV] = PreviewBox(fig, [0 0],'Preview', volumeCAPrepareHandle, previewBoxCAV);
    
    
    %filterBoxUpdate = FilterBox(fig, [0 430], 'FilterBox', @filter2map);
    
    [getSourceSettings getBoundsHandle] = SourceBox(fig, 'Source', [0 430], @source2color, sourceSettings);
    
    [mapBoxUpdate getMapFn getMapSettings] = MapBox(fig,[0 280],[], getBoundsHandle, mapBoxSettings);
    
    closeWindowHandle = @closeWindow;
    
    function volumeCAPrepare()
        volumeCA = [];
        if ~isempty(volume)
            %[selectedVol dum1 dum2 dum3] = Crop3(volume);
            %selectedVol = volume;
            mapFn = getMapFn();
            
            maxV = max(max(max(volume)))
            minV = min(min(min(volume)))
            
            s = round(size(volume,1)/2);
            
            
            %[demo dum dum] = meshgrid(1:100, 1:20, 1:20);

            %mappedPut = mapFn((demo/100)*(maxV-minV)+minV);
            
            mappedPut = mapFn(volume(:,:,:));
            
            %min(min(min(max(mappedPut(:,:,:,1:4)*255))));
            %max(max(max(max(mappedPut(:,:,:,1:4)*255))));
            
            %figure();
            %im1(:,:,1) = squeeze(volume(s,:,:)/(maxV-minV)+minV);
            %im1(:,:,2) = im1(:,:,1);
            %im1(:,:,3) = im1(:,:,1);
            %im2(:,:,1) = squeeze(mappedPut(s,:,:,1));
            %im2(:,:,2) = im2(:,:,1);
            %im2(:,:,3) = im2(:,:,1);
            %subplot(1,2,1); image(im1);
            %subplot(1,2,2); image(im2);
            
            volumeCA = uint8(mappedPut*255);
            
            %TEST COLOR MAPPING
            
            
        end
        updateVolumeCAHandle(volumeCA);
        disp('volumeCAPrepare');
        externalUpdate();
    end

    function source2color(selectionVolume, sourceVolume)
        volume = double(selectionVolume).*double(sourceVolume);
        if ~isempty(mapBoxUpdate)
            mapBoxUpdate(volume);
        end
        disp('filter2Color');
    end

    %function filter2map(newVolume)
    %    volume = newVolume;
    %    mapBoxUpdate(newVolume);
    %    disp('filter2map');
    %end

    %function source2filter(selectionVolume, sourceVolume)
    %    filterBoxUpdate(double(selectionVolume).*double(sourceVolume));
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
        clear('previewBoxCAV');
        clear('mapBoxSettings');
        clear('sourceSettings');
        clear('settings');
        
    end

    % settings related functions
    function updateSettings()
        settings.previewBoxCAV = getCAV(); 
        settings.mapBoxSettings = getMapSettings();
        settings.sourceSettings = getSourceSettings();
    end

    function [set] = getSettings()
        set = settings;
    end
end
