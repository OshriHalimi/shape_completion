%Author: James W. Ryland
%June 14, 2012

function [ updateVolumeCAHandle getCAVHandle] = PreviewBox(fig, pos, title, externalUpdateHandle, initCAV)
%PREVIEWBOX this function gives a full 3d rotatable rendering preview
%   fig is the parent figure or panel, pos is the position inside the
%   parent figure that PreviewBox will occupy. externalUpdateHandle updates
%   tells an external function that the refresh and apply button has been
%   pressed. initCAV is a structure containing initilaztion perameters.
    
    volumeCA = [];
    if ~isempty(initCAV)
        volumeCA = initCAV;
    else
        %test data (Shaded For Convenience)
        volume = uint8(zeros(20, 20, 20));
        %e = fspecial3('ellipsoid', [ 3 3 3]);
        %volume = convn(volume, e, 'same');
        %volume = uint8(smooth3(volume));
        volumeCA(:,:,:,1) = volume;
        volumeCA(:,:,:,2) = volume;
        volumeCA(:,:,:,3) = volume;
        volumeCA(:,:,:,4) = volume;
    end
        
    maxDimIndPrevious = 0;
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end

    % initialize buttons
    previewPanel = uipanel('Parent', fig, 'Title', title, 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 280 ],...
        'DeleteFcn', @previewPanel_DeleteFcn);
    
    refreshButton = uicontrol('Parent', previewPanel, 'Style', 'PushButton', 'String', 'Apply & Refresh', 'Position', [10 240 260 20],...
        'CallBack', @refreshButton_CallBack);
    
    %Cross Platform Formating
    uicomponents = [ previewPanel refreshButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    
    updateVolumeCAHandle = @updateVolumeCA;
    getCAVHandle = @getCAV;
    
    
    % initialize view
    viewAxis = axes('Parent', previewPanel, 'Units', 'pixels', 'Position', [ 10 10 260 230 ]);
    axes(viewAxis);
    
    % initialize rotation
    [slices1 slices2 slices3] = volumeRender(volumeCA, [], [], [], viewAxis);
    rot3d = rotate3d(viewAxis);
    set(rot3d, 'Enable', 'on');
    setAllowAxesRotate(rot3d,viewAxis,true);
    set(rot3d, 'ActionPostCallBack', @rot3d_ActionPostCallBack);
    %axis('off');
    
    % kickstart visibility
    rot3d_ActionPostCallBack([], []);
    set(viewAxis, 'Color', 'black');
    
    %CallBacks
    function refreshButton_CallBack(h, EventData)
        set(refreshButton, 'Enable', 'off');
        pause(.05);
        externalUpdate();
        axes(viewAxis);
        cla(viewAxis);
        [slices1 slices2 slices3] = volumeRender(volumeCA, [], [], [], viewAxis);
        rot3d = rotate3d(viewAxis);
        set(rot3d, 'Enable', 'on');
        setAllowAxesRotate(rot3d,viewAxis,true);
        set(rot3d, 'ActionPostCallBack', @rot3d_ActionPostCallBack);
        %axis('off');
        maxDimIndPrevious = 0;
        rot3d_ActionPostCallBack([], []);
        set(viewAxis, 'Color', 'black');
        pause(.05);
        set(refreshButton, 'Enable', 'on');
    end

    function rot3d_ActionPostCallBack(h, EventData)
        % x=s3 y=s2 z=s3
        relPos = campos(viewAxis)-camtarget(viewAxis);
    
        maxDim = abs(relPos)==max(abs(relPos));
        maxDimInd = find(maxDim);
        
        if maxDimIndPrevious~=maxDimInd(1)
            switch maxDimInd(1)
                case 1
                    set(slices3, 'Visible','on');
                    set(slices2, 'Visible','off');
                    set(slices1, 'Visible','off');
                case 2
                    set(slices3, 'Visible','off');
                    set(slices2, 'Visible','on');
                    set(slices1, 'Visible','off');
                case 3
                    set(slices3, 'Visible','off');
                    set(slices2, 'Visible','off');
                    set(slices1, 'Visible','on');
            end
        end
        maxDimIndPrevious = maxDimInd(1);
    end

    function previewPanel_DeleteFcn(h, EventData)
        clear('slices3');
        clear('slices2');
        clear('slices1');
        clear('volumeCA');
    end

    %update functions
    function updateVolumeCA(newVolumeCA)
        volumeCA = newVolumeCA;
    end

    function externalUpdate()
        externalUpdateHandle();
    end

    %getters and Setters
    function [CAV] = getCAV()
        CAV = volumeCA;
    end

end
