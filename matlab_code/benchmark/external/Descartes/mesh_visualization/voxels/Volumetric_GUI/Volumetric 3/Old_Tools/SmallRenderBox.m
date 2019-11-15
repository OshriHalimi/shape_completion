%Author: James W. Ryland
%June 14, 2012

function [ updateVolumeCAHandle ] = SmallRenderBox(pos, title, externalUpdateHandle )
%PREVIEWBOX give the user a resizable rotateable 3d rendering of a CAV
%volume that can be refreshed to update with changes.
%   Pos is the position on the desktop that SmallRenderBox will occupy.
%   ExternalUpdateHandle is not used and can be empty.
    
    %test data (Shaded For Convenience)
    volume = ones(20,20,20,'uint8');
    %volume = rand(100, 100, 100);
    %e = fspecial3('ellipsoid', [ 5 5 5]);
    %volume = convn(volume, e, 'same');
    %volume = volume>.525;
    %volume = smooth3(volume);
    volumeCA(:,:,:,1) = volume;
    volumeCA(:,:,:,2) = volume;
    volumeCA(:,:,:,3) = volume;
    volumeCA(:,:,:,4) = volume;
    
    maxDimIndPrevious = 1;
    
    
    if isempty(pos)
        pos = [0 0];
    end
    
    fig = figure('Name', title, 'NumberTitle', 'off','Position', [pos(1) pos(2) 400 400],...
        'ResizeFcn', @fig_ResizeFcn, 'CloseRequestFcn', @fig_CloseRequestFcn);
    
    figureAdjust(fig);
    
    figPos = get(fig, 'Position');
    
    % initialize buttons
    refreshButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Refresh', 'Units', 'pixels', 'Position', [1 (figPos(4)-20) figPos(3) 20],...
        'CallBack', @refreshButton_CallBack);
    
    %Cross Platform Formating
    uicomponents = [refreshButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    
    updateVolumeCAHandle = @updateVolumeCA;
    refreshButtonHandle = @refreshButton_CallBack;
    
    % initialize view
    viewAxis = axes('Parent', fig, 'Units', 'Pixels', 'OuterPosition', [ 1 1 figPos(3) figPos(4)-20], 'ActivePositionProperty', 'OuterPosition');
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
        set(refreshButton,'Enable', 'off');
        pause(0.05);
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
        set(refreshButton,'Enable', 'on');
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

    function fig_CloseRequestFcn(h, EventData)
        clear('volumeCA');
        clear('slices1');
        clear('slices2');
        clear('slices3');
        delete(fig);
    end

    function fig_ResizeFcn(h, EventData)
        cla(viewAxis);
        set(viewAxis, 'Color', 'black');
        figr = gcbo;
        figPos = get(figr, 'Position');
        set(refreshButton, 'Position', [1 (figPos(4)-20) figPos(3) 20]);
        set(viewAxis, 'Position',[ 1 1 figPos(3) figPos(4)-20]);
    end

    %update functions
    function updateVolumeCA(newVolumeCA)
        volumeCA = uint8(newVolumeCA);
    end

    function externalUpdate()
        externalUpdateHandle();
    end

end
