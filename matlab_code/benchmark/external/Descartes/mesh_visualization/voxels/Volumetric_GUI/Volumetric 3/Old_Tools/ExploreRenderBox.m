%Author: James W. Ryland
%June 14, 2012

function [ updateVolumeCAHandle updateVisibleHandle drawBoundsHandle] = ExploreRenderBox(pos, title, externalUpdateHandle )
%EXPLORERENDERBOX makes a render window with the ability to be manipulated
%in more ways than the standard one.
%   POS is the position on the screen that ExploreRenderBox will occupy.
%   ExternalUpdateHandle updates external functions (Not Used Currently).
%   It returns several important function handles updateVisibleHandle and
%   drawBoundsHandle in particular. updateVisible changes which slices are
%   visible along each of the main axes. drawBoundsHanle tells the Render
%   Window to draw a bounding box or cross.
    
    %test data (Shaded For Convenience)
    %volume = rand(100, 100, 100);
    %e = fspecial3('ellipsoid', [ 5 5 5]);
    %volume = convn(volume, e, 'same');
    %volume = volume>.525;
    volume = zeros(20,20,20,'uint8');
   
    bounds = [];
    
    volumeCA(:,:,:,1) = volume;
    volumeCA(:,:,:,2) = volume;
    volumeCA(:,:,:,3) = volume;
    volumeCA(:,:,:,4) = volume;
    clear('volume');
    
    [sX sY sZ dum] = size(volumeCA);
    
    dimS = [1 1 1];
    dimE = [sX sY sZ];
    slices1 = [];
    slices2 = [];
    slices3 = [];
    
    maxDimIndPrevious = -1;
    viewBlanked = 0;
    changeVisible = 1;
    
    if isempty(pos)
        pos = [0 0];
    end
    
    loadedVariable = load('GlobalSettings.mat');
    GlobalSettings = loadedVariable.GlobalSettings;
    
    fig = figure('Name', title, 'NumberTitle', 'off','Position', [pos(1) pos(2) 400 400],...
        'ResizeFcn', @fig_ResizeFcn, 'Color', GlobalSettings.bgColor, 'CloseRequestFcn', @fig_CloseRequestFcn);
    
    
    %prevents background inversion when printing
    set(fig,'InvertHardcopy','off');
    
    figPos = get(fig, 'Position');
    
    figureAdjust(fig);
    
    % initialize buttons
    refreshButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Refresh', 'Units', 'pixels', 'Position', [1 (figPos(4)-20) figPos(3) 20],...
        'CallBack', @refreshButton_CallBack);
    
    alphaIcon = imread('AlphaIcon20.jpg');
    
    alphaLabel = uicontrol('Parent', fig, 'Style', 'PushButton', 'Units', 'pixels', 'Position', [0 0 20 20],...
        'CData', alphaIcon);
    
    alphaSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'String', 'alpha', 'Units', 'pixels', 'Position', [20 -3 figPos(3)-50 20],...
        'CallBack', @alphaSlider_CallBack, 'Max', 5, 'Min', 0, 'Value', 1, 'SliderStep', [.01 .01]);
    
    cameraImg = imread('camera30.jpg');
    
    captureButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'Units', 'pixels', 'Position', [figPos(3)-30 0 30 30], 'CData', cameraImg,...
        'CallBack', @captureButton_CallBack);
    
        
    %Cross Platform Formating
    uicomponents = [refreshButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    
    updateVolumeCAHandle = @updateVolumeCA;
    refreshButtonHandle = @refreshButton_CallBack;
    updateVisibleHandle = @updateVisible;
    drawBoundsHandle = @drawBounds;
    
    % initialize view
    viewAxis = axes('Parent', fig, 'Units', 'Pixels', 'OuterPosition', [ 1 20 figPos(3) figPos(4)-40], 'ActivePositionProperty', 'OuterPosition', 'Projection', 'Perspective');
    set(gcf,'PaperPositionMode','auto');
    set(gcf,'InvertHardcopy','off');
    axes(viewAxis);
    
    % initialize rotation
    rot3d = rotate3d(viewAxis);
    set(rot3d, 'Enable', 'on');
    setAllowAxesRotate(rot3d,viewAxis,true);
    set(rot3d, 'ActionPostCallBack', @rot3d_ActionPostCallBack);
    
    
    % kickstart visibility
    refreshButton_CallBack([],[]);
    
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
        viewBlanked = 1;
        rot3d_ActionPostCallBack([], []);
        set(viewAxis, 'Color', 'Black');
        pause(.05);
        set(refreshButton,'Enable', 'on');
        viewBlanked = 0;
        set(alphaSlider, 'Value', 1);
    end

    function rot3d_ActionPostCallBack(h, EventData)
        % x=s3 y=s2 z=s1
        relPos = campos(viewAxis)-camtarget(viewAxis);
    
        maxDim = abs(relPos)==max(abs(relPos));
        maxDimInd = find(maxDim);
        
        %disp(maxDimIndPrevious~=maxDimInd(1));
        %disp(maxDimInd(1));
        
        if (maxDimIndPrevious~=maxDimInd(1))||changeVisible
            switch maxDimInd(1)
                case 1
                    %disp('1');
                    size(dimS(1):dimE(1));
                    set(slices3, 'Visible','off');
                    set(slices2, 'Visible','off');
                    set(slices1, 'Visible','off');
                    
                    %FIGURE OUT WHY THIS DOESN"T WORK!!!! If it can be
                    %fixed there will be a marginal performance gain
                    %set(slices3(1:dimS(1)), 'Visible','off');
                    set(slices3(dimS(1):dimE(1)), 'Visible','on');
                    %set(slices3(dimS(1):sX), 'Visible','off');
                    
                case 2
                    %disp('2');
                    size(dimS(2):dimE(2));
                    set(slices3, 'Visible','off');
                    set(slices2, 'Visible','off');
                    set(slices1, 'Visible','off');
                    
                    %set(slices2(1:dimS(2)), 'Visible','off');
                    set(slices2(dimS(2):dimE(2)), 'Visible','on');
                    %set(slices2(dimS(2):sY), 'Visible','off');
                    
                case 3
                    %disp('3');
                    size(dimS(3):dimE(3));
                    set(slices3, 'Visible','off');
                    set(slices2, 'Visible','off');
                    set(slices1, 'Visible','off');
                    
                    %set(slices1(1:dimS(3)), 'Visible','off');
                    set(slices1(dimS(3):dimE(3)), 'Visible','on');
                    %set(slices1(dimS(3):sZ), 'Visible','off');
            end
        end
        maxDimIndPrevious = maxDimInd(1);
        changeVisible = 0;
    end

    function fig_ResizeFcn(h, EventData)
        axes(viewAxis);
        cla(viewAxis);
        set(viewAxis, 'Color', 'black');
        set(viewAxis, 'Visible', 'off');
        figr = gcbo;
        figPos = get(figr, 'Position');
        set(refreshButton, 'Position', [1 (figPos(4)-20) figPos(3) 20]);
        set(viewAxis, 'Position',[ 1 20 figPos(3) figPos(4)-40]);
        set(alphaSlider,'Position', [20 -3 figPos(3)-50 20]);
        set(captureButton, 'Position', [figPos(3)-30 0 30 30]);
        viewBlanked = 1;
    end

    function fig_CloseRequestFcn(h, EventData)
        clear('volumeCA');
        clear('slices1');
        clear('slices2');
        clear('slices3');
        delete(fig);
    end

    %update functions
    function updateVisible(newDimS,newDimE)
        if viewBlanked
           refreshButton_CallBack([],[]);
        end
        dimS = newDimS;
        dimE = newDimE;
        changeVisible = 1;
        rot3d_ActionPostCallBack([], []);
    end

    function drawBounds(onOff)
        %line along axis
        axes(viewAxis);
        
        if ~isempty(bounds)&&ishandle(bounds(1));
            delete(bounds);
        end
        
        if strcmp(onOff, 'on')
            bounds =[...
                        line([dimS(1) dimE(1)],[dimS(2) dimS(2)],[dimS(3) dimS(3)],'Color', [1 0 0]),...
                        line([dimS(1) dimE(1)],[dimS(2) dimS(2)],[dimE(3) dimE(3)],'Color', [1 0 0]),...
                        line([dimS(1) dimE(1)],[dimE(2) dimE(2)],[dimS(3) dimS(3)],'Color', [1 0 0]),...
                        line([dimS(1) dimE(1)],[dimE(2) dimE(2)],[dimE(3) dimE(3)],'Color', [1 0 0]),...
                        
                        line([dimS(1) dimS(1)],[dimS(2) dimE(2)],[dimS(3) dimS(3)],'Color', [0 1 0]),...
                        line([dimS(1) dimS(1)],[dimS(2) dimE(2)],[dimE(3) dimE(3)],'Color', [0 1 0]),...
                        line([dimE(1) dimE(1)],[dimS(2) dimE(2)],[dimS(3) dimS(3)],'Color', [0 1 0]),...
                        line([dimE(1) dimE(1)],[dimS(2) dimE(2)],[dimE(3) dimE(3)],'Color', [0 1 0]),...
                        
                        line([dimS(1) dimS(1)],[dimS(2) dimS(2)],[dimS(3) dimE(3)],'Color', [0 0 1]),...
                        line([dimS(1) dimS(1)],[dimE(2) dimE(2)],[dimS(3) dimE(3)],'Color', [0 0 1]),...
                        line([dimE(1) dimE(1)],[dimS(2) dimS(2)],[dimS(3) dimE(3)],'Color', [0 0 1]),...
                        line([dimE(1) dimE(1)],[dimE(2) dimE(2)],[dimS(3) dimE(3)],'Color', [0 0 1]),...
                    ];
        elseif strcmp(onOff, 'cross')
            bounds =[...
                        line([dimS(1) dimE(1)], [dimE(2)/2 dimE(2)/2], [dimE(2)/2 dimE(2)/2],'Color', [1 0 0]),...
                        
                        line([dimE(1)/2 dimE(1)/2], [dimS(2) dimE(2)], [dimE(2)/2 dimE(2)/2],'Color', [0 1 0]),...
                        
                        line([dimE(1)/2 dimE(1)/2], [dimE(2)/2 dimE(2)/2], [dimS(2) dimE(2)],'Color', [0 0 1]),...
                    ];
            
        end
           
        if strcmp(onOff, 'cross')||strcmp(onOff, 'on')
            set(bounds, 'Parent', viewAxis) %, 'Color', 'red');
        end
    end

    function alphaSlider_CallBack(h, EventData)
    
        newALim = get(alphaSlider, 'Value');
        
        set(viewAxis, 'ALim', [0 1/newALim])
        
    end

    function captureButton_CallBack(h, EventData)
        
        ScreenShotWindow(pos+[60 -60], fig ,[]);
    
    end
    
    function updateVolumeCA(newVolumeCA)
        volumeCA = newVolumeCA;
        [sX sY sZ dum] = size(volumeCA);
        dimS = [1 1 1];
        dimE = [sX sY sZ];
        refreshButton_CallBack([], []);
    end

    function externalUpdate()
        externalUpdateHandle();
    end

end
