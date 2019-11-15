function [UpdateVolsHandle, UpdateMapColorHandle, UpdateMapAlphaHandle, GetCamPropertiesHandle, fig, viewAxis ] = ViewWindow( RecalculateHandle, layFig)
% this lets the user view the finished product

    %INITIALIZE
    height = 600;
    width = 600;
    
    scr = get(0,'ScreenSize');
    
    pos = [ 400+100 (scr(4)-height)];

    title = '3D View';
    
    fig = figure('Name',title, 'Resize', 'on', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height], 'CloseRequestFcn', @MyCloseRequestFcn, 'DeleteFcn', @MyDeleteFcn); %'MenuBar', 'None'
    set(fig, 'Color', 'black');
    set(fig, 'Resize', 'off');

    sX = 1;
    sY = 1;
    spX = 0;
    spY = 0;
    
    viewAxis = axes('Parent', fig, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(viewAxis, 'XTick', [], 'YTick', [], 'ZTick', []);
    set(viewAxis, 'Projection', 'Perspective');
    set(viewAxis, 'color', 'black');
    
    rot3d = rotate3d(viewAxis);
    set(rot3d, 'Enable', 'on');
    set(rot3d, 'RotateStyle', 'Orbit');
    setAllowAxesRotate(rot3d,viewAxis,true);
    set(rot3d, 'ActionPostCallBack', @rot3d_ActionPostCallBack);
    
    sX = .05;
    sY = .05;
    spX = .95;
    spY = .0;
    resizeButton = uicontrol('Parent', fig, 'Style', 'ToggleButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @resizeButton_Callback);
    set(resizeButton, 'String', '+-', 'FontSize', 20);

    
    
    %INITIALIZE
    UpdateVolsHandle = @UpdateVols;
    UpdateMapColorHandle = @UpdateMapColor;
    UpdateMapAlphaHandle = @UpdateMapAlpha;
    GetCamPropertiesHandle = @GetCamProperties;
    
    
    
    maxDimIndPrevious = 1;
    
    slices1 = [];
    slices2 = [];
    slices3 = [];
    set(fig,'ResizeFcn', @MyResizeFcn)
    
    
    %CALLBACKS
    
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

    function MyCloseRequestFcn(hObject, eventData)
        
        close(layFig);
        
    end

    function MyDeleteFcn(hObject, eventData)
        setAllowAxesRotate(rot3d,viewAxis,false);
        %delete(slices1);
        %delete(slices2);
        %delete(slices3);
        cla (viewAxis);
    end

    function resizeButton_Callback(hObject, eventData, handles)
        val = get(resizeButton, 'Value');
        
        if val==1

            set(fig, 'Resize', 'on');
        else

            set(fig, 'resize', 'off');
            
            RecalculateHandle()
        end
        
    end

    function MyResizeFcn(hObject, eventData, handles)
        
        clear('slices1','slices2','slices3');
        cla(viewAxis);
        
    end
    

    %EXTERNAL USE
    function UpdateVols(indexedColor, indexedAlpha)
        
        set(0,'CurrentFigure', fig);
        set(fig,'CurrentAxes', viewAxis)
        
        
        if ~isempty(indexedColor) && ~isempty(indexedAlpha)
            CA(:,:,:,1) = indexedColor;
            CA(:,:,:,2) = indexedAlpha;
            cla(viewAxis);
            
            clear('slices1','slices2','slices3');% mem management
            [slices1, slices2, slices3] = volumeRenderMono(CA, [],[],[], viewAxis);
            
            clear CA;
            
            set(slices1, 'Visible', 'On');
            
        else
            
            clear('slices1','slices2','slices3');% mem management
            slices1 = [];
            slices2 = [];
            slices3 = [];
            cla(viewAxis);
            
            set(viewAxis, 'color', 'black');
        end
    end
    
    function UpdateMapColor(mapColor)
        set(fig,'colormap', mapColor);
        clear mapColor;
    end

    function UpdateMapAlpha(mapAlpha)
        set(fig,'alphamap', mapAlpha);
        clear mapAlpha;
    end
    
    function [camPos, volCenter, camAngle, camUp, camFor] = GetCamProperties()
        
        camPos = campos(viewAxis);
        volCenter = camtarget(viewAxis);
        camAngle = camva(viewAxis);
        camUp = camup(viewAxis)/norm(camup(viewAxis));
        camFor = volCenter-camPos;
        camFor = camFor/norm(camFor);
        
        camRight = cross(camUp, camFor);
        camRight = camRight/norm(camRight);
        camUp = cross(camFor, camRight);
        camUp = camUp/norm(camUp);
        
    end
    
end

