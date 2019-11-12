%Author: James W. Ryland
%June 14, 2012

function [ updateVolumeHandle getMapFnHandle getSettingsHandle] = MapBox( fig, pos, externalUpdateHandle, getBoundsHandle, initSettings )
%MAPBOX this allows a user to construct a color and alpha map by defining a
%series of points in color alpha space anchored to data values.
%   fig is the parent figure or panel that MapBox will occupy. pos is the
%   position that MapBox will occupy inside the parent figure.
%   externalUpdateHandle updates external functions with the new
%   mapFnHandle. 

    volume = zeros(20,20,20);
    
    mapFnHandle = [];
    
    currentColorWindowClose = [];
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    allSet = [80 40 40];
    
    np = 0;
    
    
    
    mapPanel = uipanel('Parent', fig, 'Units', 'pixels', 'title', 'Color Map', 'Position', [pos(1) pos(2) 280 140 ]);
    
    pointSlider = uicontrol('Parent', mapPanel, 'Style', 'Slider', 'Position', [0 120 140 20],... 
        'Max', 1, 'Min', 0, 'Value', 0, 'CallBack', @pointSlider_CallBack, 'SliderStep', [.05 .2]);
    
    makeNewPointButton = uicontrol('Parent', mapPanel, 'Style', 'PushButton', 'Clipping', 'On','Position',  [np allSet],...
        'CallBack', @makeNewPointButton_CallBack, 'String', '+');
    
    applyMapButton = uicontrol('Parent', mapPanel, 'Style','PushButton', 'String', 'Color Map', 'Position', [0 0 140 80]);
    
    %Cross Platform Formating
    uicomponents = [mapPanel makeNewPointButton applyMapButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    

    [updateViewVolume updateViewImage] = ViewBox(mapPanel, [140 0], @editImage);
    
    updateVolumeHandle = @updateVolume;
    getMapFnHandle = @getMapFn;
    getSettingsHandle = @getSettings;
    
    % initializes important variables
    pointButtons = {};
    points = {};
    pointSlider_CallBack([],[]);
    updateVolume(volume);
    
    if ~isempty(initSettings)
        points = initSettings.points;
        volume = initSettings.volume;
        initPointButtons();
    end
    
    %Initialize button functions
    function initPointButtons()
        for i=1:size(points,2)
            initPointButton();
        end
        updateButtonIcons();
    end
    
    function initPointButton()
        np = np+40;
        numB = size(pointButtons,2);
        set(makeNewPointButton, 'Position', [np allSet]) 
        newPointButton = uicontrol('Parent', mapPanel, 'Style', 'PushButton','Clipping', 'On', 'Position',  [np-40 allSet]);
        pointButtons{numB+1} = newPointButton;
        pixLength = numB*40+40;
        set(pointSlider, 'Max', pixLength);
        
        function point_CallBack(h,EventData)
            maxV = max(max(max(volume)));
            minV = min(min(min(volume)));
            point = [];
            
            if numB>1
                minV = points{numB}(5);
            end
            if (numB+2)<=size(points,2)
                maxV = points{numB+2}(5);
            end
            if size(points,2)>=(numB+1)
                point = points{numB+1};
            end

            bounds = [minV maxV];
            pointUpdate = makePointUpdator(numB+1);
            newColorWindowClose = ColorWindow([],bounds,pointUpdate,point);
            disp(numB+1);
            
            if ~isempty(currentColorWindowClose)
                currentColorWindowClose();
            end
            currentColorWindowClose = newColorWindowClose;
        end

        set(newPointButton,'CallBack', @point_CallBack);
        startV = min(min(min(volume)));
        if numB>0
            startV = points{numB}(5);
        end
        pointSlider_CallBack([],[]);
    end
    
    
    %CallBacks
    function makeNewPointButton_CallBack(h, EventData)
        np = np+40;
        numB = size(pointButtons,2);
        set(makeNewPointButton, 'Position', [np allSet]) 
        newPointButton = uicontrol('Parent', mapPanel, 'Style', 'PushButton','Clipping', 'On', 'Position',  [np-40 allSet]);
        pointButtons{numB+1} = newPointButton;
        pixLength = numB*40+40;
        set(pointSlider, 'Max', pixLength);
        
        function point_CallBack(h,EventData)
            maxV = max(max(max(volume)));
            minV = min(min(min(volume)));
            point = [];
            
            if numB>1
                minV = points{numB}(5);
            end
            if (numB+2)<=size(points,2)
                maxV = points{numB+2}(5);
            end
            if size(points,2)>=(numB+1)
                point = points{numB+1};
            end

            bounds = [minV maxV];
            pointUpdate = makePointUpdator(numB+1);
            newColorWindowClose = ColorWindow([],bounds,pointUpdate,point);
            disp(numB+1);
            
            if ~isempty(currentColorWindowClose)
                currentColorWindowClose();
            end
            currentColorWindowClose = newColorWindowClose;
        end

        set(newPointButton,'CallBack', @point_CallBack);
        startV = min(min(min(volume)));
        if numB>0
            startV = points{numB}(5);
        end
        points{numB+1} = [1 1 1 1 startV];
        pointSlider_CallBack([],[]);
        updateButtonIcons();
    end

    function pointSlider_CallBack(h, EventData)
        offset = round(get(pointSlider, 'Value'));
        numB = size(pointButtons, 2);
        for i = 1:numB
            newCorner = ((i-1)*40-offset);
            set(pointButtons{i}, 'Position', [newCorner allSet]);
            if (newCorner+40)>140||(newCorner<0)
                set(pointButtons{i}, 'Visible', 'Off');
            else
                set(pointButtons{i}, 'Visible', 'On');
            end
        end
        npCorner = (numB*40-offset);
        set(makeNewPointButton, 'Position', [npCorner allSet]);
        
        if (npCorner+40)>140||(npCorner<0)
            set(makeNewPointButton, 'Visible', 'Off');
        else
            set(makeNewPointButton, 'Visible', 'On');
        end
    end
    
    function mapPanel_DeleteFcn(h, EventData)
        clear(volume);
        clear(initSettings);
        
    end

    %Update Functions
    function updateVolume(newVolume)
        volume = double(newVolume);
        updateViewVolume(volume);
        updateViewImage();
    end
    

    %I believe there is an issue with this function // image produced seems
    %satrurated at the upper end of the spectrum compared to 3d output.
    function [newImageCA] = editImage(imageCA)
        newImageCA = zeros(20,20);
        if ~isempty(mapFnHandle)
            
            [minV maxV] = getBoundsHandle();
            
            CA = mapFnHandle( double(imageCA(:,:,1))*(maxV-minV)/255 + minV);
            %disp(maxV)
            %disp(minV)
            
            
            [sX sY dumy] = size(CA);
            %CH = CheckerGrid(10, sX, sY);
            %with transparency
            %CA(:,:,1) = CA(:,:,1).*CA(:,:,4)+CH.*(1-CA(:,:,4));
            %CA(:,:,2) = CA(:,:,2).*CA(:,:,4)+CH.*(1-CA(:,:,4));
            %CA(:,:,3) = CA(:,:,3).*CA(:,:,4)+CH.*(1-CA(:,:,4));
            %without transparency
            
            newImageCA = CA(:,:,1:3);
            
        else
            maxV = max(max(max(imageCA)));
            minV = min(min(min(imageCA)));
            if ((maxV>1)||(minV<0))&&((maxV-minV)>0)
                imageCA = (imageCA-minV)/(maxV-minV);
            elseif (maxV-minV)>0
                imageCA = zeros(size(imageCA));
            end
            newImageCA = imageCA;
        end
        
        newImageCA = newImageCA;
    end


    function updateButtonIcons()
        
        for i=1:size(pointButtons,2)
            
            R = points{i}(1);
            G = points{i}(2);
            B = points{i}(3);
            A = points{i}(4);
            im = ones(40, 40, 3);
            im(:,:,1) = R*im(:,:,1);
            im(:,:,2) = G*im(:,:,2);
            im(:,:,3) = B*im(:,:,3);
            %ch(:,:,1) = CheckerGrid(10, 40, 40)/4+.25;
            %ch(:,:,2) = ch(:,:,1);
            %ch(:,:,3) = ch(:,:,2);
            
            im = im;%*A+ch*(1-A);
            
            set(pointButtons{i}, 'CData', im, 'String', points{i}(5));
            
        end
        updateApplyMapButtonIcon();
        updateViewImage();
    end
    
    function updateApplyMapButtonIcon()
        aPos = get(applyMapButton,'Position')
        sX = aPos(3);
        sY = aPos(4);
        if size(points,2)>1
            [mapFnHandle useable] = makeMapFn();
            if useable==1
                maxV = max(max(max(volume)));
                minV = min(min(min(volume)));
                
                
                ch(:,:,1) = CheckerGrid(10, sY/2, sX)/2+.25;
                ch(:,:,2) = ch(:,:,1);
                ch(:,:,3) = ch(:,:,2);
                [demo dum] = meshgrid(1:sX, 1:(sY/2));
                max(max(demo))
                demo = (demo/sX)*(maxV-minV)+minV;
                imCA = squeeze(mapFnHandle(demo));
                size(imCA)
                imC = zeros(sY, sX, 3);
                size(imC)
                imC(1:(sY/2),:,1) = imCA(:,:,1).*imCA(:,:,4)+ch(:,:,1).*(1-imCA(:,:,4));
                imC(1:(sY/2),:,2) = imCA(:,:,2).*imCA(:,:,4)+ch(:,:,2).*(1-imCA(:,:,4));
                imC(1:(sY/2),:,3) = imCA(:,:,3).*imCA(:,:,4)+ch(:,:,3).*(1-imCA(:,:,4));
                imC((sY/2+1):sY,:,1) = imCA(:,:,1);
                imC((sY/2+1):sY,:,2) = imCA(:,:,2);
                imC((sY/2+1):sY,:,3) = imCA(:,:,3);
                
                size(imC)
                set(applyMapButton, 'CData', imC);
            end
        end
    end

    % Get functions
    function [mapFnHandle2] = getMapFn()
        mapFnHandle2 = mapFnHandle;
    end

    function [set] = getSettings()
        set.points = points;
        set.volume = volume;
        
    end
    
    % Special Functions
    function [updator] = makePointUpdator(ind)
        function pointUpdator(newPoint)
            points{ind} = newPoint;
            updateButtonIcons();
        end
        updator = @pointUpdator;
    end
    
    % Very Special Functions
    function [mapFnHandle useable] = makeMapFn()
        [rFn useable] = makeChanFn(points, 1);
        gFn = makeChanFn(points, 2);
        bFn = makeChanFn(points, 3);
        aFn = makeChanFn(points, 4);
        
        function [volumeCA] = mapFn(scalerVolume)
            volumeCA(:,:,:,1) = rFn(scalerVolume);
            volumeCA(:,:,:,2) = gFn(scalerVolume);
            volumeCA(:,:,:,3) = bFn(scalerVolume);
            volumeCA(:,:,:,4) = aFn(scalerVolume);
            
        end
        mapFnHandle = @mapFn;
    end

    function [chanFnHandle usable] = makeChanFn(pointsLocal, channel)
        
        usable = 1;
        numPoints = size(pointsLocal, 2);
        x = zeros(numPoints,1);
        y = zeros(numPoints,1);
        
        for i=1:numPoints
            x(i) = pointsLocal{i}(5);
            y(i) = pointsLocal{i}(channel);
        end
        [b m] = unique(x);
        x = x(m);
        y = y(m);
        
        function [chanOut] = chanFn(scalerValue)
            if length(x)>1
                chanOut = interp1(x, y, scalerValue,'linear', 0);
            else
                chanOut = 0;
            end
        end
        
        if length(x)<=1
            usable = 0;
        end
        chanFnHandle = @chanFn; 
        
    end

end
