%Author: James W. Ryland
%June 14, 2012

function [dimStart dimEnd methodGroup] = manipulationWindow()
%MANIPULATIONWINDOW allows a user to load a CAV file, view it, and
%manipulate it by interpolation or cropping or both.
%   This function makes use of ExploreWindow. It can cut into the rendering
%   in real time by manipulating the visibility of the Slices most along
%   the view axis.
    
    
    screenS = get(0, 'ScreenSize');
    interpMethod = 'None';
    selectionView = 'on';
    originalCAV = [];
    currentCAV = [];
    
    
    fig = figure;
    set(fig, 'Name', 'Manipulation Window', 'position', [1 screenS(4)-270 400 400],...
            'MenuBar', 'none', 'NumberTitle', 'off', 'resize', 'off', 'SelectionHighlight', 'off',...
            'CloseRequestFcn', @fig_CloseRequestFcn);
    
    figureAdjust(fig);
        
    zoomPanel = uipanel('parent', fig, 'Title', 'Zoom Options',...
                        'units', 'pixels', 'position', [10, 140, 140, 120]);
    
    selectZoomInButton = uicontrol(zoomPanel, 'style', 'pushbutton',...
                                'String', 'Crop & Interp',...
                                'Position', [10 60 120, 40], 'CallBack', @selectZoomInButton_CallBack);
    
    selectZoomOutButton = uicontrol(zoomPanel, 'style', 'pushbutton',...
                                'String', 'Reload Original',...
                                'Position', [10 10 120, 40], 'CallBack', @selectZoomOutButton_CallBack, 'Enable', 'off');
    
    loadPanel = uipanel('parent', fig, 'Title', 'File Options',...
                        'units', 'pixels', 'position', [10, 270, 140, 120]);
    
    loadFilesButton = uicontrol(loadPanel, 'style', 'pushbutton',...
                                'String', 'Load Files',...
                                'Position', [10 60 120, 40], 'CallBack', @loadFilesButton_CallBack);
    
    saveCAVButton = uicontrol(loadPanel, 'style', 'pushbutton',...
                                'String', 'Save Vizualization',...
                                'Position', [10 10 120, 40], 'CallBack', @saveCAVButton_CallBack);
                            
    methodGroup = uibuttongroup('parent', fig, 'Title', 'Selection View',...
                        'units', 'pixels', 'position', [160, 140, 230, 120], 'SelectionChangeFcn',@methodGroup_SelectionChangeFcn);
                    
    barsRadio = uicontrol('parent', methodGroup, 'style', 'radio',...
                        'String', 'Bars','position', [10 70 120 30], 'Tag', 'on');
                    
    noneRadio = uicontrol('parent', methodGroup, 'style', 'radio',...
                        'String', 'None','position', [10 40 120 30], 'Tag', 'off');
                    
    controlPanel = uipanel('parent', fig, 'Title', 'Selection Control',...
                        'units', 'pixels', 'position', [160, 270, 230, 120]);
                    
    dimStartSlider = zeros(3,1);
    dimEndSlider = zeros(3,1);
    dimLabel = ['X', 'Y', 'Z'];
    
    %Create Selection Controls
    for i = 1:3
        bgColor = zeros(1,3);
        bgColor(i) = 1;
        
        dimLabel(i) = uicontrol(controlPanel, 'Style', 'text', 'String', dimLabel(i),...
            'ForegroundColor', bgColor, 'Position', [10 (105-i*30) 20 20]); 
        
        dimStartSlider(i) = uicontrol(controlPanel,'Style', 'slider', 'min', 1, 'max',100,...
                        'Value', 1, 'SliderStep', [.01 .01],...
                        'Position', [ 30 (105-i*30) 90 20], 'CallBack', @selectSliders_CallBack);
        
        dimEndSlider(i) = uicontrol(controlPanel,'Style', 'slider', 'min', 1, 'max',100,...
                        'Value', 100, 'SliderStep', [.01 .01],...
                        'Position', [ 130 (105-i*30) 90 20], 'CallBack', @selectSliders_CallBack);
        
    end
    
    interpPanel = uipanel('parent', fig, 'Title', 'Interpolation Options',...
                        'units', 'pixels', 'position', [10, 10, 380, 120]);

    interpGroup = uibuttongroup('parent', interpPanel, 'Title', 'Setting',...
                        'units', 'pixels', 'position', [10, 10, 120, 90], 'SelectionChangeFcn', @interpGroup_SelectionChangeFcn);
    
    noneRadio = uicontrol('parent', interpGroup, 'style', 'radio',...
                        'String', 'None','position', [10 50 90 20]);   
                    
    autoRadio = uicontrol('parent', interpGroup, 'style', 'radio',...
                        'String', 'Automatic','position', [10 30 90 20]);
                    
    targetLabel = uicontrol('parent', interpPanel, 'Style', 'Text', 'String', 'Targect Voxel Number',... 
                        'Position', [140 80 140 15]);
    
    targetEdit = uicontrol('parent', interpPanel, 'Style', 'Edit', 'String', '16,000,000',... 
                        'Position', [220 60 100 20]);
    
    alphaCorrectLabel = uicontrol('parent', interpPanel, 'Style', 'Text', 'String', 'Alpha Correction Multiplier',... 
                        'Position', [140 30 130 15]);
    
    alphaCorrectEdit = uicontrol('parent', interpPanel, 'Style', 'Edit', 'String', '1',... 
                        'Position', [275 30 40 20]);
    
                    
    %Setup Explorer window
    [updateCAVolumeHandle updateVisibleHandle drawBoundsHandle] = ExploreRenderBox([400 (screenS(4)-270)], 'Exploration View', []);
    
    
    %Cross Platform Formatting
    %do here...
    
    
    %CallBacks
    function loadFilesButton_CallBack(h, EventData)
        CavFileWindow('CAV Loader', [], @setVolumeParameters);
        
    end

    function saveCAVButton_CallBack(h, EventData)
        SaveWindow('CAV saver', [], currentCAV);
        
    end


    function setVolumeParameters(newCAV)
        originalCAV = newCAV;
        currentCAV = newCAV;
        [sX sY sZ dum] = size(originalCAV);
        dimE = [sX sY sZ];
        for i=1:3
            set(dimStartSlider(i), 'Max', dimE(i), 'Min', 1, 'Value', 1);
            set(dimEndSlider(i), 'Max', dimE(i), 'Min', 1, 'Value', dimE(i));
        end
        updateCAVolumeHandle(originalCAV);
    end
    
    function selectSliders_CallBack(h, EventData)
        dimS = zeros(1,3);
        dimE = zeros(1,3);
        boundsGood = 0;
        for i=1:3
            newS = round(get(dimStartSlider(i), 'Value'));
            newE = round(get(dimEndSlider(i), 'Value'));
            if newS<newE
                dimS(i) = newS;
                dimE(i) = newE;
                boundsGood = boundsGood+1;
            end
            
        end
        if boundsGood==3
            updateVisibleHandle(dimS, dimE);
            drawBoundsHandle(selectionView);
        end
        
    end

    function selectSliders_Enable(onOff)
        for i=1:3
            set(dimStartSlider(i), 'Enable', onOff);
            set(dimEndSlider(i), 'Enable', onOff);
        end
    end
    
    function interpGroup_SelectionChangeFcn(h, EventData)
        selectedMethodButton = EventData.NewValue;
        interpMethod = get(selectedMethodButton, 'String');
        disp(interpMethod);
        
    end

    function methodGroup_SelectionChangeFcn(h, EventData)
        selectedMethodButton = EventData.NewValue;
        selectionView = get(selectedMethodButton, 'tag');
        disp(selectionView);
        
    end

    function selectZoomInButton_CallBack(h, EventData)
        if ~isempty(originalCAV)
            
            set(selectZoomInButton, 'Enable', 'off');
            pause(0.05);
            
            dimS = zeros(1,3);
            dimE = zeros(1,3);
            for i=1:3
                dimS(i) = round(get(dimStartSlider(i), 'Value'));
                dimE(i) = round(get(dimEndSlider(i), 'Value'));
            end
            
            
            tempCAV = originalCAV(dimS(1):dimE(1),dimS(2):dimE(2),dimS(3):dimE(3),1:4);
            
            currentCAV = tempCAV;
            
            switch interpMethod
                case 'None'
                    currentCAV = tempCAV;
                case 'Automatic'
                    disp('Interpolating Please Wait');
                    targ = str2double(get(targetEdit, 'String'));
                    alphaM = str2double(get(alphaCorrectEdit, 'String'));
                    
                    currentCAV = uint8([]);
                    currentCAV(:,:,:,1) = uint8(targetedInterp3(tempCAV(:,:,:,1), targ));
                    currentCAV(:,:,:,2) = uint8(targetedInterp3(tempCAV(:,:,:,2), targ));
                    currentCAV(:,:,:,3) = uint8(targetedInterp3(tempCAV(:,:,:,3), targ));
                    currentCAV(:,:,:,4) = uint8(targetedInterp3(tempCAV(:,:,:,4), targ)*alphaM);
                    
                    disp('currentCAV is uint8');
                    disp(isa(currentCAV,'uint8'));
                    
                case 'Specify'
                    currentCAV = tempCAV;
            end
            clear('tempCAV');
            
            updateCAVolumeHandle(currentCAV);
            
            selectSliders_Enable('off');
            set(selectZoomOutButton, 'Enable', 'on');
            pause(0.05);
        end
    end

    function selectZoomOutButton_CallBack(h, EventData)
        if ~isempty(originalCAV)
            
            set(selectZoomOutButton, 'Enable', 'off');
            pause(0.05);
            
            updateCAVolumeHandle(originalCAV);
            selectSliders_CallBack([],[]);
            
            selectSliders_Enable('on');
            set(selectZoomInButton, 'Enable', 'on');
            pause(0.05);
        end
    end

    function fig_CloseRequestFcn(h, EventData)
        clear('originalCAV');
        clear('currentCAV');
        delete(fig);
    end

end
