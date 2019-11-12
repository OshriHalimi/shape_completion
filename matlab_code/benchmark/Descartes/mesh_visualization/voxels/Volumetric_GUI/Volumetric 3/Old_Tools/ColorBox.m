%Author: James W. Ryland
%June 14, 2012

function [ inputUpdateHandle getCAHandle ] = ColorBox( fig, title, pos, externalUpdateHandle, initCA)
%COLORBOX allows a user to specify a point in color alpha space for styling
%a representation of a volume.
%   Fig is the parent figure that ColorBox will reside in. Pos is the position
%   that ColorBox will occupy in the figure. ExternalUpdateHandle updates
%   external functions with the chosen values. initCA is the initialization
%   parameters for the colorbox.
    
    scalerVolume = uint8(rand(50,50,50)>.5);
    colorVolume = [];
    
    if ~isempty(initCA)
        iR = initCA(1);
        iB = initCA(3);
        iG = initCA(2);
        iA = initCA(4);
    else
        iR = 1;
        iB = 1;
        iG = 1;
        iA = 1;
    end
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    colorPanel = uipanel('Parent', fig, 'Title', title, 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 150 ],...
        'DeleteFcn', @colorPanel_DeleteFcn);
    
    rLabel = uicontrol('Parent', colorPanel, 'Style', 'Text', 'String', 'R', 'Position', [10 110 20 20]);
    
    rSlider = uicontrol('Parent', colorPanel, 'Style', 'Slider', 'String', 'R', 'Position', [30 110 100 20],...
        'Max', 1, 'Min', 0, 'Value', iR, 'CallBack', @slider_CallBack);
    
    bLabel = uicontrol('Parent', colorPanel, 'Style', 'Text', 'String', 'B', 'Position', [10 80 20 20]);
    
    bSlider = uicontrol('Parent', colorPanel, 'Style', 'Slider', 'String', 'B', 'Position', [30 80 100 20],...
        'Max', 1, 'Min', 0, 'Value', iB, 'CallBack', @slider_CallBack);
    
    gLabel = uicontrol('Parent', colorPanel, 'Style', 'Text', 'String', 'G', 'Position', [10 50 20 20]);
    
    gSlider = uicontrol('Parent', colorPanel, 'Style', 'Slider', 'String', 'G', 'Position', [30 50 100 20],...
        'Max', 1, 'Min', 0, 'Value', iG, 'CallBack', @slider_CallBack);
    
    aLabel = uicontrol('Parent', colorPanel, 'Style', 'Text', 'String', 'A', 'Position', [10 20 20 20]);
    
    aSlider = uicontrol('Parent', colorPanel, 'Style', 'Slider', 'String', 'A', 'Position', [30 20 100 20],...
        'Max', 1, 'Min', 0, 'Value', iA, 'CallBack', @slider_CallBack);
    
    %Cross Platform Formating
    uicomponents = [colorPanel rLabel bLabel gLabel aLabel];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    [updateVolumeHandle updateImageHandle]= ViewBox(colorPanel, [140 0], @modifyImage);

    inputUpdateHandle = @inputUpdate;
    getCAHandle = @getCA;
    
    %setDefaults
    slider_CallBack([], []);
    
    
    % CallBacks
    function slider_CallBack(h, EventData)        
        updateImageHandle();
        externalUpdate();
    end
    
    % Update Functions
    function updateViewBox()
        updateVolumeHandle(scalerVolume);
    end

    function inputUpdate(newBooleanVolume)
        scalerVolume = uint8(newBooleanVolume*255);
        updateViewBox();
        externalUpdate();
    end

    function externalUpdate()
        if ~isempty(externalUpdateHandle)
            if ~isempty(colorVolume);
                externalUpdateHandle(colorVolume);
            end
        end
    end

    function colorPanel_DeleteFcn(h,EventData)
        clear('scalerVolume');
        clear('colorVolume');
        
    end

    % view modifacation functions
    function [newImage] = modifyImage(oldImage) % optimization changes
        [sx sy sc] = size(oldImage);
        R = get(rSlider, 'Value');
        G = get(gSlider, 'Value');
        B = get(bSlider, 'Value');
        A = get(aSlider, 'Value');
        checker(:,:,1) = double(((CheckerGrid(10, sx, sy)/4+(.5-1/4))*255));
        checker(:,:,2) = checker(:,:,1);
        checker(:,:,3) = checker(:,:,2);
        overImage(:,:, 1) = double(oldImage(:,:,1))*R;
        overImage(:,:, 2) = double(oldImage(:,:,2))*G;
        overImage(:,:, 3) = double(oldImage(:,:,3))*B;
        alphaImage = double(oldImage)*A;     
        
        newImage = uint8(overImage.*alphaImage/255 + checker.*(255-alphaImage)/255);
    end

    function [CA] = getCA()
        R = get(rSlider, 'Value');
        G = get(gSlider, 'Value');
        B = get(bSlider, 'Value');
        A = get(aSlider, 'Value');
        CA = [R G B A];
    end
end
