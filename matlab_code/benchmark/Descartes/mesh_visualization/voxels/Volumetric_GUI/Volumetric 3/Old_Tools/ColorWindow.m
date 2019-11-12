%Author: James W. Ryland
%June 14, 2012

function [closeWindowHandle ] = ColorWindow( pos, bounds, externalUpdateHandle, RGBAV )
%COLORWINDOW color window allos the user to specify a point in value, color
%, and alpha space in order to construct a color map for volumetric data.
%   Pos is the position ColorWindow will occupy on the desktop. Bounds are
%   the boundries in Value space that this point can occupy.
%   ExternalUpdateHandle updates external functions with the point chosen
%   by ColorWindow. RGBAV the initial value that the window will spawn
%   with.

    if isempty(pos)
        pos = [0 0];
    end
    if isempty(bounds)
        bounds = [0 1]; 
    end
    
    figDeleted = 0;
    
    fig = figure('Name', 'Color & Alpha', 'NumberTitle', 'off','Position', [pos(1) pos(2) 140 200],...
        'CloseRequestFcn', @fig_CloseRequestFcn);
   
    figureAdjust(fig);
    
    rLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'R', 'Position', [10 110 20 20]);
    
    rSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'String', 'R', 'Position', [30 110 100 20],...
        'Max', 1, 'Min', 0, 'Value', 1, 'CallBack', @slider_CallBack);
    
    bLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'B', 'Position', [10 80 20 20]);
    
    bSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'String', 'B', 'Position', [30 80 100 20],...
        'Max', 1, 'Min', 0, 'Value', 1, 'CallBack', @slider_CallBack);
    
    gLabel = uicontrol('Parent',fig, 'Style', 'Text', 'String', 'G', 'Position', [10 50 20 20]);
    
    gSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'String', 'G', 'Position', [30 50 100 20],...
        'Max', 1, 'Min', 0, 'Value', 1, 'CallBack', @slider_CallBack);
    
    aLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'A', 'Position', [10 20 20 20]);
    
    aSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'String', 'A', 'Position', [30 20 100 20],...
        'Max', 1, 'Min', 0, 'Value', 1, 'CallBack', @slider_CallBack);
    
    vLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'V', 'Position', [10 140 20 20]);
    
    vSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'String', 'A', 'Position', [30 140 70 20],...
        'Max', bounds(2), 'Min', bounds(1), 'Value', bounds(1), 'CallBack', @vSlider_CallBack);
    
    valueLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', bounds(1), 'Position', [100 140 40 20]);
    

    useButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Use', 'Position', [10 170 120 30],...
        'CallBack', @useButton_CallBack );
    
    %Cross Platform Formating
    uicomponents = [ rLabel bLabel gLabel aLabel vLabel valueLabel useButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    closeWindowHandle = @closeWindow;
    
    %initialyze setup
    if ~isempty(RGBAV)
        set(rSlider, 'Value', RGBAV(1));
        set(gSlider, 'Value', RGBAV(2));
        set(bSlider, 'Value', RGBAV(3));
        set(aSlider, 'Value', RGBAV(4));
        set(vSlider, 'Value', RGBAV(5));
        set(valueLabel, 'String', RGBAV(5));
        
    else
        RGBAV = [1 1 1 1 bounds(1)];
    end
    
    slider_CallBack([],[]);
    
    %CallBacks
    function slider_CallBack(h, EventData)
        R = get(rSlider, 'Value');
        G = get(gSlider, 'Value');
        B = get(bSlider, 'Value');
        A = get(aSlider, 'Value');
        im = ones(30, 120, 3);
        im(:,:,1) = R*im(:,:,1);
        im(:,:,2) = G*im(:,:,2);
        im(:,:,3) = B*im(:,:,3);
        ch(:,:,1) = CheckerGrid(10, 30, 120)/4+.25;
        ch(:,:,2) = ch(:,:,1);
        ch(:,:,3) = ch(:,:,2);
        
        im = im*A+ch*(1-A);
        
        set(useButton, 'CData', im);
        RGBAV(1:4) = [R G B A];
    end

    function vSlider_CallBack(h, EventData)
        v = get(vSlider, 'Value');
        set(valueLabel, 'String', double(v));
        RGBAV(5) = v;
    end
    
    function useButton_CallBack(h, EventData)
        externalUpdate();
    end

    function fig_CloseRequestFcn(h, EventData)
        closeWindow();
    end
    
    %Update Functions
    function externalUpdate()
        if ~isempty(externalUpdateHandle)
            externalUpdateHandle(RGBAV);
        end
    end

    function closeWindow()
        if ~isempty(fig)
            delete(fig);
            fig = [];
        end
    end
    
end
