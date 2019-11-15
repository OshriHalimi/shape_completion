%Author: James W. Ryland
%June 14, 2012

function [] = TopMenu(pos)
%TOPMENU This function creates a window that lists all of the applications
%That comprise volumetric. And sets a few system variables for preperative
%bug avoidence
    
    AddThisPath();
    
    % Linux Nvidia Duel Moniter FIX 
    Temp = load('GlobalSettings.mat');
    disp(Temp.GlobalSettings)
    if strcmp(Temp.GlobalSettings.linuxMachine, 'yes') == 1
        set(0,'DefaultFigureRenderer','OpenGL');
    end
    % Fix end (important: only activate for linux)
    

    if isempty(pos)
        scr = get(0, 'ScreenSize');
        pos = [(scr(3)/2-140) (scr(4)/2-125)];
    end
    
    fig = figure('Name', 'Volumetric', 'NumberTitle','off', 'Resize', 'off', 'MenuBar', 'none', 'Position', [pos(1) pos(2) 240 380]);
    
    %vis panel
    visPanel = uipanel('Parent', fig, 'Title', 'Visualization Tools', 'Units', 'pixels','Position', [10 260 220 120]);
    
    layerButton = uicontrol('Parent', visPanel, 'Style', 'pushbutton', 'String', 'Layer Editor', 'Position', [10 10 200 40],...
        'CallBack',@layerButton_CallBack);
    
    explorerButton = uicontrol('Parent', visPanel, 'Style', 'pushbutton', 'String', 'Volumetric Explorer', 'Position', [10 50 200 40],...
        'CallBack',@explorerButton_CallBack);
    
    
    %edit panel
    editPanel = uipanel('Parent', fig, 'Title', 'Volume Edit Tools', 'Units', 'pixels', 'Position', [10 90 220 160]);
    
    boolButton = uicontrol('Parent', editPanel, 'Style', 'pushbutton', 'String', 'Boolean Editor', 'Position', [10 10 200 40],...
        'CallBack', @boolButton_CallBack);
    
    scalingButton = uicontrol('Parent', editPanel, 'Style', 'pushbutton', 'String', 'Scaling Editor', 'Position', [10 50 200 40],...
        'CallBack', @scalingButton_CallBack);
    
    rotationButton = uicontrol('Parent', editPanel, 'Style', 'pushbutton', 'String', 'Rotation Editor', 'Position', [10 90 200 40],...
        'CallBack', @rotationButton_CallBack);
    
    %help Panel
    
    helpPanel = uipanel('Parent', fig, 'Title', 'More Information', 'Units', 'pixels', 'Position', [10 10 220 70]);
    
    helpButton = uicontrol('Parent', helpPanel, 'Style', 'pushbutton', 'String', 'Help', 'Position', [10 10 200 40],...
        'CallBack', @helpButton_CallBack);
    
    
    %cross-platform formatting
    uicomponents = [visPanel layerButton explorerButton editPanel boolButton scalingButton rotationButton helpPanel helpButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 15, 'FontName', 'FixedWidth');
    
    
    %CallBacks
    function layerButton_CallBack(h,EventData)
        LayersWindow([],'Volume Layering Window');
    end

    function explorerButton_CallBack(h,EventData)
        manipulationWindow();
    end

    function boolButton_CallBack(h,EventData)
        BooleanWindow([]);
    end

    function scalingButton_CallBack(h,EventData)
        ResizeWindow([]);
    end

    function rotationButton_CallBack(h,EventData)
        RotateWindow([]);
    end
    function helpButton_CallBack(h,EventData)
        help;
    end
end

