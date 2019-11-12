function [ range, smooth, toMask, toShell, shift ] = ImportOptions( pos, range, counts, values )
% this figure lets a user alter the way a volume file's values are loaded
% into the layer.

    height = 500;
    width = 400;
    
    smooth = 0;
    
    scr = get(0,'ScreenSize');
    
    if isempty(pos)
        pos = [1 (scr(4)-height)];
    end
    
    title = 'Import Options';
    
    fig = figure('Name',title, 'Resize', 'off', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height]);
    
    sX = .9;
    sY = .5;
    spX = .05;
    spY = .5;
    histAxis = axes('Parent', fig, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'ButtonDownFcn', @colorMapDisp_ButtonDownFcn);
    bar(histAxis, values, counts);
    
    
    sX = .1;
    sY = .05;
    spX = .05;
    spY = .4;
    minText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(minText, 'String', 'Min', 'FontSize', 20);
    
    sX = .3;
    sY = .05;
    spX = .15;
    spY = .4;
    minEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(minEdit, 'String', num2str(range(1)), 'FontSize', 20);
    
    
    sX = .1;
    sY = .05;
    spX = .5;
    spY = .4;
    maxText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(maxText, 'String', 'Max', 'FontSize', 20);
    
    
    
    sX = .3;
    sY = .05;
    spX = .6;
    spY = .4;
    maxEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(maxEdit, 'String', num2str(range(2)), 'FontSize', 20);
    
    
    sX = .1;
    sY = .05;
    spX = .5;
    spY = .35;
    shiftText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(shiftText, 'String', 'Shift', 'FontSize', 20);
    
    
    
    sX = .3;
    sY = .05;
    spX = .6;
    spY = .35;
    shiftEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(shiftEdit, 'String', num2str(0), 'FontSize', 20);
    
    
    
    
    
    sX = .3;
    sY = .05;
    spX = .05;
    spY = .35;
    toMaskCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(toMaskCheckBox, 'Value', 0);
    set(toMaskCheckBox, 'String', 'To Mask', 'FontSize', 20);
    
    sX = .3;
    sY = .05;
    spX = .05;
    spY = .30;
    toShellCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(toShellCheckBox, 'Value', 0);
    set(toShellCheckBox, 'String', 'To Shell', 'FontSize', 20);
    
    
    
    sX = .3;
    sY = .05;
    spX = .05;
    spY = .25;
    smoothCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(smoothCheckBox, 'Value', 0);
    set(smoothCheckBox, 'String', 'Smoothing', 'FontSize', 20);
    
    
    
    sX = .7;
    sY = .05;
    spX = .05;
    spY = .20;
    maxZeroCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(maxZeroCheckBox, 'Value', 0);
    set(maxZeroCheckBox, 'String', 'Values Above Max to zero', 'FontSize', 20);
    
    
    % add an invert values function!! max(bla)-bla = blaInv
    
    sX = .4;
    sY = .1;
    spX = .3;
    spY = .05;
    applyButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', 'uiresume(gcbf)');
    set(applyButton, 'String', 'Apply', 'FontSize', 20);
    
    
    
    
    
    %UIWAIT
    
    uiwait(gcf);
    posNow = get(fig, 'position');
    start = str2double(get(minEdit, 'String'));
    stop = str2double(get(maxEdit, 'String'));
    range = [start stop];
    smooth = get(smoothCheckBox, 'Value');
    toMask = get(toMaskCheckBox, 'Value');
    toShell = get(toShellCheckBox, 'Value');
    maxToZero = get(maxZeroCheckBox, 'Value');
    shift = str2double(get(shiftEdit, 'String'));
    delete(fig);
    
    
%     % Second option tab for resolution and rotation
%     fig = figure('Name',title, 'Resize', 'off', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', posNow);
%     
%     sX = .8;
%     sY = .3;
%     spX = .1;
%     spY = .3;
%     buttonGroup = uibuttongroup('Parent', fig, 'Title', 'Resolution', 'Position', [spX spY sX sY]);
%     
%     sX = .8;
%     sY = .3;
%     spX = .1;
%     spY = .6;
%     nativeButton = uicontrol('Parent', buttonGroup, 'Style', 'RadioButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
%     set(nativeButton, 'String', 'Native', 'FontSize', 20);
%     
%     sX = .8;
%     sY = .3;
%     spX = .1;
%     spY = .3;
%     mediumButton = uicontrol('Parent', buttonGroup, 'Style', 'RadioButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
%     set(mediumButton, 'String', 'Medium 20M', 'FontSize', 20);
%     
%     sX = .8;
%     sY = .3;
%     spX = .1;
%     spY = .0;
%     pubButton = uicontrol('Parent', buttonGroup, 'Style', 'RadioButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
%     set(pubButton, 'String', 'Publication 27M', 'FontSize', 20);
%     
%     
%     sX = .8;
%     sY = .3;
%     spX = .1;
%     spY = .65;
%     rotationPanel = uipanel('Parent', fig, 'Title', 'Rotate', 'Position', [spX spY sX sY]);
%     
%      sX = .8;
%     sY = .3;
%     spX = .1;
%     spY = .6;
%     rotationLabel = uicontrol('Parent', rotationPanel, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
%     set(rotationLabel, 'String', 'Enter a series of 90 degree rotations (e.g -x,-y,-y,-z)', 'FontSize', 20);
%     
%     sX = .8;
%     sY = .2;
%     spX = .1;
%     spY = .25;
%     rotationEdit = uicontrol('Parent', rotationPanel, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
%     set(rotationEdit, 'String', '', 'FontSize', 20);
%     
%     
%     sX = .4;
%     sY = .1;
%     spX = .3;
%     spY = .05;
%     applyButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', 'uiresume(gcbf)');
%     set(applyButton, 'String', 'Apply', 'FontSize', 20);
%     
%     
%     %UIWAIT
%     
%     uiwait(gcf);
%     delete(fig);
    
    
    
end

