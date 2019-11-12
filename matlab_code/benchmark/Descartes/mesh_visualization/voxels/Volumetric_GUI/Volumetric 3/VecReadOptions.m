function [ reshapeVec, colWise ] = VecReadOptions( pos, vecLength )
% this function is called if an input file is in a vector format

    height = 500;
    width = 400;
    
    smooth = 0;
    
    reshapeVec = [0 0 0];
    colWise = 1;
    rowWiseDefault = 0;
    
    %fmri 64*64*38 preset Alice O'toole's Lab
    if vecLength == (64*64*38)
        reshapeVec(1) = 64;
        reshapeVec(2) = 64;
        reshapeVec(3) = 38;
        rowWiseDefault = 1;
    
    %mri structural preset Alice O'toole's Lab
    elseif vecLength == (256*256*160) 
        reshapeVec(1) = 256;
        reshapeVec(2) = 256;
        reshapeVec(3) = 160;
        rowWiseDefault = 1;
    end
    
    
    scr = get(0,'ScreenSize');
    
    if isempty(pos)
        pos = [1 (scr(4)-height)];
    end
    
    title = 'Vector Reshape Options';
    
    fig = figure('Name',title, 'Resize', 'off', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height]);
    
    
    instruction = {[
        'The file you are trying to load is in vector format, ',...
        'please give the correct dimensions and read order to ',...
        'reconstruct the original volume' ]};
    
    
    sX = .9;
    sY = .3;
    spX = .05;
    spY = .7;
    instructText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(instructText, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(instructText, 'String', textwrap(instructText, instruction));
    
    
    sX = .3;
    sY = .05;
    spX = .05;
    spY = .6;
    instructText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(instructText, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(instructText, 'String', 'Dimensions');
    
    
    sX = .05;
    sY = .05;
    spX = .05;
    spY = .52;
    xDimText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(xDimText, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(xDimText, 'String', 'X');
    
    sX = .15;
    sY = .05;
    spX = .05+.05;
    spY = .52;
    xDimEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(xDimEdit, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(xDimEdit, 'String', num2str(reshapeVec(1)));
    
    
    
    sX = .05;
    sY = .05;
    spX = .05+.3;
    spY = .52;
    yDimText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(yDimText, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(yDimText, 'String', 'Y');
    
    sX = .15;
    sY = .05;
    spX = .05+.3+.05;
    spY = .52;
    yDimEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(yDimEdit, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(yDimEdit, 'String', num2str(reshapeVec(2)));
    
    
    
    sX = .05;
    sY = .05;
    spX = .05+.6;
    spY = .52;
    zDimText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(zDimText, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(zDimText, 'String', 'Z');
    
    sX = .15;
    sY = .05;
    spX = .05+.6+.05;
    spY = .52;
    zDimEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(zDimEdit, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(zDimEdit, 'String', num2str(reshapeVec(3)));
    
    sX = .5;
    sY = .1;
    spX = .05;
    spY = .30;
    radioGroup = uibuttongroup('Parent', fig, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    
    sX = 1;
    sY = .5;
    spX = 0;
    spY = .5;
    readColRadio = uicontrol('Parent', radioGroup, 'Style', 'Radiobutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(readColRadio, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(readColRadio, 'String', 'Fill Column Wise');
    
    sX = 1;
    sY = .5;
    spX = 0;
    spY = 0;
    readRowRadio = uicontrol('Parent', radioGroup, 'Style', 'Radiobutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(readRowRadio, 'FontSize', 20, 'HorizontalAlignment', 'Left');
    set(readRowRadio, 'String', 'Fill Row Wise');
    
    sX = 1;
    sY = .1;
    spX = 0;
    spY = .15;
    voxNumText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(voxNumText, 'FontSize', 20, 'HorizontalAlignment', 'Left', 'ForegroundColor', 'red');
    set(voxNumText, 'String', '');
    
    sX = .4;
    sY = .1;
    spX = .3;
    spY = .05;
    applyButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @applyButton_Callback);
    set(applyButton, 'String', 'Apply', 'FontSize', 20);
    
    
    %INITIALIZATION
    set(readRowRadio, 'Value', rowWiseDefault);
    
    
    % CALLBACKS
    
    function applyButton_Callback(hObject, eventData, handles)
        
        reshapeVec(1) = round(str2double(get(xDimEdit, 'String')));
        reshapeVec(2) = round(str2double(get(yDimEdit, 'String')));
        reshapeVec(3) = round(str2double(get(zDimEdit, 'String')));
        
        colWise = get(readColRadio, 'Value');
        
        if prod(reshapeVec)==vecLength
            uiresume(gcbf);
        else
            set(voxNumText, 'String', ['Voxel Number does not match Vector Length: ' num2str(vecLength) ' ~= ' num2str(prod(reshapeVec))]);
        end
    end
    
    
    % first check to see if the voxel num matches!
    uiwait(gcf);
    
    delete(fig);
    
    
end

