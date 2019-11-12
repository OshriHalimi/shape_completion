function [ ] = ScreenShotWindow( pos, fig2Shoot, externalUpdateHandle)
%   ScreenShotWindow allows the user to save a function 


    if isempty(pos)
        pos = [0 0];
    end
    
    fig = figure('Name', 'Screen Shot Options', 'NumberTitle','off', 'Resize', 'off', 'MenuBar', 'none', 'Position', [pos(1) pos(2) 300 200]);
    
    fileLabel = uicontrol('Parent', fig, 'Style', 'text', 'String', 'File Name', 'Position', [10 170 50 15]);
    
    fileNameEdit = uicontrol('Parent', fig, 'Style', 'edit', 'String', 'File Name', 'Position', [60 170 160 20]);
    
    fileLabel = uicontrol('Parent', fig, 'Style', 'text', 'String', 'File Type', 'Position', [10 140 50 15]);
    
    imageFormatList = {'jpeg', 'tiff'};
    
    fileTypeMenu = uicontrol('Parent', fig, 'Style', 'popupmenu', 'String', imageFormatList, 'Position', [60 140 160 20]);
    
    acceptButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Ok', 'Position', [170 10 60 20],...
        'Callback', @acceptButton_CallBack);
    
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Cancel', 'Position', [230 10 60 20],...
        'Callback', @cancelButton_CallBack);
    
    
    
    
    
    
    %CallBacks
    
    function acceptButton_CallBack(h, EventData)
        
        formatStrings = {'jpg', 'tiff'};
        %set(0,'CurrentFigure',fig2Shoot);
        saveas(fig2Shoot, get(fileNameEdit, 'String'), formatStrings{get(fileTypeMenu, 'Value')});
        fig_CloseRequestFcn([],[]);
    end

    function cancelButton_CallBack(h, EventData)
        fig_CloseRequestFcn([],[]);
        
    end

    function fig_CloseRequestFcn(h, EventData)
        delete(fig);
    end
    
    
end

