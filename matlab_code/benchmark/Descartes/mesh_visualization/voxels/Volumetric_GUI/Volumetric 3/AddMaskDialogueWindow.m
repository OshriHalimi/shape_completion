function [ ] = AddMaskDialogueWindow( viewAxis, offset, GetActualSize, AddGeneratedLayer)
%ADDMASKDIALOGUEWINDOW Summary of this function goes here
%   Detailed explanation goes here
    scr = get(0,'ScreenSize');

    width = 300; 
    height = 200;
    
    pos = [ 400+100+600 (scr(4)-height)];


    title = 'Mask Type Selection';
    fig = figure('Name',title, 'Resize', 'on', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height]);  % 'CloseRequestFcn', @MyCloseRequestFcn, 'DeleteFcn', @MyDeleteFcn)
    
    
    sX = .7;
    sY = .4;
    spX = .15;
    spY = .3;
    typeMenu = uicontrol('Parent', fig, 'Style', 'popupmenu', 'String', 'Resolutions', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15);
    
    %typesList = {'Box','Free Plane', 'Sphere'}; More to be implemented
    %soon, But Box is the staple for most people. More advanced masks can
    %still be created manually.
    typesList = {'Box'};
    
    set(typeMenu, 'String', typesList);
    
    
    sX = .3;
    sY = .20;
    spX = .2;
    spY = sY;
    okButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'OK', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @okButton_Callback);
    
    sX = .3;
    sY = .20;
    spX = .5;
    spY = sY;
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'cancel', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @cancelButton_Callback);
    
    
    type = 'Box';

    %callbacks
    function okButton_Callback(hObject, eventData, handles)
        
        ind = get(typeMenu, 'Value');
        
        if     ind == 1 % Box
            disp(type);
            type = 'Box';
            delete(fig);
            AddBoxMaskWindow(viewAxis, offset,GetActualSize, AddGeneratedLayer);
            
        elseif ind == 2 % Free Plane
            disp(type);
            type = 'Free Plane';
            delete(fig);
            % Not implemented yet
            
        elseif ind == 3 % Sphere
            disp(type);
            type = 'Sphere';
            delete(fig);
            % Not implemented yet
            
        end

        disp(type);
        
    end
    
    function cancelButton_Callback(hObject, eventData, handles)
        delete(fig);
    end

end

