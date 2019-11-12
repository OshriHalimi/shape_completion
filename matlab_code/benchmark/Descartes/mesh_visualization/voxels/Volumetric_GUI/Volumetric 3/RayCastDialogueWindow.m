function [  ] = RayCastDialogueWindow(pos, DataStruct, GetCamProperties )
% Here we are going to request information before precieding with the
% raycast.
    
    scr = get(0,'ScreenSize');

    width = 300; 
    height = 300;
    
    pos = [ 400+100+600 (scr(4)-height)];


    title = 'Ray Cast Settings';
    fig = figure('Name',title, 'Resize', 'on', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height]);  % 'CloseRequestFcn', @MyCloseRequestFcn, 'DeleteFcn', @MyDeleteFcn)
    
    
    sX = .7;
    sY = .4;
    spX = .15;
    spY = .4;
    rezMenu = uicontrol('Parent', fig, 'Style', 'popupmenu', 'String', 'Resolutions', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15);
    set(rezMenu, 'String', {'200x200 - Thumbnail', '400x400 - Standard',  '600x600 - High Rez',  '800x800 - Very High Rez', '1000x1000 - Should use with high density volume'});
    
    
    % Shading Sliders
    sX = .4;
    sY = .2;
    spX = .45;
    spY = .4;
    shadingLevelSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 0, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    
    spY = .25;
    shadingContSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 0, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY])
    
    sX = .3;
    sY = .1;
    spX = .15;
    spY = .5;
    shadingLevelLabel = uicontrol('Parent', fig, 'Style', 'text','Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(shadingLevelLabel, 'String', 'Shading', 'fontsize', 12);
    
    sX = .3;
    sY = .1;
    spX = .15;
    spY = spY-.15;
    shadingContLabel = uicontrol('Parent', fig, 'Style', 'text','Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(shadingContLabel, 'String', 'Sh Conrast', 'fontsize', 12);
    
    
    
    
    
    sX = .3;
    sY = .10;
    spX = .2;
    spY = sY;
    okButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'OK', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @okButton_Callback);
    
    sX = .3;
    sY = .10;
    spX = .5;
    spY = sY;
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'cancel', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @cancelButton_Callback);
    
    

    %callbacks
    function okButton_Callback(hObject, eventData, handles)
        
        ind = get(rezMenu, 'Value');
        
        rezes = [200, 400, 600, 800, 1000];
        
        rez = rezes(ind);
        
        
        shadingInfo.level = get(shadingLevelSlider, 'value');
        
        shadingInfo.cont = get(shadingContSlider, 'value');
        
        delete(fig);
        pause(.001);
        
        RayCastWindow( pos, DataStruct, GetCamProperties, rez, shadingInfo );
        
        clear('DataStruct', 'DataStruct', 'GetCampProperties');
        

    end
    
    function cancelButton_Callback(hObject, eventData, handles)
        clear('DataStruct', 'DataStruct', 'GetCampProperties');
        delete(fig);
    end
    
end

