function [ fig ] = PointerWindow( GetCropOffset, viewAxis )
%POINTERWINDOW Summary of this function goes here
%   Detailed explanation goes here


    % Initialize graphics
    scr = get(0,'ScreenSize');
    
    width = 300; 
    height = 400;
    
    
    pos = [ 400+100 (scr(4)-height-700)];


    title = '3D Pointer Control';
    fig = figure('Name',title, 'Resize', 'on', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height], 'CloseRequestFcn', @MyDeleteFcn, 'DeleteFcn', @MyDeleteFcn);
    
    sPosScaled = [0, 0, 0];
    
    % X SLIDER
    sX = .7;
    sY = .10;
    spX = .2;
    spY = .8;
    sliderX = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', sPosScaled(1), 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    
    sX = .1;
    sY = .10;
    spX = spX-.1;
    sliderXText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(sliderXText, 'String', 'X', 'FontSize', 20, 'foregroundcolor', 'red');
    
    % Y SLIDER
    sX = .7;
    sY = .10;
    spX = .2;
    spY = .8-.2;
    sliderY = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', sPosScaled(2), 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    
    sX = .1;
    sY = .10;
    spX = spX-.1;
    sliderYText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(sliderYText, 'String', 'Y', 'FontSize', 20, 'foregroundcolor', 'green');
    
    % Z SLIDER
    sX = .7;
    sY = .10;
    spX = .2;
    spY = .8-.4;
    sliderZ = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', sPosScaled(3), 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    
    sX = .1;
    sY = .10;
    spX = spX-.1;
    sliderZText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(sliderZText, 'String', 'Z', 'FontSize', 20, 'foregroundcolor', 'blue');
    
    
    sX = .7;
    sY = .10;
    spX = .1;
    spY = .8-.6;
    positionText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(positionText, 'String', 'X:0  Y:0  Z:0', 'FontSize', 20, 'foregroundcolor', [1 0 1]);
    
    
    sX = .1;
    sY = .10;
    spX = .8;
    spY = .8-.6;
    pointerOn = 'off';
    pointerOnButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @pointerOnButton_Callback);
    set(pointerOnButton, 'String', pointerOn, 'FontSize', 12);
    
    

    lineX= [];
    lineY= [];
    lineZ= [];
    
    xLineP = [0, 0, 0;
              1, 0, 0];
    yLineP = [0, 0, 0;
              0, 1, 0];
    zLineP = [0, 0, 0;
              0, 0, 1];
          
    cropOffset = [0 0 0];
          
    [lineX, lineY, lineZ] = buildLines();
    
    function [lineX, lineY, lineZ] = buildLines()
        %lines
    
        cropOffset = GetCropOffset();
        
        xLineP = [0, 0, 0;
                  1, 0, 0];
        lineX = line(xLineP(:,1), xLineP(:,2), xLineP(:,3), 'Parent', viewAxis, 'color', 'red', 'visible', pointerOn);

        yLineP = [0, 0, 0;
                  0, 1, 0];
        lineY = line(yLineP(:,1), yLineP(:,2), yLineP(:,3), 'Parent', viewAxis, 'color', 'green', 'visible', pointerOn);

        zLineP = [0, 0, 0;
                  0, 0, 1];
        lineZ = line(zLineP(:,1), zLineP(:,2), zLineP(:,3), 'Parent', viewAxis, 'color', 'blue', 'visible', pointerOn);

    end
    
    
    function slider_Callback(hObject, eventData, handles)
        
        sPosScaled = [get(sliderX, 'Value'), get(sliderY, 'Value'), get(sliderZ, 'Value')];
        
        sPosDisp = [0 0 0 ];
        
        if ~isempty(viewAxis)
            
            
            try
                delete(lineX, lineY, lineZ);
            catch
                disp('No prob bob');
            end
            
            [lineX, lineY, lineZ] = buildLines();
            
            xl = xlim(viewAxis);
            yl = ylim(viewAxis);
            zl = zlim(viewAxis);
            
            sPosDisp = round(sPosScaled.*[xl(2) yl(2) zl(2)]);
            pd = sPosDisp;
            
            xLineP = [0,        pd(2),      pd(3);
                      xl(2),    pd(2),      pd(3)];
                  
            yLineP = [pd(1),     0,         pd(3);
                      pd(1),     yl(2),     pd(3)];
                  
            zLineP = [pd(1),     pd(2),     0;
                      pd(1),     pd(2),     zl(2)];      

            set(lineX, 'XData', xLineP(:,1), 'YData', xLineP(:,2), 'ZData', xLineP(:,3), 'visible', pointerOn);
            set(lineY, 'XData', yLineP(:,1), 'YData', yLineP(:,2), 'ZData', yLineP(:,3), 'visible', pointerOn);
            set(lineZ, 'XData', zLineP(:,1), 'YData', zLineP(:,2), 'ZData', zLineP(:,3), 'visible', pointerOn);
            
        else
            
            sPosDisp = round(sPosScaled*100)/100;
            
        end
        
        cropOffset
        
        sPosDispOff = sPosDisp+cropOffset;
        
        set(positionText, 'String', ['X:' num2str(sPosDispOff(1)) '  Y:' num2str(sPosDispOff(2)) '  Z:' num2str(sPosDispOff(3))] );
    end

    function pointerOnButton_Callback(hObject, eventData, handles)
        
        if strcmp(pointerOn, 'on')
            pointerOn = 'off';
        else
            pointerOn = 'on';
        end
            
        set(pointerOnButton, 'String', pointerOn, 'FontSize', 12);
        
        
        slider_Callback([],[],[]);
        
    
    end


    function MyDeleteFcn(hObject, eventData)
        try
            delete(lineX);
            delete(lineY);
            delete(lineZ);
        catch
        end
        delete(fig);
    end
    
end

