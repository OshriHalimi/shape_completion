function [ ] = AddBoxMaskWindow(viewAxis,offset,GetActualSize,AddGeneratedLayer)
    
    

    % Initialize graphics
    scr = get(0,'ScreenSize');
    
    width = 300; 
    height = 400;
    
    
    pos = [ 400+100 (scr(4)-height-700)];


    title = 'Box Mask Settings';
    fig = figure('Name',title, 'Resize', 'on', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height], 'CloseRequestFcn', @MyDeleteFcn, 'DeleteFcn', @MyDeleteFcn);
    

    
    % X SLIDER
    sX = .7;
    sY = .10;
    spX = .2;
    spY = .8;
    sliderX1 = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 0, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    spY= spY-.05;
    sliderX2 = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 1, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    
    
    sX = .1;
    sY = .10;
    spX = spX-.1;
    spY = .8;
    sliderXText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(sliderXText, 'String', 'X', 'FontSize', 20, 'foregroundcolor', 'red');
    
    % Y SLIDER
    sX = .7;
    sY = .10;
    spX = .2;
    spY = .8-.2;
    sliderY1 = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 0, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    spY = .8-.2-.05;
    sliderY2 = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 1, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    
    
    sX = .1;
    sY = .10;
    spX = spX-.1;
    spY = .8-.2;
    sliderYText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(sliderYText, 'String', 'Y', 'FontSize', 20, 'foregroundcolor', 'green');
    
    % Z SLIDER
    sX = .7;
    sY = .10;
    spX = .2;
    spY = .8-.4;
    sliderZ1 = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 0, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    spY = .8-.4-.05;
    sliderZ2 = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', 1, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @slider_Callback);
    
    
    sX = .1;
    sY = .10;
    spX = spX-.1;
    spY = .8-.4;
    sliderZText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(sliderZText, 'String', 'Z', 'FontSize', 20, 'foregroundcolor', 'blue');
    
    
    sX = .3;
    sY = .10;
    spX = .2;
    spY = .1;
    createButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Create', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @createButton_Callback);
    
    sX = .3;
    sY = .10;
    spX = .5;
    spY = .1;
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Cancel', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @MyDeleteFcn);
    
    

    
    
    xl = xlim(viewAxis);
    yl = ylim(viewAxis);
    zl = zlim(viewAxis);
    
    [p1,p2,p3,p4,p5,p6] = MakeBox();
    
    
    % Make Box
    function [p1, p2, p3, p4, p5, p6] = MakeBox()
        
        xb = [get(sliderX1,'value')  get(sliderX2,'value')]*(xl(2));
        yb = [get(sliderY1,'value')  get(sliderY2,'value')]*(yl(2));
        zb = [get(sliderZ1,'value')  get(sliderZ2,'value')]*(zl(2));
        
        % planes tangent to Z
        p1= patch(  [xb(1) xb(2) xb(2) xb(1)],...
                    [yb(1) yb(1) yb(2) yb(2)],... 
                    [zb(1) zb(1) zb(1) zb(1)], 'k', 'parent', viewAxis, 'facecolor', [0,0,0], 'edgecolor', [1,1,1]);

        p2= patch(  [xb(1) xb(2) xb(2) xb(1)],...
                    [yb(1) yb(1) yb(2) yb(2)],... 
                    [zb(2) zb(2) zb(2) zb(2)], 'k', 'parent', viewAxis, 'facecolor', [0,0,0], 'edgecolor', [1,1,1]);

            % Planes tangent to Y
        p3= patch(  [xb(1) xb(2) xb(2) xb(1)],...
                    [yb(1) yb(1) yb(1) yb(1)],... 
                    [zb(1) zb(1) zb(2) zb(2)], 'k', 'parent', viewAxis, 'facecolor', [0,0,0], 'edgecolor', [1,1,1]);

        p4= patch(  [xb(1) xb(2) xb(2) xb(1)],...
                    [yb(2) yb(2) yb(2) yb(2)],... 
                    [zb(1) zb(1) zb(2) zb(2)], 'k', 'parent', viewAxis, 'facecolor', [0,0,0], 'edgecolor', [1,1,1]);

            % Planes tangent to X
        p5= patch(  [xb(1) xb(1) xb(1) xb(1)],...
                    [yb(1) yb(2) yb(2) yb(1)],... 
                    [zb(1) zb(1) zb(2) zb(2)], 'k', 'parent', viewAxis, 'facecolor', [0,0,0], 'edgecolor', [1,1,1]);

        p6= patch(  [xb(2) xb(2) xb(2) xb(2)],...
                    [yb(1) yb(2) yb(2) yb(1)],... 
                    [zb(1) zb(1) zb(2) zb(2)], 'k', 'parent', viewAxis, 'facecolor', [0,0,0], 'edgecolor', [1,1,1]);

    end

    
    
    

    % Callbacks
    function slider_Callback(hObject, eventData, handles)
        

        xb = [get(sliderX1,'value')  get(sliderX2,'value')]*(xl(2));
        yb = [get(sliderY1,'value')  get(sliderY2,'value')]*(yl(2));
        zb = [get(sliderZ1,'value')  get(sliderZ2,'value')]*(zl(2));
        
        set(p1, 'xdata',[xb(1) xb(2) xb(2) xb(1)],...
                'ydata',[yb(1) yb(1) yb(2) yb(2)],... 
                'zdata',[zb(1) zb(1) zb(1) zb(1)])
            
        set(p2, 'xdata',[xb(1) xb(2) xb(2) xb(1)],...
                'ydata',[yb(1) yb(1) yb(2) yb(2)],... 
                'zdata',[zb(2) zb(2) zb(2) zb(2)])
        
        set(p3, 'xdata',[xb(1) xb(2) xb(2) xb(1)],...
                'ydata',[yb(1) yb(1) yb(1) yb(1)],... 
                'zdata',[zb(1) zb(1) zb(2) zb(2)])
            
        set(p4, 'xdata',[xb(1) xb(2) xb(2) xb(1)],...
                'ydata',[yb(2) yb(2) yb(2) yb(2)],... 
                'zdata',[zb(1) zb(1) zb(2) zb(2)])
        
        set(p5, 'xdata',[xb(1) xb(1) xb(1) xb(1)],...
                'ydata',[yb(1) yb(2) yb(2) yb(1)],... 
                'zdata',[zb(1) zb(1) zb(2) zb(2)])
        
        set(p6, 'xdata',[xb(2) xb(2) xb(2) xb(2)],...
                'ydata',[yb(1) yb(2) yb(2) yb(1)],... 
                'zdata',[zb(1) zb(1) zb(2) zb(2)])
        
    end
    
    function createButton_Callback(hObject, eventData, handles)
        
        % cropped bounds using offset 
        xb = [get(sliderX1,'value')  get(sliderX2,'value')]*(xl(2))+offset(1);
        yb = [get(sliderY1,'value')  get(sliderY2,'value')]*(yl(2))+offset(2);
        zb = [get(sliderZ1,'value')  get(sliderZ2,'value')]*(zl(2))+offset(3);
        
        MyDeleteFcn([],[]);
        
        %Full index matrices
        [sizeVec] = GetActualSize();
        
        offset
        sizeVec
        xi=sizeVec(1);
        yi=sizeVec(2);
        zi=sizeVec(3);
        
        
        [X,Y,Z] = meshgrid(1:xi, 1:yi, 1:zi);
        
        % Should already set everything correctly !!! WOOT!!
        boundBox = ( xb(1)<X & X<xb(2) ) &...
                   ( yb(1)<Y & Y<yb(2) ) &...
                   ( zb(1)<Z & Z<zb(2) );
        
        boundBox = permute(boundBox, [2 1 3]);
        
        genLayer.name = 'Box Mask Layer';
        genLayer.AlphaMap = zeros(64,1);
        genLayer.ColorMap = zeros(64,3);
        genLayer.ScaledVolAlpha = boundBox;
        genLayer.ScaledVolColor = boundBox;
        
        clear('boundBox', 'X','Y','Z');
        
        
        AddGeneratedLayer(genLayer);
        
    end

    
    function MyDeleteFcn(hObject, eventData)
        try
            delete(p1,p2,p3,p4,p5,p6, fig);
        catch
            disp('Some things may be missing');
        end
    end

    

end

