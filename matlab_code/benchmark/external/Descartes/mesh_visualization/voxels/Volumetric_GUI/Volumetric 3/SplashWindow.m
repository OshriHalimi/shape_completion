function [  ] = SplashWindow( )
  
    scr = get(0,'ScreenSize');
    
    logoIm = [];
    
    % Choose appropriate logo size
    % premade for best quality
    if scr(3)>=1800
        logoIm = imread('CroppedLogo.jpg');
        
    elseif scr(3)>=1200
        logoIm = imread('CroppedLogoSizeA.jpg');
        
    else scr(3)>=600
        logoIm = imread('CroppedLogoSizeB.jpg');
    end
    
    [sY, sX, ~]  = size(logoIm);
    
    pos(1) = round(scr(3)/2-sX/2);
    pos(2) = round(scr(4)/2-sY/2);
    
    
    title = 'Volumetric 3';
    fig = figure('Name',title, 'Resize', 'on', 'MenuBar', 'None', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) sX sY]);
    
    viewAxis = axes('Parent', fig, 'Units', 'Normalized', 'Position', [ 0, 0, 1, 1]);
    set(viewAxis, 'XTick', [], 'YTick', [], 'ZTick', []);
    set(viewAxis, 'Projection', 'Perspective');
    set(viewAxis, 'color', 'black');
    
    image(logoIm);

    pause(3)
    
    delete(fig);
    clear('logoIm');
end

