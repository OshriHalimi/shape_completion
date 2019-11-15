%Author: James W. Ryland
%June 14, 2012

function [ ] = figureAdjust(figureHandle)
%FIGUREDADJUST adjusts a figure's position to fit on the screen without
%changing the the size of the figure.
%   figureHanle is the handle to the figure that needs to be moved to
%   another position on the screen.

    rect = get(figureHandle, 'OuterPosition');
    pos = get(figureHandle, 'Position');
    screen = get(0, 'ScreenSize')
    
    adjust = [0 0];
    
    vert(1) = rect(1);
    vert(2) = rect(2);
    vert(3) = rect(3)+rect(1);
    vert(4) = rect(4)+rect(2);
    
    vert
    
    adjust = [0 0];
    
    adjust(1) = -vert(1)*(vert(1)<1);
    adjust(2) = -vert(2)*(vert(2)<1);
    adjust(1) = -(vert(3)-screen(3))*(vert(3)>screen(3))+adjust(1);
    adjust(2) = -(vert(4)-screen(4))*(vert(4)>screen(4))+adjust(2);
    
    screenTooSmall = ((vert(1)<1)&&(vert(3)>screen(3)))||((vert(2)<1)&&(vert(4)>screen(4)));
    
    adjust
    
    if screenTooSmall
        delete(fig);
        disp('Error your screen resolution is too small.');
    else
        newPos = [(pos(1)+adjust(1)) (pos(2)+adjust(2)) pos(3) pos(4)]
        set(figureHandle, 'Position', newPos);
    end
    
end

