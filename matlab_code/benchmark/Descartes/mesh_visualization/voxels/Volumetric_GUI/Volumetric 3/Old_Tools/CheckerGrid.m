%Author: James W. Ryland
%June 14, 2012

function [ checkers ] = CheckerGrid( sc, sx, sy )
%CHECKERGRID creates a binary CheckerGrid of a specified size.
%   SC is the length of each checker. SX specifies the x size and SY
%   specifies the y size. This is used to create the background for some of
%   the previews used by the the editor windows.

    [X Y] = meshgrid(1:sy, 1:sx);
    
    checkers = xor((mod(round(X/sc), 2)==0),(mod(round(Y/sc), 2)==0));
    
    %imagesc(checkers);
    %axis('off');
    %axis('image');
    %colormap('gray');

end
