%Author: James W. Ryland
%June 14, 2012

function [boundsX boundsY boundsZ  ] = Bounds3( matrix3 )
%Bounds3 finds the nonzero bounds of a volume.
%   This can be used for finding cropping values and comparing them accross
%   matrices.
    matrix3 = matrix3~=0;

    xSum = sum(sum(matrix3,2),3);
    ySum = sum(sum(matrix3,1),3);
    zSum = sum(sum(matrix3,1),2);
    
    minX = min(find(xSum));
    maxX = max(find(xSum));
    minY = min(find(ySum));
    maxY = max(find(ySum));
    minZ = min(find(zSum));
    maxZ = max(find(zSum));
    
    boundsX = [(minX) (maxX)];
    boundsY = [(minY) (maxY)];
    boundsZ = [(minZ) (maxZ)];
    
    
end

