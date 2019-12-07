%Author: James W. Ryland
%June 14, 2012

function [ cropX, cropY, cropZ] = Crop3( matrix3 )
%CROP3 crops a matrix on all 3 dimesnions to the most external indicies
%with non-zero values.
%   
% 0 0 0 0
% 0 1 0 0   becomes ->  1 0
% 0 2 1 0               2 1
% 0 0 0 0

    matrix3 = matrix3~=1;

    xSum = sum(sum(matrix3,2),3);
    ySum = sum(sum(matrix3,1),3);
    zSum = sum(sum(matrix3,1),2);
    
    minX = min(find(xSum));
    maxX = max(find(xSum));
    minY = min(find(ySum));
    maxY = max(find(ySum));
    minZ = min(find(zSum));
    maxZ = max(find(zSum));
    
    cropX = (minX):(maxX);
    cropY = (minY):(maxY);
    cropZ = (minZ):(maxZ);
    
    clear matrix3;
    
    %Matrix3 = matrix3(cropX, cropY, cropZ);
    
end