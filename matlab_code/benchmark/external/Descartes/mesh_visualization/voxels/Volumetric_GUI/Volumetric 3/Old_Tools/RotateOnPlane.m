%Author: James W. Ryland
%June 14, 2012

function [ matrixOut ] = RotateOnPlane(matrixIn, dim1, dim2, ntimes)
%ROTATEONAXIS performs a 90 degree rotation around either the X, Y, or Z
%axis on a source volume.
%   
    
    original = [1 2 3];
    permVec = [1 2 3];
    permVec = permVec((find((dim1~=permVec)&(dim2~=permVec))));
    permVec = [dim1 dim2 permVec];
    matrixPermed = permute(matrixIn , permVec);
    numPlanes = size(matrixPermed, 3);

    matrixOutP = permute(zeros(size(matrixPermed)), [2 1 3]);

    for i=1:numPlanes
        matrixOutP(:,:,i) = rot90(matrixPermed(:,:,i), ntimes);
    end
    
    if sum(original==permVec)>=1
        matrixOut = permute(matrixOutP, permVec);
    elseif sum(permVec==([2 3 1]))
        matrixOut = permute(matrixOutP, [3 1 2]);
    elseif sum(permVec==([3 1 2]))
        matrixOut = permute(matrixOutP, [2 3 1]);
    end

end