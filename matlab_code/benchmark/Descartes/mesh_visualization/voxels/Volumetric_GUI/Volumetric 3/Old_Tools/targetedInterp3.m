%Author: James W. Ryland
%June 14, 2012

function [ vDataTarg  voxNum c] = targetedInterp3( vData, voxTarget )
%TARGETEDINTERP3 interpolates a volume to an approximate voxel number
%   vData is the volume that is to be interpolated. voxTarget is the
%   approximate voxel target that the algorithm will try to interpolate
%   towards.This function is used by ResizeWindow and by ManipulateWindow
    isUINT8 = 0;
    if isa(vData,'uint8')
        isUINT8 = 1;
        vData = double(vData);
    end
    

    [xs ys zs] = size(vData);
    Vold = (xs-1)*(ys-1)*(zs-1);
    
    c = (voxTarget/Vold)^(1/3)

    xst = 1/c;
    yst = 1/c;
    zst = 1/c;
    xoff = mod(xs-1,1/c);
    yoff = mod(ys-1,1/c);
    zoff = mod(zs-1,1/c);
    xi = (1:xst:xs)+xoff/2;
    yi = (1:yst:ys)+yoff/2;
    zi = (1:zst:zs)+zoff/2;
    xil = size(xi,2);
    yil = size(yi,2);
    zil = size(zi,2);
    
    xs;
    ys;
    zs;
    %size(1:xs,2)
    %size(1:ys,2)
    %size(1:zs,2)
    
    
    [YIo XIo ZIo] = meshgrid(1:(ys), 1:(xs), 1:(zs));
    
    size(XIo);
    size(YIo);
    size(ZIo);
    
    [YI XI ZI] = meshgrid(xi, yi, zi);
    
    size(XI);
    size(YI);
    size(ZI);
    
    vDataTarg = permute(interp3(YIo, XIo, ZIo, vData, XI, YI, ZI, 'linear', 2000), [2 1 3]);

    size(vDataTarg);
    
    voxNum = xil*yil*zil
    
    if isUINT8
        vDataTarg = uint8(vDataTarg);
    end
    
    
end