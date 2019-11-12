function [voxNum c newDim] = targetedInterp3NonCubeEst( vData, voxTarget, voxDim)
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
    
    dx = voxDim(1);
    dy = voxDim(2);
    dz = voxDim(3);
    
    [xs ys zs] = size(vData);
    Vold = (xs-1)*dx*(ys-1)*dy*(zs-1)*dz;
    
    c = (voxTarget/Vold)^(1/3)

    
    xst = 1/(c*dx);
    yst = 1/(c*dy);
    zst = 1/(c*dz);
    xoff = mod(xs-1,1/(c*dx));
    yoff = mod(ys-1,1/(c*dy));
    zoff = mod(zs-1,1/(c*dz));
    xi = (1:xst:xs)+xoff/2;
    yi = (1:yst:ys)+yoff/2;
    zi = (1:zst:zs)+zoff/2;
    xil = size(xi,2);
    yil = size(yi,2);
    zil = size(zi,2);
    
    sxi = size(xi,2)
    syi = size(yi,2)
    szi = size(zi,2)
    
    newDim = [sxi syi szi];
    
    %[YIo XIo ZIo] = meshgrid(1:(ys), 1:(xs), 1:(zs));
    
    %size(XIo);
    %size(YIo);
    %size(ZIo);
    
    %[YI XI ZI] = meshgrid(xi, yi, zi);
    
    %size(XI);
    %size(YI);
    %size(ZI);
    
    %vDataTarg = permute(interp3(YIo, XIo, ZIo, vData, XI, YI, ZI, 'linear', 2000), [2 1 3]);

    %size(vDataTarg);
    
    voxNum = xil*yil*zil
    
    %if isUINT8
    %    vDataTarg = uint8(vDataTarg);
    %end
    
    
end