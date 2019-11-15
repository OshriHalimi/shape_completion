%Author: James W. Ryland
%June 14, 2012

function [ ] = targetedInterp3Dir( prefix, voxTarget )
%TARGETEDINTERP3DIR interpolates a directory .mat or .img files to an
%approximate desired voxal number
%   This file may or may not be included in the final release of Volumetric
%   as it is not a GUI utility. Its functionality is also redundent given
%   that ResizeWindow performs a similar operation but with more control
%   given to the user.

    dirStruct = dir;
    dirName = {dirStruct.name}';
    dirIsDir = {dirStruct.isdir}';
    
    for i=1:size(dirName)
        disp(dirName{i});
        if dirIsDir{i}==0
            [dum name ext] = fileparts(dirName{i});
            disp(name);
            disp(ext);
            if strcmp(ext, '.img')
                temp = analyze75read(dirName{i});
                [temp voxNum] = mriReorient(temp, voxTarget);
                save([prefix name '.mat'],'temp');
            elseif strcmp(ext, '.mat')
                structCell = struct2cell(load(dirName{i}));
                temp = structCell{1};
                [temp voxNum] = targetedInterp3(temp, voxTarget);
                save([prefix name '.mat'],'temp');
            end
        end
    end

end

function [ vDataTarg  voxNum] = targetedInterp3( vData, voxTarget )
%TARGETEDINTERP3 Summary of this function goes here
%   Detailed explanation goes here
    
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
end