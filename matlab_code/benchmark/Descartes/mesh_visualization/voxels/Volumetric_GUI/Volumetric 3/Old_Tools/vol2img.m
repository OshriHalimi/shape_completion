function [  ] = vol2img( volumeToConvert, fileName)
%MAT2IMG convers a .mat volumetric matrix to a .img file
%   This does not create a header file so you will need to find an
%   appropriate .hdr file to associate with this one.


    fname = [fileName '.img'];
    fid=fopen(fname,'w','l');
    perm = permute(volumeToConvert, [2 1 3]);
    fwrite(fid,perm,'int16');
    
end
