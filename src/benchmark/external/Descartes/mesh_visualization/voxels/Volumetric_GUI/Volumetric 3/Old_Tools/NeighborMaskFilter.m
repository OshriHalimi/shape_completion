%Author
%James Ryland


function [ newMaskVol ] = NeighborMaskFilter(maskVol, filtWidth, numNeighbors )
%used to get rid of extranious noise voxels in a masking volume.
%finds voxels without enough neighbors suggesting that thay are not part of
%the main structure of the volume. The mask volume needs to be a binary
%array or the utility may not work or at least it should only be zeros and
%ones.
    
    filt = fspecial3('ellipsoid', filtWidth);
    
    disp('Neighborhood Size');
    
    disp(nnz(filt));
    
    filt = filt*nnz(filt);
    
    disp('Num Vox Before');
    disp(nnz(maskVol));
        
    newMaskVol = convn(double(maskVol), filt, 'same').*double(maskVol)>=numNeighbors;
    
    disp('Num Vox After');
    disp(nnz(newMaskVol));
    
    
end

