%Author: James W. Ryland
%June 14, 2012

function [  ] = RotateWindow( pos )
%ROTATEWINDOW allows a user perform a series of 90 degree rotations around
%the three major axes on source volumes in the form of .mat files and .img
%files.
%   Pos is the position rotate window will occupy. RotateWindow makes use
%   of SmallRenderBox, FileSelectBox and RotateBox.
    if isempty(pos)
        pos = [0 0];
    end
    
    refVolume = [];
    
    rotVolume = [];
    
    addVolumes = {};
    
    fig = figure('Name', 'Rotate Window', 'Resize', 'off', 'NumberTitle', 'off','Position', [pos(1) pos(2) 280 420]);

    figureAdjust(fig);
    
    [getRotationHandle getPrefixHandle] = RotateBox(fig, [1 280], @rotateReference, @applyRotations, @resetRotations);
    
    [getFileContentsHandle] = FileSelectBox(fig, [1 1], 3, @updateReference);
    
    [updateVolumeCAHandle dum drawBoundsHandle] = ExploreRenderBox([280 pos(2)],'Rotate View',[]);
    
    drawBoundsHandle('cross');
    
    
    function rotateReference()
        [dim1s dim2s ntimes] = getRotationHandle();

        tempVol = refVolume;
        for i=1:size(dim1s, 2);
            tempVol = RotateOnPlane(tempVol,dim1s(i), dim2s(i), ntimes(i));
        end
        rotVolume = uint8(tempVol);
        
        CAV(:,:,:,1) = rotVolume;
        CAV(:,:,:,2) = rotVolume;
        CAV(:,:,:,3) = rotVolume;
        CAV(:,:,:,4) = rotVolume;
        
        updateVolumeCAHandle(CAV);
        drawBoundsHandle('cross');
    end

    function applyRotations()
        [vols fileNames ] = getFileContentsHandle();
        [prefix] = getPrefixHandle();
        [dim1s dim2s ntimes] = getRotationHandle();
        
        
        for f = 1:size(fileNames,2)
            
            tempVol = vols{f};
            for v=1:size(dim1s, 2);
                tempVol = RotateOnPlane(tempVol,dim1s(v), dim2s(v), ntimes(v));
            end
            rotation = tempVol;
            
            [pathstr, name, ext] = fileparts(fileNames{f});
            
            newFileName = strcat(prefix, name, '.mat');
            disp(newFileName);
            save(newFileName, 'rotation');
            
        end
    end

    function updateReference(newRefVolume)
        minV = double(min(min(min(newRefVolume))));
        maxV = double(max(max(max(newRefVolume))));
        
        newRefUint8 = uint8((double(newRefVolume)-minV)/(maxV-minV)*255);
        
        refVolume = newRefUint8;
        rotVolume = refVolume;
        
        CAV(:,:,:,1) = rotVolume;
        CAV(:,:,:,2) = rotVolume;
        CAV(:,:,:,3) = rotVolume;
        CAV(:,:,:,4) = rotVolume;
        
        updateVolumeCAHandle(CAV);
        drawBoundsHandle('cross');
        
    end

    function resetRotations(newRefVolume)
        minV = double(min(min(min(newRefVolume))));
        maxV = double(max(max(max(newRefVolume))));
        
        newRefUint8 = uint8((double(newRefVolume)-minV)/(maxV-minV)*255);
        
        
        refVolume = newRefUint8;
        rotVolume = refVolume;
        
        CAV(:,:,:,1) = rotVolume;
        CAV(:,:,:,2) = rotVolume;
        CAV(:,:,:,3) = rotVolume;
        CAV(:,:,:,4) = rotVolume;
        
        updateVolumeCAHandle(CAV);
        drawBoundsHandle('cross');
    end
    
end

