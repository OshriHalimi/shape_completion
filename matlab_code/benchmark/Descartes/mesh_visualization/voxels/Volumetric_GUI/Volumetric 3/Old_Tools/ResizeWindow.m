%Author: James W. Ryland
%June 14, 2012

function [  ] = ResizeWindow( pos )
%RESIZEWINDOW allows a user to interpolate a set of volumes to a desired
%approximate voxel count
%   POS is the position that ResizeWindow will occupy on the desktop.
%   ResizeWindow makes use of ResizeBox, and FileSelectBox.
    if isempty(pos)
        pos = [0 0];
    end

    refVolume = [];
    
    targVoxNum = [];

    fig = figure('Name', 'Resize Window', 'Resize', 'off', 'NumberTitle', 'off','Position', [pos(1) pos(2) 280 440],...
        'MenuBar', 'none');

    figureAdjust(fig);
    
    [updateReferenceHandle getTargetVoxNumHandle getVoxDimHandle getPrefixHandle] = ResizeBox(fig,[1 280],@applyResize);
    
    [getFileContentsHandle] = FileSelectBox(fig, [1 1], 3, @updateReference);

    
    % updates
    function updateReference(newReference)
        refVolume = newReference;
        updateReferenceHandle(newReference);
    end
    
    function applyResize()
        [vols fileNames ] = getFileContentsHandle();
        voxTarget = getTargetVoxNumHandle();
        [prefix] = getPrefixHandle();
        [voxDim] = getVoxDimHandle();
        
        for f = 1:size(fileNames,2)
            
            
            interpVol = targetedInterp3NonCube(double(vols{f}), voxTarget, voxDim);
        
            [pathstr, name, ext] = fileparts(fileNames{f})
            
            newFileName = strcat(prefix, name, '.mat');
            disp(newFileName);
            save(newFileName, 'interpVol');
            
        end
    end

end

