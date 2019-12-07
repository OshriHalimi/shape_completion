%Author: James W. Ryland
%June 14, 2012

function [ ] = CreateRenderBox(fig, pos, getVolumeCA )
%CREATERENDERBOX creates a box that allows the user to create a Rendering
%   fig is the parent figure that CreateRenderBox will reside in. Pos is the
%   position that CreateRenderBox will occupy in the parent figure.
%   getVolumeCA is a function that returns the current Color Alpha Volume
%   to be rendered by the to be created render box.


    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    savePanel = uipanel('Parent', fig, 'Title', 'Render Options', 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 140 ] );
    
    rendPreviewButton = uicontrol('Parent', savePanel, 'Style', 'pushbutton', 'String', 'Create Normal Preview Render','Position', [10 50 260 20],...
        'CallBack', @rendPreviewButton_CallBack);
    

    
    
    function rendPreviewButton_CallBack(h, EventData)
       
        [updateVolumeCA] = SmallRenderBox([],'Render Window',@refreshResponse);
        
        function refreshResponse()
            updateVolumeCA(getVolumeCA());
        end
        
    end
    
end

