%Author: James W. Ryland
%June 14, 2012

function [ ] = LayersWindow(pos, title )
%LAYERSWINDOW allows a user to create different types of CAV layers, to
%create a render window to display a combination of the different layers,
%and to save that CAV output or the layer workspace.
%   Pos is the position the figure will occupy on the desktop. This
%   function makes use of LayerBox, CreateRenderBox, and SaveBox.


    if isempty(pos)
        
        scr = get(0,'ScreenSize');
        
        pos = [ 1 (scr(4)-580)];
        
    end

    fig = figure('Name',title, 'MenuBar', 'None', 'Resize', 'off', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) 280 510]);

    figureAdjust(fig);
    
    [getVolumeCA getLayers] = LayerBox(fig,[1 230],[]);
    
    CreateRenderBox(fig, [1 90], getVolumeCA);
    
    SaveBox(fig, [1 1], getVolumeCA, getLayers);
    
end
