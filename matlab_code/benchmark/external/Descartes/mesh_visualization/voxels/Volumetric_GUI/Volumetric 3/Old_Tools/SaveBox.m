%Author: James W. Ryland
%June 14, 2012

function [  ] = SaveBox( fig, pos, getCAV, getLayers, settings)
%SAVEBOX gives the user options to save the workspace of layers created by
%LayerBox or the combined CAV output of all the layers.
%   fig will be the parent to all of the graphical components of SaveBox.
%   Pos is the position in the figure that SaveBox will occupy. getCAV is a
%   function handle that allow SaveBox to retireve the combined CAV output
%   of all the layers. getLayers is a function handle that retrieves the
%   layers structure from LayerBox.

    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    
    savePanel = uipanel('Parent', fig, 'Title', 'Save Options', 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 90 ] );
    
    saveLayersButton = uicontrol('Parent', savePanel, 'Style', 'pushbutton', 'String', 'Save Layers' ,'Position', [10 50 260 20],...
        'CallBack', @saveLayersButton_CallBack);
    
    saveCAVButton = uicontrol('Parent', savePanel, 'Style', 'pushbutton', 'String', 'Save CAV Output' ,'Position', [10 20 260 20],...
        'CallBack', @saveCAVButton_CallBack);
    
    %Cross Platform Formating
    uicomponents = [saveLayersButton saveCAVButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
     
    
    
    function saveLayersButton_CallBack(h, EventData)
        layers = getLayers();
        SaveWindow('Save Layers', [],layers);
    end

    function saveCAVButton_CallBack(h, EventData)
        CAV = getCAV();
        SaveWindow('Save Color Alpha Volume', [], CAV);
    end
    

end

