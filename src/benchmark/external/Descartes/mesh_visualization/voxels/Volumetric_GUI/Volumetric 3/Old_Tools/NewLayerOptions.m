%Author: James W. Ryland
%June 14, 2012

function [ ] = NewLayerOptions(pos, externalUpdateHandle)
%NEWLAYEROPTIONS this function gives the user a choice of what type of
%layer to make.
%   Pos is the position this figure will occupy on the screen.
%   externalUpdateHandle is a function that informs LayerBox what
%   layer type has been selected. This function is called by LayerBox.
    
    if isempty(pos)
        pos = [0 0]; 
    end

    fig = figure('Name', 'New Layer Options', 'NumberTitle', 'off','Position', [pos(1) pos(2) 280 170],...
        'CloseRequestFcn',@fig_CloseRequestFcn);
    
    nameLabel = uicontrol('Parent', fig, 'Style', 'text', 'String', 'Name', 'Position', [10 150 40 20]);
    
    nameEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'String', 'New Layer', 'Position', [50 150 210 20]);
    
    choiceList = uicontrol('Parent', fig, 'Style', 'listbox', 'Position', [ 10 30 260 100]);
    
    okButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Ok', 'pos', [10 10 130 20],...
        'CallBack', @okButton_CallBack);

    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Cancel', 'pos', [140 10 130 20],...
        'CallBack', @cancelButton_CallBack);
    
    layerTypes = {'Volume','Shell','Gradient'};
    
    set(choiceList, 'String', layerTypes);
    
    layerEditors = {@VolumeLayerWindow,@ShellLayerWindow,@GradientLayerWindow};
    
    
    
    % CallBacks
    function okButton_CallBack(h, EventData)
        choiceIndex = get(choiceList, 'Value');
        name = get(nameEdit, 'String');
        externalUpdate(layerEditors{choiceIndex}, name);
        fig_CloseRequestFcn([],[]);
    end

    function cancelButton_CallBack(h, EventData)
        fig_CloseRequestFcn([],[]);
    end

    function fig_CloseRequestFcn(h, EventData)
        delete(fig);
    end

    % Update Functions
    function externalUpdate(layerEditor, layerName)
        if ~isempty(externalUpdateHandle)
            externalUpdateHandle(layerEditor, layerName);
        end
    end
    
end
