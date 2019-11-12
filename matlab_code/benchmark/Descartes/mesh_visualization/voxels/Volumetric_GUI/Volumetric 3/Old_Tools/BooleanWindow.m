%Author: James W. Ryland
%June 14, 2012

function [  ] = BooleanWindow(pos )
%BOOLEANWINDOW allows a user to combine volume selections using boolean
%methods like AND OR XOR
%   Pos is the position that boolean window will occupy on the desktop.
    
    volume = [];
    volSize1 = [0 0 0];
    volSize2 = [0 0 0];
    
    
    if isempty(pos)
        pos = [0 0];
    end

    fig = figure('Name', 'Boolean Volume Editor', 'Resize', 'off', 'NumberTitle', 'off','MenuBar', 'none','Position', [pos(1) pos(2) 280 480]);
    
    figureAdjust(fig);
    
    saveButton = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Save', 'Position', [10 10 260 30],...
        'CallBack', @saveButton_CallBack, 'Enable', 'off'); 
    
    checkDimensions = [];
    
    [updateVol1 updateVol2 checkDimensions] = BoolBox(fig, [0 50], @updateVolume);
    
    SourceBox(fig, 'Source 2', [0 200], updateVol2, [] );
    
    SourceBox(fig, 'Source 1', [0 340], updateVol1, [] );
    
    
    
    %CallBacks
    function saveButton_CallBack(h, EventData)
        if ~isempty(volume)
            SaveWindow('Save Boolean Product', [], volume);
        end
    end
    
    %Update Functions
    function updateVolume(newVolume)
        volume = newVolume;
        if ~isempty(checkDimensions)
            if ~isempty(volume)&&checkDimensions()
                set(saveButton, 'Enable', 'on');
            else
                set(saveButton, 'Enable', 'off');
            end
        end
    end
    
end
