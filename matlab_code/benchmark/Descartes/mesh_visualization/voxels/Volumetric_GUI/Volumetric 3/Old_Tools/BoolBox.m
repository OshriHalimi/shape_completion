%Author: James W. Ryland
%June 14, 2012

function [ updateVolume1Handle updateVolume2Handle checkDimensionsHandle ] = BoolBox( fig, pos, externalUpdateHandle )
%BOOLBOX bool box allows a user to select a boolean opperand to be applied
%to a volume. (AND OR XOR)
%   fig is the figure that boolBox will occupy. Pos is the position in the
%   parent window or panel that BoolBox will occupy. ExternalUpdateHandle
%   updates external functions with the output of the boolean operation.
    

    selectionVolume = [];
    volume1 = uint8(rand(100,100,100)>.5);
    volume1Full = uint8(rand(100,100,100)>.5);
    volume2 = uint8(rand(100,100,100)>.5);
    booleanOp = 1;
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    boolPanel = uipanel('Parent', fig, 'Title', 'Boolean Operations', 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 150 ] );
    
    boolGroup = uibuttongroup('Parent', boolPanel, 'Units', 'Pixels', 'Position', [0 0 140 140],...
        'SelectionChangeFcn', @boolGroup_SelectionChangeFcn);
    
    unionRadio = uicontrol('Parent', boolGroup, 'Style', 'radio', 'String', 'Union', 'Position', [10 100 100 20]);
    
    intersectRadio = uicontrol('Parent', boolGroup, 'Style', 'radio', 'String', 'Intersect', 'Position', [10 70 100 20]);
    
    xorRadio = uicontrol('Parent', boolGroup, 'Style', 'radio', 'String', 'XOR', 'Position', [10 40 100 20]);
    
    maskRadio = uicontrol('Parent', boolGroup, 'Style', 'radio', 'String', 'Mask 1 with 2', 'Position', [10 10 100 20]);
    
    
    [volumeUpdateHandle updateImageHandle] = ViewBox(boolPanel, [140 0], []);
    
    
    % initialize settings
    updateVolume1Handle = @updateVolume1;
    updateVolume2Handle = @updateVolume2;
    checkDimensionsHandle = @checkDimensions;
    updateSelection();
    
    % CallBacks
    
    function boolGroup_SelectionChangeFcn(h, EventData)
        switch EventData.NewValue
            case unionRadio
                booleanOp = 1;
            case intersectRadio
                booleanOp = 2;
            case xorRadio
                booleanOp = 3;
            case maskRadio
                booleanOp = 4;
        end
        
        updateSelection();
        externalUpdate();
    end

    % update functions
    function updateSelection()
        if (3==length(size(volume1)))&&(3==length(size(volume2)))
            if 3==sum(size(volume1)==size(volume2))
                switch booleanOp
                    case 1
                        selectionVolume = uint8(volume1|volume2);
                    case 2
                        selectionVolume = uint8(volume1&volume2);
                    case 3
                        selectionVolume = uint8((volume1&(~volume2))|(~volume1&volume2));
                    case 4
                        selectionVolume = double(volume1Full).*double(volume2);
                end
                volumeUpdateHandle(uint8(selectionVolume)*255);
                externalUpdate();
            end
        end
    end
        
    function updateVolume1(newVolume1, newFullVolume)
        volume1 = uint8(newVolume1);
        volume1Full = newFullVolume;
        updateSelection();
    end

    function updateVolume2(newVolume2, newFullVolume)
        volume2 = uint8(newVolume2);
        updateSelection();
    end
    
    function externalUpdate()
        if ~isempty(selectionVolume)
            if ~isempty(externalUpdateHandle)
                externalUpdateHandle(selectionVolume);
            end
        end
    end

    %getters and setters
    function [dimMatch] = checkDimensions()
        dimMatch = 0;
        if (3==length(size(volume1)))&&(3==length(size(volume2)))
            if 3 == sum(size(volume1)==size(volume2))
                dimMatch = 1;
            end
        end
    end
    
end
