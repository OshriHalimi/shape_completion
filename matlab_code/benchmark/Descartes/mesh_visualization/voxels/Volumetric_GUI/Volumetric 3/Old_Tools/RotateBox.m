%Author: James W. Ryland
%June 14, 2012

function [getRotationHandle getPrefixHandle] = RotateBox(fig, pos, externalUpdateRotateHandle, externalUpdateApplyHandle, externalUpdateResetHandle)
%ROTATEBOX allows a user to create a series of rotations and to apply them
%to a reference volume and to apply them to set of additional files.
%   fig is the parent figure that all of the graphical components of ROTATE
%   BOX will reside in. POS is the position that RotateBox will occupy in
%   the parent fig. ExternalUpdateRotateHandle is a function that update
%   the rendering to show the results of the rotations.
%   ExternalUpdateApplyHandle is a function that applies the selected
%   rotation to a set of additional files. ExternalUpdateResetHandle resets
%   the refernce volume rendering to its initial unrotated state and clears
%   the rotation list.
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    planeDim1s = [];
    planeDim2s = [];
    rotDirs = [];
    rotString = [];
    
    rotatePanel = uipanel('Parent', fig, 'Title', 'File Selection', 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 140 ]);
    
    rotatePos = zeros(3,1);
    rotateNeg = zeros(3,1);
    axLabel = zeros(3,1);
    axString = ['X', 'Y', 'Z'];
    
    prefixLabel = uicontrol('Parent', rotatePanel, 'Style', 'text', 'String', 'Prefix', 'Position', [10 100 50 20]);
    
    prefixEdit = uicontrol('Parent', rotatePanel, 'Style', 'edit', 'String', 'R', 'Position', [60 100 70 20]);
    
    rotationEdit = uicontrol('Parent', rotatePanel, 'Style', 'edit', 'String', '', 'Position', [10 70 120 30], 'Enable', 'inactive');
    
    applyButton = uicontrol('Parent', rotatePanel, 'Style', 'PushButton', 'String', 'Apply', 'Position', [10 40 120 30],...
        'CallBack', @applyButton_CallBack);
    
    resetButton = uicontrol('Parent', rotatePanel, 'Style', 'PushButton', 'String', 'Reset', 'Position', [10 10 120 30],...
        'CallBack', @resetButton_CallBack);
    
    %Create Selection Controls
    for i = 1:3
        bgColor = zeros(1,3);
        bgColor(i) = 1;
        
        axLabel(i) = uicontrol(rotatePanel, 'Style', 'text', 'String', axString(i),...
            'ForegroundColor', bgColor, 'Position', [160 (105-i*30) 20 20]); 
        
        rotatePos(i) = uicontrol(rotatePanel,'Style', 'PushButton', 'String', '+',...
                        'Position', [ 180 (105-i*30) 20 20], 'CallBack', @rotateButtons);
        
        rotateNeg(i) = uicontrol(rotatePanel,'Style', 'PushButton', 'String', '-',...
                        'Position', [ 200 (105-i*30) 20 20], 'CallBack', @rotateButtons);
        
    end
    
    %Cross Platform 
    
    
    %function handles to be passed
    getRotationHandle = @getRotation;
    getPrefixHandle = @getPrefix;
    
    function rotateButtons(h, EventData)
        
        posRotOn = rotatePos == h;
        negRotOn = rotateNeg == h;
        posi = sum(posRotOn);
        negi = sum(negRotOn);
        
        ax = [1 2 3];
        axor = find((posRotOn+negRotOn)>0);
        ax(axor) = [];
        
        planeDim1 = ax(1);
        planeDim2 = ax(2);
        rotDir = posi - negi;
        
        %disp([planeDim1 planeDim2]);
        %disp(rotDir);
        
        ci = size(rotDirs,2);
        
        planeDim1s(ci+1) = planeDim1;
        planeDim2s(ci+1) = planeDim2;
        rotDirs(ci+1) = rotDir;
        
        dirString = '';
        if negi
            dirString = '-';
        end
        axRotString = axString(axor);
        rotString = [rotString dirString axRotString ','];
        set(rotationEdit, 'String', rotString);
        externalUpdateRotate();
    
    end

    function applyButton_CallBack(h, EventData)
        if ~isempty(externalUpdateApplyHandle)
            externalUpdateApplyHandle();
        end
    end

    function resetButton_CallBack(h, EventData)
        planeDim1s = [];
        planeDim2s = [];
        rotDirs = [];
        rotString = [];
        set(rotationEdit, 'String', rotString);
        externalUpdateReset();
    end

    function externalUpdateRotate()
        if ~isempty(externalUpdateRotateHandle)
            externalUpdateRotateHandle();
        end
    end

    function externalUpdateApply()
        if ~isempty(externalUpdateApplyHandle)
            set(applyButton, 'Enable', 'off');
            pause(0.05);
            externalUpdateApplyHandle();
            set(applyButton, 'Enable', 'on');
        end
    end

    function externalUpdateReset()
        if ~isempty(externalUpdateResetHandle)
            externalUpdateResetHandle();
        end
    end

    %Getters and Setters
    function [p1s p2s rDirs] = getRotation()
        p1s = planeDim1s;
        p2s = planeDim2s;
        rDirs = rotDirs;
    end

    function [prfx] = getPrefix()
        prfx = get(prefixEdit, 'String');
        if isstr(prfx)
            prfxcellstr = cellstr(prfx);
            prfx = prfxcellstr{1};
        end
    end
end

