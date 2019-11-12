%Author: James W. Ryland
%June 14, 2012

function [ updateReferenceHandle getTargVoxNumHandle getVoxDimHandle getPrefixHandle] = ResizeBox( fig, pos, externalApplyHandle)
%RESIZEBOX allows the user to input an approximate target number of voxels
%to interpolate a volume to and to get a projection of increase accross
%each dimension and the actual number of voxels that will be generated.
%This function is used by ResizeWindow.
%   FIG is the figure or panel that all of the graphic components of
%   ResizeBox will reside in. POS is the position that ResizeBox will
%   occupty. externalApplyHandle updates the parent function with the
%   target voxel number so that it can be applied to a list of volumes.

    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    refVolume = [];
    
    targVoxNum = [];
    
    vDimX = 1;
    vDimY = 1;
    vDimZ = 1;
    
    resizePanel = uipanel('Parent', fig, 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 160 ]);
    
    prefixLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Prefix', 'Position', [90 130 50 20],...
        'HorizontalAlignment', 'right');
    
    prefixEdit = uicontrol('Parent', resizePanel, 'Style', 'edit', 'String', 'INT_', 'Position', [150 133 70 21]);
    
    currentDimLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Current Dimensions', 'Position', [10 105 130 20],...
        'HorizontalAlignment', 'right');
    
    curDimDisp = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', '[0 0 0]', 'Position', [150 105 90 20],...
        'HorizontalAlignment', 'left');
    
    %Voxel Ratio stuff
    
    voxDimLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Voxel Dimensions', 'Position', [10 85 130 20],...
        'HorizontalAlignment', 'right');
    
    voxDimXLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'X', 'Position', [150 85 20 20],...
        'HorizontalAlignment', 'left');
    
    voxDimYLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Y', 'Position', [190 85 20 20],...
        'HorizontalAlignment', 'left');
    
    voxDimZLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Z', 'Position', [230 85 20 20],...
        'HorizontalAlignment', 'left');
    
    offset = 10;
    
    voxDimEditX = uicontrol('Parent', resizePanel, 'Style', 'edit', 'String', '1', 'Position', [150+offset 87 30 22],...
        'HorizontalAlignment', 'left', 'CallBack', @voxDim_CallBack);
    
    voxDimEditY = uicontrol('Parent', resizePanel, 'Style', 'edit', 'String', '1', 'Position', [190+offset 87 30 22],...
        'HorizontalAlignment', 'left', 'CallBack', @voxDim_CallBack);
    
    voxDimEditZ = uicontrol('Parent', resizePanel, 'Style', 'edit', 'String', '1', 'Position', [230+offset 87 30 22],...
        'HorizontalAlignment', 'left', 'CallBack', @voxDim_CallBack);
    
    
    
    targetVoxLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Target Voxel Number', 'Position', [10 65 130 20],...
        'HorizontalAlignment', 'right');
    
    targVoxEdit = uicontrol('Parent', resizePanel, 'Style', 'edit', 'String', '16,000,000', 'Position', [150 65 90 23],...
        'HorizontalAlignment', 'left', 'CallBack', @targVoxEdit_CallBack);
    
    projectVoxLabel = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', 'Projected Voxel Number', 'Position', [10 40 130 20],...
        'HorizontalAlignment', 'right');
    
    proDimDisp = uicontrol('Parent', resizePanel, 'Style', 'text', 'String', '[0 0 0]', 'Position', [150 40 90 20],...
        'HorizontalAlignment', 'left');
    
    applyButton = uicontrol('Parent', resizePanel, 'Style', 'pushbutton', 'String', 'Apply Interpolation', 'Position', [70 10 140 20],...
        'CallBack', @applyButton_CallBack, 'Enable', 'off');
    
    
    %functions to Pass
    updateReferenceHandle = @updateReference;
    getTargVoxNumHandle = @getTargVoxNum;
    getPrefixHandle = @getPrefix;
    getVoxDimHandle = @getVoxDim;
    
    %CallBacks
    function targVoxEdit_CallBack(h, EventData)
        [temp] = str2double(get(targVoxEdit,'String'));
        targVoxNum = [];
        if ~isnan(temp)
            targVoxNum  = temp;
        end
        
        
        if (~isempty(refVolume))&&(~isempty(targVoxNum))
            
            [xs ys zs] = size(refVolume);
            Vold = (xs-1)*(ys-1)*(zs-1);
            
            [voxNum c newDim] = targetedInterp3NonCubeEst( refVolume , targVoxNum, [vDimX vDimY vDimZ]);
            
            set(proDimDisp, 'String', ['[' mat2str(newDim(1)) ' ' mat2str(newDim(2)) ' ' mat2str(newDim(3)) ']']);
            
            set(applyButton, 'Enable', 'on');
            
        else
            set(proDimDisp, 'String', '[0 0 0]');
            
            set(applyButton, 'Enable', 'off');
        end    
    end

    function voxDim_CallBack(h, EventData)
        [temp] = str2double(get(h,'String'))
        
        if ~isnan(temp)&&(temp>0)
            
            set(h, 'String',temp);
            
            vDimX = str2double(get(voxDimEditX,'String'));
            vDimY = str2double(get(voxDimEditY,'String'));
            vDimZ = str2double(get(voxDimEditZ,'String'));
            
            
            if (~isempty(refVolume))&&(~isempty(targVoxNum))
                [voxNum c newDim] = targetedInterp3NonCubeEst( refVolume , targVoxNum, [vDimX vDimY vDimZ]);            
                set(proDimDisp, 'String', ['[' mat2str(newDim(1)) ' ' mat2str(newDim(2)) ' ' mat2str(newDim(3)) ']']);
                
            end
        else
            set(h, 'String','1');
            
        end
            
        
    
    end

    function applyButton_CallBack(h, EventData)
        if ~isempty(externalApplyHandle)
            set(applyButton, 'Enable', 'off');
            pause(0.05);
            externalApplyHandle();
            set(applyButton, 'Enable', 'on');
        end
    end

    % update functions
    function updateReference(newReference)
        refVolume = newReference;
        sz = size(refVolume);
        set(curDimDisp,'String', mat2str(sz));
        targVoxEdit_CallBack([],[]);
    end

    function [TVN] = getTargVoxNum()
        TVN = targVoxNum;
    end

    % getters and setters
    function [prfx] = getPrefix()
        prfx = get(prefixEdit, 'String');
        if isstr(prfx)
            prfxcellstr = cellstr(prfx);
            prfx = prfxcellstr{1};
        end
    end

    function [voxDim] = getVoxDim()
        voxDim = [vDimX vDimY vDimZ];
    end

end

