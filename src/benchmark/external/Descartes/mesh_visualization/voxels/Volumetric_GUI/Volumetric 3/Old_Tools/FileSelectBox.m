%Author: James W. Ryland
%June 14, 2012

function [ getFileContentsHandle] = FileSelectBox(fig, pos, fileDimNum, externalUpdateRefHandle)
%FILESELECTBOX allows a user to select several files, and a reference file
%for editing utilities.
%   Fig is the parent figure that FileSelectBox will occupy. Pos is the
%   position that FileSelectBox will occupy in the parent figure.
%   fileDimNum tells fileSelect box how many dimension that the contents of
%   each .mat or .img file should contain. externalUpdateRefHandle gives
%   the calling function the contents and names of the reference files and
%   the additional files selected.

    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    referenceVolume = [];
    volNames = {};
    volumes = {};
    
    
    filePanel = uipanel('Parent', fig, 'Title', 'File Selection', 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 280 ]);
    
    referenceButton = uicontrol('Parent', filePanel, 'Style', 'PushButton', 'String', 'Load Reference Volume', 'Position', [10 240 260 20],...
        'CallBack', @referenceButton_CallBack);
    
    referenceFileLabel = uicontrol('Parent', filePanel, 'Style', 'Text', 'String', '...', 'Position', [10 210 260 20],...
        'CallBack', @referenceFileButton_CallBack); 
    
    addFileButton = uicontrol('Parent', filePanel, 'Style', 'PushButton', 'String', 'Load Additional Volumes', 'Position', [10 190 260 20],...
        'CallBack', @addFileButton_CallBack, 'Enable', 'off');
    
    fileListBox = uicontrol('Parent', filePanel, 'Style', 'ListBox', 'String', {}, 'Position', [10 30 260 160]);
    
    removeButton = uicontrol('Parent', filePanel, 'Style', 'PushButton', 'String', 'Remove Volume', 'Position', [10 10 260 20],...
        'CallBack', @removeButton_CallBack, 'Enable', 'off');
    
    
    getFileContentsHandle = @getFileContents;
    
    
    %CallBacks
    function referenceButton_CallBack(h, EventData)
        FileWindow('Reference File Window',[],@updateReferenceVolume);
    end

    function addFileButton_CallBack(h, EventData)
        FileWindow('Addition File Window',[],@addVolumes);
    end

    function removeButton_CallBack(h, EventData)
        index = get(fileListBox, 'Value');
        l = size(volumes, 2);
        if 0<size(volumes, 2)
            volNames(index) = [];
            volumes(index) = [];
            set(fileListBox, 'String', volNames);
        end
        if (l==index)&&(0~=size(volumes, 2))
            set(fileListBox, 'Value', l-1);
        end
        if 0==size(volumes, 2)
            set(removeButton, 'Enable', 'off');
        end
    end

    %update functions
    function updateReferenceVolume(newReferenceVolume, newReferenceFileName)
        referenceVolume = newReferenceVolume;
        set(referenceFileLabel, 'String', newReferenceFileName);
        set(addFileButton, 'Enable', 'on');
        externalUpdateRef();
    end

    function addVolumes(newVolumes, newFileName)
        sz = size(volumes, 2);
        volumes{sz+1} = newVolumes;
        volNames{sz+1} = newFileName;
        
        set(fileListBox,'String', volNames);
        set(removeButton, 'Enable', 'on');
    end

    function externalUpdateRef()
        if ~isempty(externalUpdateRefHandle)
            externalUpdateRefHandle(referenceVolume);
        end
    end

    %getters and setters
    function [vols names] = getFileContents()
        l = size(volumes, 2);
        vols = volumes;
        names = volNames;
        vols{l+1} = referenceVolume;
        names{l+1} = get(referenceFileLabel, 'String');
    end
end

