%Author: James W. Ryland
%June 14, 2012

function [  ] = FileWindow(title, pos, inputChangeHandle)
%FILEWINDOW allows a user to select a file to for the calling
%function.
%   Pos is the position FileWindow will occupy on the screen.
%   InputChangeHandle updates the calling function with the selected files
%   contents.

    fileContents = [];
    fileName = [];

    if isempty(pos)
        pos = [ 1 1 ];
    end

    fig = figure('Name',title, 'NumberTitle', 'off',  'MenuBar', 'none', 'Position', [pos(1) pos(2) 300 340],...
        'CloseRequestFcn',@fig_CloseRequestFcn);
    
    figureAdjust(fig);
    
    fileListBox = uicontrol('Parent', fig, 'Style', 'ListBox', 'Position', [10 50 280 280],...
        'CallBack', @fileListBox_CallBack);
    
    openButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Open', 'Position', [150 10 70 30],...
        'CallBack', @openButton_CallBack, 'Enable', 'off'); 
    
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Cancel', 'Position', [220 10 70 30],...
        'CallBack', @cancelButton_CallBack); 

    %file selection and directory muck
    dirStruct = dir;
    dirName = {dirStruct.name}';
    dirIsDir = {dirStruct.isdir}';
    set(fileListBox, 'String', dirName);
 
    % Call Backs
    
    function openButton_CallBack(h, EventData)
        if ~isempty(fileContents)
            if ~isempty(inputChangeHandle)
                
                if(~isvector(fileContents))
                    inputChangeHandle(fileContents,fileName);
                    fig_CloseRequestFcn([],[]);
                else
                    fig_CloseRequestHandle = @fig_CloseRequestFcn;
                    VectorLoadWindow(fileContents,fileName,[],inputChangeHandle,fig_CloseRequestHandle);
                end
                
            end
        end
    end
    
    function cancelButton_CallBack(h, EventData)
        fig_CloseRequestFcn([],[]);
    end
    
    
    function fileListBox_CallBack(h, EventData)
                
        selection = get(fileListBox, 'Value');
        
        
        if strcmp(get(fig,'SelectionType'),'open')
           
            
            if dirIsDir{selection}==1
                cd(dirName{selection});
                dirStruct = dir;
                dirName = {dirStruct.name}';
                dirIsDir = {dirStruct.isdir}';
           
                set(fileListBox, 'Value', 1, 'String', dirName);
                
            end
            
        else
            if dirIsDir{selection}==0
                fileName = dirName{selection};
                l = length(fileName);
                fileType = fileName((l-3):l); %can do this easier with filePart
                tempVol = [];
                if strcmp(fileType, '.img')
                    tempVol = analyze75read(fileName);
            
                elseif strcmp(fileType, '.mat')
                    tempStruct = load(fileName);
                    tempCell = struct2cell(tempStruct);
                    tempVol = tempCell{1};
                    clear('tempStruct');
                    clear('tempCell');
                
                elseif strcmp(fileType, '.cav')
                    tempStruct = load(fileName, '-mat');
                    tempCell = struct2cell(tempStruct);
                    tempVol = tempCell{1};
                    clear('tempStruct');
                    clear('tempCell');
                
                elseif strcmp(fileType, '.lay')
                    tempStruct = load(fileName, '-mat');
                    tempCell = struct2cell(tempStruct);
                    tempVol = tempCell{1};
                    clear('tempStruct');
                    clear('tempCell');
                
                end
                    
                fileContents = tempVol;
                clear('tempVol');
                if ~isempty(fileContents)
                    set(openButton, 'Enable', 'on');
                end
                
            end
        end
    end
    
    %Update and Close functions

    function fig_CloseRequestFcn(h, EventData)
        clear('fileContents');
        delete(fig);
    end


end
