%Author: James W. Ryland
%June 14, 2012

function [  ] = CavFileWindow(title, pos, inputChangeHandle  )
%CAVFILEWINDOW allows the the user to load a CAV file
%   Pos is the position that CavFileWindow will occupy on the screen.
%   InputChangeHandle updates external functions with the contents of the
%   CAV file.
    
    fileContents = [];

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
                inputChangeHandle(fileContents);
                fig_CloseRequestFcn([],[]);
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
            
        elseif dirIsDir{selection}==0
            
            fileName = dirName{selection};
            l = length(fileName);
            fileType = fileName((l-3):l); %can do this easier with filePart
            tempVol = [];
            if strcmp(fileType, '.mat')
                tempStruct = load(fileName);
                tempCell = struct2cell(tempStruct);
                tempVol = tempCell{1};
                clear('tempStruct');
                clear('tempCell');
            end
            fileContents = tempVol;
            if ~isempty(fileContents)
                if isnumeric(fileContents)&&(4==size(size(fileContents),2));
                    set(openButton, 'Enable', 'on');
                end
            end
            
        end
    end
    
    %Update and Close functions

    function fig_CloseRequestFcn(h, EventData)
        delete(fig);
    end

end

