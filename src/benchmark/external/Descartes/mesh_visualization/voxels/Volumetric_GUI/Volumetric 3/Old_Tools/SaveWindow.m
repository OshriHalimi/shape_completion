%Author: James W. Ryland
%June 14, 2012

function [  ] = SaveWindow( title, pos, variableToSave)
%SAVEWINDOW allows the user to save a variable under a given file name.
%   pos is the position that the SaveWindow will occupy. variableToSave is
%   the variable that will be save in the .mat file. This function is used
%   by SaveBox which is in turn used by LayerWindow.

    if isempty(pos)
        pos = [ 1 1 ];
    end
    
    isCAVformat = 0;
    isLAYformat = 0;
    isVOLformat = 0;
    
    varDim = size(variableToSave)
    
    if ~iscell(variableToSave);
        if (length(varDim)==4)
            
            if varDim(4)==4
                isCAVformat = 1;
            end
            
        elseif length(varDim) == 3
            isVOLformat = 1
        end
        
    else
        isLAYformat = 1;
    end

    fig = figure('Name', title, 'NumberTitle', 'off', 'MenuBar', 'none', 'Resize', 'off', 'Position', [pos(1) pos(2) 300 370],...
        'CloseRequestFcn',@fig_CloseRequestFcn);
    
    figureAdjust(fig);
    
    fileNameLabel = uicontrol('Parent', fig, 'Style', 'text', 'String', 'File Name', 'Position', [10 350 60 20]);
    
    fileNameEdit = uicontrol('Parent', fig, 'Style', 'edit', 'Position', [10 330 280 20]);
    
    fileListBox = uicontrol('Parent', fig, 'Style', 'ListBox', 'Position', [10 50 280 280],...
        'CallBack', @fileListBox_CallBack);
    
    matCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'String', '.MAT', 'Position', [40 15 70 30],...
                             'Max', 1, 'Min', 0, 'Value',1, 'Visible', 'off');
    
    imgCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'String', '.IMG', 'Position', [90 15 70 30],...
                             'Max', 1, 'Min', 0, 'Value',0, 'Visible', 'off');
                         
    saveButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Save', 'Position', [150 10 70 30],...
        'CallBack', @saveButton_CallBack); 
    
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Cancel', 'Position', [220 10 70 30],...
        'CallBack', @cancelButton_CallBack); 

    if isVOLformat
        set([matCheckBox imgCheckBox], 'Visible', 'on');
    end
    
    
    %file selection and directory muck
    dirStruct = dir;
    dirName = {dirStruct.name}';
    dirIsDir = {dirStruct.isdir}';
    set(fileListBox, 'String', dirName);

    
    % Call Backs
    
    function saveButton_CallBack(h, EventData)
        fileName = get(fileNameEdit, 'String');
        
        %disp(fileName);
        %disp(variableToSave);
        
        if ~isempty(variableToSave)
            if ~isempty(fileName);
                set(saveButton, 'Enable', 'off');
                pause(.05);
                
                if get(matCheckBox, 'Value')==1 
                    save(fileName, 'variableToSave', '-v7.3');
                end
                if get(imgCheckBox, 'Value')==1
                    vol2img(variableToSave, get(fileNameEdit,'String'));
                    
                end    
                
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
        else
            if dirIsDir{selection}==0
                fileName = dirName{selection};
                l = length(fileName);
                fileType = fileName((l-3):l); %can do this easier with filePart
                
                if strcmp(fileType, '.mat')||strcmp(fileType, '.img')
                    
                    [pathstr, name, ext] = fileparts(fileName) 
                    set(fileNameEdit,'String', name);
                
                end
            end
        end
    end
    
    %Update and Close functions
    function fig_CloseRequestFcn(h, EventData)
        delete(fig);
    end
end

