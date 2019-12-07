function [ ] = VectorLoadWindow(vectorToShape, fileName, pos, parentInputChangeHandle, parentCloseRequestHandle)
%VectorLoadWindow
%   When file window detects that the user is loading a vector from any
%   source file, this function will be called so that the user is asked to
%   choose appropriate reshape and reading parameters to turn the vector
%   into a volumetric matrix usable with volumetric.

    if isempty(pos)
        pos = [ 1 1 ];
    end
    
    
    %Internal Varaibles
    shapedVector = [];
    xDim = 0;
    yDim = 0;
    zDim = 0;
    elemNum = length(vectorToShape);
    rowWiseDefault = 0;
    
    %check for presets
    
    %fmri 64*64*38 preset Alice O'toole's Lab
    if elemNum == (64*64*38)
        xDim = 64;
        yDim = 64;
        zDim = 38;
        rowWiseDefault = 1;
    
    %mri structural preset Alice O'toole's Lab
    elseif elemNum == (256*256*160) 
        xDim = 256;
        yDim = 256;
        zDim = 160;
        rowWiseDefault = 1;
    end
    
    %Instruction String
    instructString = ['Volumetric has detected that the file you are trying to load ',...
                'is a vector. This window lets you reshape the vector into the ',...
                'volumetric matrix. Depending on how the vecor was constructed ',...
                'you may need to choose to fill row wise in order to get the ',...
                'original 3D matrix back in correct orientation. Several common settings ',...
                'will automatically load depending on the size of the vector. The number of ',...
                'Voxels in the 3D matrix needs to match the number of Elements in the vector.'];
    
    
    %GUI instantiation
    fig = figure('Name','Vector Load Window', 'NumberTitle', 'off',  'MenuBar', 'none', 'Position', [pos(1) pos(2) 300 340],...
        'CloseRequestFcn',@fig_CloseRequestFcn);
    
    instructionLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', instructString, 'Position', [10 230 280 100],...
                        'HorizontalAlignment', 'left');
    
    offset = 80;
    
    xEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'String', mat2str(xDim), 'Position', [(10+offset*1) 180 30 20 ],...
                        'CallBack', @xEdit_CallBack);
                    
    yEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'String', mat2str(yDim), 'Position', [(10+offset*2) 180 30 20 ],...
                        'CallBack', @yEdit_CallBack);
                    
    zEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'String', mat2str(zDim), 'Position', [(10+offset*3) 180 30 20 ],...
                        'CallBack', @zEdit_CallBack);
                    
                    
    xLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'X', 'Position', [(60+offset*0) 180 30 20]);
    
    yLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'Y', 'Position', [(60+offset*1) 180 30 20]);
    
    zLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'Z', 'Position', [(60+offset*2) 180 30 20]);
    
    
    elementStr = '# elements: ';
    
    voxelStr =   '# voxels:   ';
    
    vLengthLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', elementStr, 'Position', [60 150 200 20],...
                               'HorizontalAlignment', 'Left'); 
    
    voxNumLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', voxelStr, 'Position', [60 120 200 20],...
                               'HorizontalAlignment', 'Left'); 
                     
    rowWiseLabel = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'Fill Matrix Row Wise', 'Position', [60, 90, 100, 20],...
                              'HorizontalAlignment', 'Left');
                           
    rowWiseCheckBox = uicontrol('Parent', fig, 'Style', 'CheckBox', 'Position', [(60+100) 90 30 30],...
                              'Max', 1, 'Min', 0, 'Value', rowWiseDefault); 
                           
    readButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Read', 'Position', [150 10 70 30],...
        'CallBack', @openButton_CallBack, 'Enable', 'off');                 
    
    
    cancelButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Cancel', 'Position', [220 10 70 30],...
        'CallBack', @cancelButton_CallBack); 
    
    checkIfDimensionsAgree();
    
    
    %Call Backs
    
    function openButton_CallBack(h, EventData)
        if ~isempty(parentInputChangeHandle)
            
            volumetricMatrix = filledMatrix();
            parentInputChangeHandle(volumetricMatrix,fileName);
            SaveWindow('Save Window', [], volumetricMatrix);
            clear('volumetricMatrix');
            
            if ~isempty(parentCloseRequestHandle)
                parentCloseRequestHandle([],[]);
            end
            
            fig_CloseRequestFcn();
            
            %lastly ask to save as a .mat file or a .img file
            %coule make easier by asking for the file name too...
            
        end
    end
    
    function xEdit_CallBack(h, EventData)
        
        xtemp = str2double(get(xEdit, 'String'));
        if((~isnan(xtemp))&&(mod(xtemp,1)==0))
            xDim = xtemp;
        end
        set(xEdit, 'String', mat2str(xDim));
        checkIfDimensionsAgree();
    end

    function yEdit_CallBack(h, EventData)
        
        ytemp = str2double(get(yEdit, 'String'));
        if((~isnan(ytemp))&&(mod(ytemp,1)==0))
            yDim = ytemp;
        end
        set(yEdit, 'String', mat2str(yDim));
        checkIfDimensionsAgree();
    end

    function zEdit_CallBack(h, EventData)
        
        ztemp = str2double(get(zEdit, 'String'));
        if((~isnan(ztemp))&&(mod(ztemp,1)==0))
            zDim = ztemp;
        end
        set(zEdit, 'String', mat2str(zDim));
        checkIfDimensionsAgree();
    end

    function checkIfDimensionsAgree()
        
        elemNum = length(vectorToShape);
        voxNum =(xDim*yDim*zDim);
        
        if(elemNum==voxNum)
            set(readButton, 'Enable', 'On');
        else
            set(readButton, 'Enable', 'Off');
        end
        
        set(vLengthLabel, 'String', [elementStr mat2str(elemNum)]);
        set(voxNumLabel,'String',   [voxelStr mat2str(voxNum)]);
        
    end

    function [matrix3] = filledMatrix()
        
        matrix3 = 0;
        
        if get(rowWiseCheckBox,'Value')==0
            
            matrix3 = reshape(vectorToShape, xDim, yDim, zDim);
            
        elseif get(rowWiseCheckBox,'Value')==1
            
            %reverse xDim and yDim
            %reshape will fill col wise
            matrix3 = reshape(vectorToShape, yDim, xDim, zDim);
            
            %switches the axes x and y
            %leaving a matrix that was filled row wise
            matrix3 = permute(matrix3, [2 1 3]);
            
        end
    end

    function cancelButton_CallBack(h, EventData)
        fig_CloseRequestFcn([],[]);
    end

    function fig_CloseRequestFcn(h, EventData)
        clear('vectorToShape');
        delete(fig);
    end
    


end
