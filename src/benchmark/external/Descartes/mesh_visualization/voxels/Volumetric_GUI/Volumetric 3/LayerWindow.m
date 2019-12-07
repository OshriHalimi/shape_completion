function [  ] = LayerWindow( DataStruct)

    

    % INITIALIZATION

    if isempty(DataStruct)
       [DataStruct] = AddLayer(DataStruct, [], [1 0]);
       
    end
    [displayList, indexList] = DisplayList(DataStruct);
   
    DataStruct.Master.Alpha = []; % empty is shorthand for do not render
    DataStruct.Master.Color = [];
    DataStruct.Master.VolIndexAlpha = [];
    DataStruct.Master.VolIndexColor = [];
    
    pos = [];
    
    % use screen size to set these
    height = 500;
    width = 400;
    
    if isempty(pos)
        
        scr = get(0,'ScreenSize');
        
        pos = [ 1 (scr(4)-height)];
        
    end
    
    title = 'Layers';
    
    fig = figure('Name',title, 'MenuBar', 'None', 'Resize', 'off', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height], 'CloseRequestFcn', @MyCloseRequestFcn, 'DeleteFcn', @MyDeleteFcn);
    
    
    optMenu = uicontextmenu; 
    item1 = uimenu(optMenu, 'Label', 'Add New Layer', 'Callback', @menu_Callback);
    %item2 = uimenu(optMenu, 'Label', 'Add New to Group', 'Callback', @menu_Callback);
    item3 = uimenu(optMenu, 'Label', 'Move', 'Callback', @menu_Callback);
    item4 = uimenu(optMenu, 'Label', 'Move to Group', 'Callback', @menu_Callback);
    item5 = uimenu(optMenu, 'Label', 'Delete', 'Callback', @menu_Callback);
    item6 = uimenu(optMenu, 'Label', 'Cancel', 'Callback', @menu_Callback);
        
    
    
    sX = .9;
    sY = .9;
    spX = (1-sX)/2;
    spY = (1-sY)/2;
    
    layerList = uicontrol('Parent', fig, 'Style', 'listbox', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY], 'Callback', @List_Callback);
    set(layerList, 'String', displayList, 'FontSize', 20);
    set(layerList, 'UIContextMenu', optMenu);
    
    
    sX = .2;
    sY = .05;
    spX = .75;
    spY = 1-sY;
    labelAction = uicontrol('Parent', fig, 'Style', 'Text', 'String', 'Action', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15);
    
    sX = .2;
    sY = .05;
    spX = .05;
    spY = 1-sY;
    loadButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Load', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @loadButton_Callback);
    
    sX = .2;
    sY = .05;
    spX = .25;
    spY = 1-sY;
    saveButton = uicontrol('Parent', fig, 'Style', 'PushButton', 'String', 'Save', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @saveButton_Callback);
    
    sX = .3;
    sY = .05;
    spX = .45;
    spY = 1-sY;
    toolsMenu = uicontrol('Parent', fig, 'Style', 'popupmenu', 'String', 'Tools', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @toolsMenu_Callback);
    set(toolsMenu, 'String', {'Tools', 'Rotate', 'Scale', 'Boolean', 'Make XYZ Volumes', 'Ray Cast Render', 'Add Mask Layer'});
    
    
    % INITIALIZATION
    UpdateLayerHandle = @UpdateLayer;
    UpdateScaledVolsHandle = @UpdateScaledVols;
    GetCropOffsetHandle = @GetCropOffset;
    AddGeneratedLayerHandle = @AddGeneratedLayer;
    GetActualSizeHandle = @GetActualSize;
    
    [UpdateEditLayerHandle, editFig] = EditWindow(pos, UpdateLayerHandle, fig);
    
    [UpdateVolsHandle, UpdateColorHandle, UpdateAlphaHandle, GetCamProperties, viewFig, viewAxis] = ViewWindow(UpdateScaledVolsHandle, fig);
    
    [pointerFig] = PointerWindow(GetCropOffsetHandle, viewAxis);
    
    
    fileTypes ={'*.vl3', 'Volumetric 3 Files( *.vl3 )'};
    mainVal = 1;
    moveTo = 1; 
    state = 'Select';
    index = indexList(mainVal,:);
    toInd = [];
    Select();
    
    
    % CALLBACKS
    function List_Callback(hObject, eventData, handles)
        % put case breaks in here
        moveTo = get(layerList, 'Value');
        toInd = indexList(moveTo,:);
        
        if strcmp(state, 'Move')
            Move();
        end
        
        if strcmp(state, 'Move to Group')
            MoveToBlend();
        end
        
        mainVal = moveTo;
        index = indexList(mainVal,:);
        
        if strcmp(state, 'Select')
            Select();
        end
        
        state = 'Select';
        set(labelAction, 'String', state);
    end

    function menu_Callback(hObject, eventData, handles)
        state = get(hObject, 'Label');
        mainVal = get(layerList, 'Value');
        index = indexList(mainVal,:);
        
        set(labelAction, 'String', state);
        
        if strcmp(state,'Add New Layer')
            AddNew();
            state = 'Select';
            set(labelAction, 'String', state);
        end
        
        if strcmp(state,'Delete')
            Delete();
            state = 'Select';
            set(labelAction, 'String', state);
        end
        
       mainVal = get(layerList, 'Value');
       index = indexList(mainVal,:); 
       Select();
        
    end

    function loadButton_Callback(hObject, eventData, handles)
        
        
        [filename, pathname] = uigetfile(fileTypes);
        
        if ~(isnumeric(filename) && filename==0)
            
            tempStruct = load([pathname filename], '-mat');
            
            DataStruct = tempStruct.DataStruct;
            
            clear tempStruct;
            
            index = [1 1];
            set(layerList, 'Value', 1);
            Select();
            UpdateDisplayList();
            UpdateScaledVols();
        end
        
        
        
    end

    function saveButton_Callback(hObject, eventData, handles)
        
        [filename, pathname] = uiputfile(fileTypes);
        
        if ~(isnumeric(filename) && filename==0)
            
            save([pathname filename], 'DataStruct');
            
        end
        
    end

    function toolsMenu_Callback(hObject, eventData, handles)
        
        toolNum = get(toolsMenu, 'value');
        
        scr = get(0,'ScreenSize');
        
        if toolNum ==2
            RotateWindow([scr(3)/2,scr(4)/2]);
        elseif toolNum == 3
            ResizeWindow([scr(3)/2,scr(4)/2]);
        elseif toolNum == 4
            BooleanWindow([scr(3)/2,scr(4)/2]);
        elseif toolNum == 5
            sz = size(DataStruct.Master.VolIndexAlpha);
            MakeXYZ(sz);
        elseif toolNum == 6
            % make a new raycast window
            % of the current scene...
            RayCastDialogueWindow([], DataStruct, GetCamProperties)
        elseif toolNum == 7
            % add a new masking layer
            AddMaskDialogueWindow(viewAxis,GetCropOffset(), GetActualSizeHandle,AddGeneratedLayerHandle)
        end
        
        
        set(toolsMenu, 'value', 1);
        
    end

    function [] = MyCloseRequestFcn(hObject, eventData)
        
        delete(fig);
    end

    function [] = MyDeleteFcn(hObject, eventData);

        disp('Thanks for using Volumetric 3');
        delete(editFig);
        delete(viewFig);
        delete(pointerFig);
        clear DataStruct;
    end


    
    % HELPER FUNCTIONS
    function [] = Select()
        
        if index(2)~=0
             [layerStruct] = GetLayer(DataStruct, index);
             UpdateEditLayerHandle(layerStruct);
        end
        
    end


    function [] = AddNew()
        
        index(2) = 0;
        
        [DataStruct] = AddLayer(DataStruct, [], index);
        [displayList, indexList ] = DisplayList(DataStruct);
        
        get(layerList, 'Value')
        
        
        if index(1) == 0
            set(layerList, 'Value', 1);
        end
        set(layerList, 'String', displayList);
        
    end
    

    function [] = Delete()
        
        [DataStruct] = DeleteLayer(DataStruct, index);
        [displayList, indexList ] = DisplayList(DataStruct);
        
        set(layerList, 'Value', 1);
        set(layerList, 'String', displayList);
        UpdateScaledVols()
    end
    

    function [] = Move()
        
        toInd(2) = 0;
        
        [DataStruct] = MoveLayer(DataStruct, index, toInd); 
        [displayList, indexList ] = DisplayList(DataStruct);
   
        set(layerList, 'String', displayList);
        UpdateScaledVols()
    end


    function [] = MoveToBlend()
        
        if toInd(2) == 0
            toInd(2) = 1;
        end

        [DataStruct] = MoveLayer(DataStruct, index, toInd); 
        [displayList, indexList ] = DisplayList(DataStruct);

        set(layerList, 'String', displayList);
        UpdateScaledVols()
    end

    function [] = UpdateDisplayList()
     
        [displayList, indexList ] = DisplayList(DataStruct);
        set(layerList, 'String', displayList);
    
    end


    function [] = UpdateScaledVols() % these are updated at the same time
        
        [passCheck, error] = CheckDataStruct(DataStruct);
        
        if passCheck
            DataStruct.Master.Alpha = MakeMap(DataStruct.Layer.AlphaMap); % empty is shorthand for do not render
            DataStruct.Master.Color = MakeMap(DataStruct.Layer.ColorMap);
            DataStruct.Master.VolIndexAlpha = MakeIndex(DataStruct.Layer.ScaledVolAlpha);
            DataStruct.Master.VolIndexColor = MakeIndex(DataStruct.Layer.ScaledVolColor);
            
        else
            DataStruct.Master.Alpha = []; % empty is shorthand for do not render
            DataStruct.Master.Color = [];
            DataStruct.Master.VolIndexAlpha = [];
            DataStruct.Master.VolIndexColor = [];
        end
        
        UpdateVolsHandle(DataStruct.Master.VolIndexColor, DataStruct.Master.VolIndexAlpha);
        UpdateAlphaHandle(DataStruct.Master.Alpha);
        UpdateColorHandle(DataStruct.Master.Color);
        
    end

    function [] = UpdateLayerMaps(part) % these can be manipulated indipendently
        
        [passCheck, error] = CheckDataStruct(DataStruct);
        
        if passCheck
            
            if strcmp(part, 'AlphaMap')
                DataStruct.Master.Alpha = MakeMap(DataStruct.Layer.AlphaMap); % empty is shorthand for do not render
                UpdateAlphaHandle(DataStruct.Master.Alpha);
                
            elseif strcmp(part, 'ColorMap');
                DataStruct.Master.Color = MakeMap(DataStruct.Layer.ColorMap);
                UpdateColorHandle(DataStruct.Master.Color);
                
            end
            
        else
            DataStruct.Master.Alpha = []; % empty is shorthand for do not render
            DataStruct.Master.Color = [];
            
            UpdateAlphaHandle(DataStruct.Master.Alpha);
            UpdateColorHandle(DataStruct.Master.Color);
        end
        
    end



    % REFERENCE FUNCTIONS FOR EXTERNAL USE
    function [] = AddGeneratedLayer(genLayer)
        
        index(2) = 0;
        
        [DataStruct] = AddLayer(DataStruct, [], index);
        
        
        
        
        %Set values for generated layer
        DataStruct.Layer.AlphaMap{1} = {genLayer.AlphaMap};
        DataStruct.Layer.ColorMap{1} = {genLayer.ColorMap};
        DataStruct.Layer.ScaledVolAlpha{1} = {genLayer.ScaledVolAlpha};
        DataStruct.Layer.ScaledVolColor{1} = {genLayer.ScaledVolColor};
        DataStruct.Layer.Name{1} = {genLayer.name};
        DataStruct.Layer.Info{1}{1}.PathAlphaVol = 'Generated';
        DataStruct.Layer.Info{1}{1}.PathColorVol = 'Generated';
        DataStruct.Layer.Info{1}{1}.Filter = [];
        DataStruct.Layer.Info{1}{1}.Bounds = [0 1];
        
        clear('genLayer');
        
        [displayList, indexList ] = DisplayList(DataStruct);
        
        get(layerList, 'Value')
        
        if index(1) == 0
            set(layerList, 'Value', 1);
        end
        set(layerList, 'String', displayList);
        
        UpdateLayerMaps('ColorMap');
        UpdateLayerMaps('AlphaMap');
        UpdateScaledVols();
        
    end
    
    function [] = UpdateLayer(layerStruct, part)
        
        [DataStruct] = SetLayer(DataStruct, layerStruct, index);
        
        if sum(strcmp(part, {'ColorMap' 'AlphaMap'}),2)
            UpdateLayerMaps(part);
        elseif strcmp(part, 'vols')
            UpdateScaledVols();
        end
        
        clear layerStruct;
        
        UpdateDisplayList();
        
    end

    function [sizeVec] = GetActualSize()
        
        sizeVec = size(DataStruct.Master.VolIndexAlpha);
        sizeVec = sizeVec(1:3);
    
    end

    function [cropOffset] = GetCropOffset()
        
        cropOffset = [0 0 0];
        
        if ~isempty(DataStruct.Master.VolIndexAlpha)
    
            [xCrop, yCrop, zCrop] = Crop3(DataStruct.Master.VolIndexAlpha);

            if ~isempty(xCrop)
                cropOffset = [xCrop(1)-1 yCrop(1)-1 zCrop(1)-1];
            end
        end
    end
    

end



