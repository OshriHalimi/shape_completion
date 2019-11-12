%Author: James W. Ryland
%June 14, 2012

function [getVolumeCAHandle, getLayersHandle] = LayerBox( fig, pos, title )
%LAYERPANEL allows a a user to make and edit layers for a CAV visualization
%
%   Fig is the parent figure or panel that LayerBox will live int. Pos is
%   the position LayerBox will occupy in the parent figure/panel. This
%   function makes use of NewLayerOptions, VolumeLayerWindow,
%   ShellLayerWindow, and GradientLayerWindow. This function is called by
%   LayerWindow.
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
    
    if isempty(title)
        title = 'Layers';
    end
    
    layerPanel = uipanel('Parent', fig, 'Title', title, 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 280 ],...
        'DeleteFcn', @layerPanel_DeleteFcn);
    
    layerList = uicontrol('Parent', layerPanel, 'Style', 'listbox', 'Position', [ 10 50 260 210]);
    
    newButton = uicontrol('Parent', layerPanel, 'Style', 'PushButton', 'String', 'New Layer', 'pos', [10 30 130 20],...
        'CallBack', @newButton_CallBack);
    
    deleteButton = uicontrol('Parent', layerPanel, 'Style', 'PushButton', 'String', 'Delete Layer', 'pos', [140 30 130 20],...
        'CallBack', @deleteButton_CallBack, 'Enable', 'off');
    
    editButton = uicontrol('Parent', layerPanel, 'Style', 'PushButton', 'String', 'Edit Layer', 'pos', [10 10 130 20],...
        'CallBack',@editButton_CallBack, 'Enable', 'off');
    
    loadButton = uicontrol('Parent', layerPanel, 'Style', 'PushButton', 'String', 'Load Layers', 'pos', [140 10 130 20],...
        'CallBack',@loadButton_CallBack);
    
    %Cross Platform Formating
    uicomponents = [layerPanel layerList newButton deleteButton editButton loadButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    
    % guts
    layers = {};
    listNames = {};
    getVolumeCAHandle = @getVolumeCA;
    getLayersHandle = @getLayers;
    
    
    % CallBacks
    function newButton_CallBack(h, EventData)
        NewLayerOptions([],@newLayerParam);
    end
    
    function deleteButton_CallBack(h, EventData)
        ind = get(layerList, 'Value');
        if ~isempty(layers{ind}.closeEditor);
            layers{ind}.closeEditor();
        end
        layers(ind) = [];
        listNames(ind) = [];
        if ind>size(listNames,2)
            if ind > 1
                set(layerList, 'Value', size(listNames,2));
            elseif ind==1
                set(layerList, 'Value', 1);
            end        
        end
        updateList();
    end

    function editButton_CallBack(h, EventData)
        ind = get(layerList, 'Value');
        startLayerEditor(ind);
    end

    function loadButton_CallBack(h, EventData)
        LayerFileWindow('Layer Load Window', [281 pos(2)], @loadLayers);
    end
    
    function layerPanel_DeleteFcn(h, EventData)
        clear('layers');
        
    end

    % Update Functions
    function newLayerParam(editor, name)
        ind = size(layers,2)+1;
        layers{ind}.CAVol = [];
        layers{ind}.settings = [];
        layers{ind}.editor = editor;
        layers{ind}.title = name;
        layers{ind}.size = [0 0 0];
        layers{ind}.layerUpdateHandle = makeLayerUpdate(ind);
        layers{ind}.closeEditor = [];
        startLayerEditor(ind);
        updateList();
    end

    function updateList()
        for i = 1:size(layers,2)
            listNames{i} = [layers{i}.title ', ' mat2str(layers{i}.size)];
            layers{i}.layerUpdateHandle = makeLayerUpdate(i);  
        end
        if size(layers,2)>0
            set(deleteButton, 'Enable', 'on');
            set(editButton, 'Enable', 'on');
        else
            set(deleteButton, 'Enable', 'off');
            set(editButton, 'Enable', 'off');
        end
        set(layerList, 'String', listNames);
    end
    
    function loadLayers(newLayers, newFileName)
        layers = newLayers;
        updateList();
    end

    % starts new layer editor and passes appropriate parameters
    function startLayerEditor(ind)
        editor = layers{ind}.editor;
        pos = [0 0]; 
        layerUpdateHandle = layers{ind}.layerUpdateHandle;
        layers{ind}.closeEditor = editor(pos, layerUpdateHandle, layers{ind}.settings);
    end

    % Makes unique updator for each layer,
    function [layerUpdateHandle] = makeLayerUpdate(ind)
        
        function layerUpdate(newCAVol, newSettings)
            disp(newSettings);
            layers{ind}.CAVol = newCAVol;
            layers{ind}.settings = newSettings;
            layers{ind}.size = size(newCAVol);
            updateList();
        end
        
        layerUpdateHandle = @layerUpdate;
    end
    
    % gets Layers
    function [myLayers] = getLayers()
        myLayers = layers;
    end

    % Takes Average By Transparency Contribution (Higher Opacities contribute more)
    function [volumeCA] = getVolumeCA()
        if ~isempty(layers)&&(~isempty(layers{1}.CAVol))
            numLayers = size(layers,2);
            [sX sY sZ nChan] = size(layers{1}.CAVol);
            
            %find bounds of data should make the following processing faster
            %too
            [bX, bY, bZ] = Bounds3(layers{1}.CAVol(:,:,:,4));
            
            for i = 2:size(layers,2)
                CAVol = double(layers{i}.CAVol);
                if 4==sum(size(CAVol)==[sX sY sZ nChan]);
                    [bXs, bYs, bZs] = Bounds3(CAVol(:,:,:,4));
                    bX(1) = min([bXs(1), bX(1)]);
                    bX(2) = max([bXs(2), bX(2)]);
                    bY(1) = min([bYs(1), bY(1)]);
                    bY(2) = max([bYs(2), bY(2)]);
                    bZ(1) = min([bZs(1), bZ(1)]);
                    bZ(2) = max([bZs(2), bZ(2)]);
                    
                end
            end
            
            cX = length(bX(1):bX(2));
            cY = length(bY(1):bY(2));
            cZ = length(bZ(1):bZ(2));
            
            alphaTotal = zeros(cX, cY, cZ);
            allAlpha = zeros(cX, cY, cZ,numLayers);
            colorTotal = zeros(cX, cY, cZ, 3);
            volumeCA = zeros(cX,cY,cZ,4,'uint8');
            
            bX
            bY
            bZ
            
            %combining layers using alpha weighted voxel average
            disp('Conbining Layers');
            for i = 1:size(layers,2)
                CAVol = double(layers{i}.CAVol);
                if 4==sum(size(CAVol)==[sX sY sZ nChan]);
                    CAVol = CAVol(bX(1):bX(2),bY(1):bY(2),bZ(1):bZ(2),:);
                    size(CAVol)
                    size(alphaTotal)
                    alphaTotal = alphaTotal + CAVol(:,:,:,4);
                    colorTotal(:,:,:,1) = colorTotal(:,:,:,1) + squeeze(CAVol(:,:,:,1)).*CAVol(:,:,:,4);
                    colorTotal(:,:,:,2) = colorTotal(:,:,:,2) + squeeze(CAVol(:,:,:,2)).*CAVol(:,:,:,4);
                    colorTotal(:,:,:,3) = colorTotal(:,:,:,3) + squeeze(CAVol(:,:,:,3)).*CAVol(:,:,:,4);
                    allAlpha(:,:,:,i) = CAVol(:,:,:,4);
                end
            end
            
            disp('Smoothing Edges');
            %maxAlpha = smooth3(max(allAlpha, [], 4));
            maxAlpha = max(allAlpha, [], 4);
            volumeCA(:,:,:,1) = uint8(colorTotal(:,:,:,1)./(alphaTotal+.000001)); 
            volumeCA(:,:,:,2) = uint8(colorTotal(:,:,:,2)./(alphaTotal+.000001));
            volumeCA(:,:,:,3) = uint8(colorTotal(:,:,:,3)./(alphaTotal+.000001));
            volumeCA(:,:,:,4) = uint8(maxAlpha);
            clear('alphaTotal');
            clear('allAlpha');
            clear('colorTotal');
        
        else
            volumeCA(:,:,:,1) = zeros(40,40,40,'uint8');
            volumeCA(:,:,:,2) = zeros(40,40,40,'uint8');
            volumeCA(:,:,:,3) = zeros(40,40,40,'uint8');
            volumeCA(:,:,:,4) = zeros(40,40,40,'uint8');
        end
        
    end

end
