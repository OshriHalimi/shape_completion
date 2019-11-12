function [ UpdateEditLayerHandle, fig] = EditWindow( layerWindowPos, UpdateLayerHandle, layFig )
% EDITWINDOW makes a window that allows the user to edit a specific layers
% color maps input and scaling.
    
    % INITIALIZE GUI
    
    pos = [];
    
    height = 500;
    width = 400;
    
    scr = get(0,'ScreenSize');
    
    if ~isempty(layerWindowPos())
        pos = [ 1 (layerWindowPos(2)-height-100)];
    else
        pos = [ 1 (scr(4)-height)];
    end

    title = 'Edit';
    
    fig = figure('Name',title, 'Resize', 'off', 'NumberTitle', 'off', 'Position', [pos(1) pos(2) width height], 'CloseRequestFcn', @MyCloseRequestFcn, 'DeleteFcn', @MyDeleteFcn, 'MenuBar', 'None');
    
    sX = .9;
    sY = .05;
    spX = (1-sX)/2;
    spY = .925;
    nameEdit = uicontrol('Parent', fig, 'Style', 'Edit', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @nameEdit_Callback); 
    set(nameEdit, 'String', 'Name', 'FontSize', 15);
    
    sX = .9;
    sY = .1;
    spX = (1-sX)/2;
    spY = .8;
    loadColorVolButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @colorVolButton_Callback); 
    set(loadColorVolButton, 'String', 'Load Color Source', 'FontSize', 20);
    sY = .05;
    spY = .8-.05;
    colorPathText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(colorPathText, 'String', 'Path\\');
    
    sX = .70;
    sY = .1;
    spX = .05;
    spY = .6;
    alphaAdvancedOpt = 'off';
    loadAlphaVolButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @alphaVolButton_Callback);
    set(loadAlphaVolButton, 'String', 'Load Alpha Source', 'FontSize', 20, 'enable', alphaAdvancedOpt);
    sX = .9;
    sY = .05;
    spY = .6-.05;
    alphaPathText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(alphaPathText, 'String', 'Path\\');
    
    sX = .20;
    sY = .1;
    spX = .75;
    spY = .6;
    alphaSourceToggle = uicontrol('Parent', fig, 'Style', 'Togglebutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @alphaSourceToggle_Callback);
    set(alphaSourceToggle, 'String', 'Advanced', 'FontSize', 12);
    
    sX = .15;
    sY = .1;
    spX = .05;
    spY = .3;
    colorPickerPixSize = round([sY*height*.8 sX*width*.8]);
    colorPickerButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @colorPickerButton_Callback);
    set(colorPickerButton, 'String', 'Color', 'FontSize', 12, 'CData', ones([colorPickerPixSize, 3]));
    
    sX = .1;
    sY = .08;
    spY = .4;
    im1 = imread('Pallet.jpg');
    % scale image for button
    pixSize = round([sY*height*.80 sX*width*.80]);
    im1 = imresize(im1,pixSize);
    colorPickerImg = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'cdata', im1,  'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @colorPickerButton_Callback);
    
    
    
    sX = .15;
    sY = .1;
    spX = .20;
    spY = .3;
    alphaPickerPixSize = round([sY*height*.8 sX*width*.8]);
    alphaPickerButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @alphaPickerButton_Callback);
    set(alphaPickerButton, 'String', 'Opacity', 'FontSize', 12, 'CData', ones([alphaPickerPixSize, 3]));
    
    
    sX = .1;
    sY = .08;
    spY = .4;
    im1 = imread('AlphaIcon100.jpg');
    % scale image for button
    im1 = imresize(im1,pixSize);
    alphaPickerImg = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'cdata', im1,  'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @alphaPickerButton_Callback);
    
    
    sX = .5;
    sY = .05;
    spX = .4;
    spY = .3;
    dStrength = .1;
    strengthSlider = uicontrol('Parent', fig, 'Style', 'Slider', 'Value', dStrength, 'Min', 0, 'Max', 1, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @strengthSlider_Callback);
    spY = .35;
    strengthText = uicontrol('Parent', fig, 'Style', 'Text', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(strengthText, 'String', 'Brush Strength', 'FontSize', 20);
    
    
    sX = .75;
    sY = .1;
    spX = .05;
    spY = .2;
    colorMapDisp = axes('Parent', fig, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'ButtonDownFcn', @colorMapDisp_ButtonDownFcn);
    set(colorMapDisp, 'xLim', [0, 64]);
    set(colorMapDisp, 'XTick', [], 'YTick', [], 'ZTick', []);
    colorMapSurf = surface([0 64; 0 64],[0 0; 1 1], [0 0; 0 0]);
    set(colorMapSurf,'facecolor','texture', 'cdatamapping', 'direct','edgealpha',1, 'parent', colorMapDisp, 'ButtonDownFcn', @colorMapDisp_ButtonDownFcn);
    daspect([ width*sX*1, height*sY*1, 1]);
    
    
    
    sX = .75;
    sY = .1;
    spX = .05;
    spY = .1;
    alphaMapDisp = axes('Parent', fig, 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY]);
    set(alphaMapDisp, 'xLim', [0, 64]);
    set(alphaMapDisp, 'XTick', [], 'YTick', [], 'ZTick', []);
    alphamap((0:2047)/2047)
    
    % BackGround for alpha Map
    hold on;
    alphaMapBack = surface([0 64; 0 64],[0 0; 1 1], [0 0; 0 0]);
    set(alphaMapBack,'facecolor','black', 'cdatamapping', 'direct','edgealpha',1, 'parent', alphaMapDisp, 'ButtonDownFcn', @alphaMapDisp_ButtonDownFcn);
    daspect([ width*sX*1, height*sY*1, 1]);
     
    % Make ten of these to simulate layered transparency additivity
    numTrans = 10;
    alphaMapSurf = zeros(numTrans,1);
    for i=1:numTrans 
        alphaMapSurf(i) = surface([0 64; 0 64],[0 0; 1 1], [i i; i i]);
    end
    
    set(alphaMapSurf,'facecolor','texture', 'facealpha','texturemap', 'AlphaDataMapping', 'scaled','edgealpha',1, 'cdata', ones(1,64,3), 'alphadata', ones(1,64), 'parent', alphaMapDisp, 'ButtonDownFcn', @alphaMapDisp_ButtonDownFcn);
    daspect([ width*sX*1, height*sY*1, 1]);
    hold off;
    
    
    
    % alpha presets
    [aPreset, aPresetNames] = AlphaMapPresets();
    sX = .4;
    sY = .05;
    spX = .05;
    spY = .05;
    aPresetMenu = uicontrol('Parent', fig, 'Style', 'popupmenu', 'String', 'Alpha Presets', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @aPresetMenu_Callback);
    set(aPresetMenu, 'String', ['Alpha Presets' aPresetNames]);
    
    
    % alpha presets
    [cPreset, cPresetNames] = ColorMapPresets([1 1 1]);
    sX = .4;
    sY = .05;
    spX = .5;
    spY = .05;
    cPresetMenu = uicontrol('Parent', fig, 'Style', 'popupmenu', 'String', 'Alpha Presets', 'Units', 'normalized' ,'Position', [ spX, spY, sX, sY],'FontSize', 15, 'Callback', @cPresetMenu_Callback);
    set(cPresetMenu, 'String', ['Color Presets' cPresetNames]);
    
    
    sX = .15;
    sY = .1;
    spX = .80;
    spY = .2;
    colorSmoothButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @colorSmoothButton_Callback);
    set(colorSmoothButton, 'String', 'Smooth', 'FontSize', 12);
    
    
    sX = .15;
    sY = .1;
    spX = .80;
    spY = .1;
    alphaSmoothButton = uicontrol('Parent', fig, 'Style', 'Pushbutton', 'Units', 'Normalized', 'Position', [ spX, spY, sX, sY], 'Callback', @alphaSmoothButton_Callback);
    set(alphaSmoothButton, 'String', 'Smooth', 'FontSize', 12);
    
    
    
    %INITIALIZE
    layerStruct.AlphaMap = zeros(64,1);
    layerStruct.ColorMap = zeros(64,3);
    layerStruct.ScaledVolAlpha = [];
    layerStruct.ScaledVolColor = [];
    layerStruct.Name = [];
    layerStruct.Info = [];
    
    brushShape = [.1 .2 .3 .4 .5 .4 .3 .2 .1]'/.5;
    colorTarg = [1 1 1];
    alphaTarg = 1/numTrans;
    alpha = 1;
    strength = dStrength;
    
    trainTime = .5;
    clickTrain = 0;
    
    fileTypes ={'*.mat;*.img', 'Volume files (*.mat, *.img)';
                    '*.mat', 'MATLAB files (*.mat)';
                    '*.img', 'Image files (*.img)'};
    
    UpdateEditLayerHandle = @UpdateEditLayer; 
    ColorPickedHandel = @ColorPicked;
    
    
    %CALLBACKS
    function nameEdit_Callback(hObject, eventData, handles)
        layerStruct.Name = get(nameEdit, 'String');
        UpdateLayerHandle(layerStruct, 'Name');
    end

    function aPresetMenu_Callback(hObject, eventData, handles)

            presetInd = get(aPresetMenu, 'value')-1;

            if presetInd>0
                layerStruct.AlphaMap = aPreset{presetInd};
                SelectionChanged();
                UpdateLayerHandle(layerStruct, 'AlphaMap');
            end

            set(aPresetMenu, 'value', 1);

    end

    function cPresetMenu_Callback(hObject, eventData, handles)

            [cPreset, cPresetNames] = ColorMapPresets(colorTarg);
            
            presetInd = get(cPresetMenu, 'value')-1;

            size(layerStruct.ColorMap)
            size(cPreset{1})
            
            if presetInd>0
                layerStruct.ColorMap = cPreset{presetInd};
                SelectionChanged();
                UpdateLayerHandle(layerStruct, 'ColorMap');
            end

            set(cPresetMenu, 'value', 1);

    end

    function colorSmoothButton_Callback(hObject, eventData, handles)

        w = 5;
        filt = ones(w,1)/w;
        layerStruct.ColorMap = convn(layerStruct.ColorMap, filt, 'same');
        SelectionChanged();
        UpdateLayerHandle(layerStruct, 'ColorMap');

    end

    function alphaSmoothButton_Callback(hObject, eventData, handles)

        w = 5;
        filt = ones(w,1)/w;
        layerStruct.AlphaMap = convn(layerStruct.AlphaMap, filt, 'same');
        SelectionChanged();
        UpdateLayerHandle(layerStruct, 'AlphaMap');

    end

    % use UIGETFILE to select input files YAY!!! this will be relatively
    % easy!! Then make your wizard for setting the import options
    function colorVolButton_Callback(hObject, eventData, handles)
          
        [filename, pathname] = uigetfile(fileTypes);
        
        if ~(isnumeric(filename) && filename==0)
            
            scaledVol = LoadFile(pathname, filename);
            layerStruct.ScaledVolColor = scaledVol;
            layerStruct.Info.PathColorVol = [pathname, filename];
            if DimCheck
                ColorMask();
            end
            UpdateLayerHandle(layerStruct, 'vols');
            
            set(colorPathText, 'String', [pathname, filename]);
            
            % If not using advanced alpha source use color source as alpha
            % source
            if strcmp(alphaAdvancedOpt, 'off')
                layerStruct.ScaledVolAlpha = scaledVol;
                layerStruct.Info.PathAlphaVol = [pathname, filename];
                if DimCheck
                    ColorMask();
                end
                UpdateLayerHandle(layerStruct, 'vols');

                set(alphaPathText, 'String', [pathname, filename]);
            end
            
        end
        
    end


    function alphaVolButton_Callback(hObject, eventData, handles)
          
        [filename, pathname] = uigetfile(fileTypes);
        
        if ~(isnumeric(filename) && filename==0)
            
            scaledVol = LoadFile(pathname, filename);
            layerStruct.ScaledVolAlpha = scaledVol;
            layerStruct.Info.PathAlphaVol = [pathname, filename];
            if DimCheck
                ColorMask();
            end
            UpdateLayerHandle(layerStruct, 'vols');
            
            set(alphaPathText, 'String', [pathname, filename]);
            
        end
        
    end
    
    function alphaSourceToggle_Callback(hObject, eventData, handles)
        
        if strcmp(alphaAdvancedOpt, 'off')
            alphaAdvancedOpt = 'on';
        else
            alphaAdvancedOpt = 'off';
        end
        
        set(loadAlphaVolButton,'enable', alphaAdvancedOpt);
        
    end
    
    function colorMapDisp_ButtonDownFcn(hObject, eventData)
        
        clickPoint = get(colorMapDisp, 'CurrentPoint');
        centerInd = ceil(clickPoint(1,1));
        layerStruct.ColorMap = ApplyBrush(layerStruct.ColorMap, centerInd, colorTarg);
        layerStruct.ColorMap(1,:) = [0 0 0]; % ensure 1 is non-colored
        SelectionChanged();
        
        clickTrain = clickTrain+1;
        pause(trainTime);
        clickTrain = clickTrain-1;
        
        clickTrain
        
        % Only update after a click train is done
        if clickTrain == 0;
            UpdateLayerHandle(layerStruct, 'ColorMap');
        end
    end

    function alphaMapDisp_ButtonDownFcn(hObject, eventData)
        
        clickPoint = get(alphaMapDisp, 'CurrentPoint');
        centerInd = ceil(clickPoint(1,1));
        layerStruct.AlphaMap = ApplyBrush(layerStruct.AlphaMap, centerInd, alphaTarg);
        layerStruct.AlphaMap(1,:) = 0; % ensure 1 is transparent
        SelectionChanged();
        
        clickTrain = clickTrain+1;
        pause(trainTime);
        clickTrain = clickTrain-1;
        
        clickTrain
        
        % Only update after a click train is done
        if clickTrain == 0;
            UpdateLayerHandle(layerStruct, 'AlphaMap');
        end
        
        
    end

    function strengthSlider_Callback(hObject, eventData, handles)
        
        strength = get(strengthSlider, 'Value');
        
    end

    function colorPickerButton_Callback(hObject, eventData, handles)
        
        color = uisetcolor(colorTarg);
        
        if size(color,2) == 3 && sum(color==colorTarg)~=3;
            ColorPicked(color);
        end
        
    end

    function alphaPickerButton_Callback(hObject, eventData, handles)
       
        % Will only use the gray scale output
        alphaOld = alpha;
        alpha = uisetcolor([1,1,1]*alpha);
        
        if size(alpha,2) == 3;
            alpha = sum(alpha)/3;
            AlphaPicked(alpha);
        else
            alpha = alphaOld;
        end
        
    end

    function MyCloseRequestFcn(hObject, eventData)
        close(layFig);
    end

    function [] = MyDeleteFcn(hObject, eventData);
        clear layerStruct;
        delete(alphaMapSurf);
        cla(alphaMapDisp);
        delete(colorMapSurf);
        cla(colorMapDisp);
    end
    
   
    
    %HELPER FUNCTIONS
    function SelectionChanged()
        set(nameEdit, 'String', layerStruct.Name);
        set(colorPathText, 'String', layerStruct.Info.PathColorVol);
        set(alphaPathText, 'String', layerStruct.Info.PathAlphaVol);
        
        dispMatC(1,:,:) = layerStruct.ColorMap;
        set(colorMapSurf, 'CData', dispMatC);
        
        dispMatA(1,:,1) = layerStruct.AlphaMap;
        set(alphaMapSurf, 'AlphaData', dispMatA(:,:,1));
        clear dispMatC;
        clear dispMatA;
    end

    function [map] = ApplyBrush(map, centerInd, targ) % temporary
        
        brushRad = floor(length(brushShape)/2);
        brushCen = round(length(brushShape)/2);
        
        ind1 = max((centerInd-brushRad), 1);
        ind2 = min((centerInd+brushRad), 64);
        rad1 = centerInd-ind1;
        rad2 = ind2-centerInd;
        
        indB1 = brushCen - rad1;
        indB2 = brushCen + rad2;
        
        subsetMap = map(ind1:ind2,:);
        subsetBrush = brushShape(indB1:indB2,:);
        
        %size(subsetMap*(1-strength))
        %size((subsetBrush*targ)*strength)
        %size(map(ind1:ind2,:))
        
        nChan = ones(1,size(map,2));
        
        disp(targ)
        
        map(ind1:ind2,:) = subsetMap.*(1-subsetBrush*nChan*strength) + (subsetBrush*targ)*strength;
        
        disp(map)
        
    end

    function [vol] = LoadFile(path, fileName)
        
        oldFolder = cd( path );
        
        [pathst, name, fileType] = fileparts(fileName);
        
        if strcmp(fileType, '.img')
            %cd (pathst)
            vol = analyze75read([fileName]);
            %cd (oldFolder);
        elseif strcmp(fileType, '.mat')
            
            tempStruct = load(fileName);
            tempCell = struct2cell(tempStruct);
            vol = tempCell{1};
            clear('tempStruct');
            clear('tempCell');
            
        end
        
        cd(oldFolder);
        
        
        % check for vector format and reshape accordingly
        if isvector(vol)
            [reshapeVec, colWise] = VecReadOptions([],length(vol));
            
            if colWise==1
                vol = reshape(vol, reshapeVec(1), reshapeVec(2), reshapeVec(3));
            
            elseif colWise==0
                vol = reshape(vol, reshapeVec(2), reshapeVec(1), reshapeVec(3));
                vol = permute(vol, [2 1 3]);
            
            end
            
        end
        
        % implement a wrap around feature, putting negative values after
        % max
        % remove negative values for now
        range = double([min(vol(:)) max(vol(:))]);
        smooth = 0;
        interval = range(2)-range(1);
        bins = range(1):(interval/50):range(2);
        [counts, smooth] = histc(vol(:), bins);
        
        % Ask User for input options
        [range, maxToZero, toMask, toShell, shift] = ImportOptions([], range, counts, bins);
        
        vol = double(vol);
        
        
        % Deal with range
        if maxToZero
            vol = vol.*(vol<=range(2));
        else
            vol(vol>range(2)) = range(2);
        end
        
        vol = vol.*double(vol>=range(1));
        vol = vol-range(1);
        vol = double(vol);
        vol = vol.*(vol>0);
        
        % deal with masking
        if toMask
            vol = double(vol~=0);
        end
        
        % deal with smoothing
        if smooth
            vol = smooth3(vol, 'box', 5);
        end
        
        
         % should take precedence over smoothing and toMask
        if toShell
            vol = double(vol>0);
            vol = smooth3(vol, 'box', 5);
            vol(vol==1) = 0; 
            
        end
        
        % Check for offset>Essentially to change where the zeros are placed<
        vol = vol+shift;
        
        % convert to indexed volume
        vol = uint8(vol/max(vol(:))*255);
        
    end

    function ColorMask()
        % masks the color with the alphas 0 values
        layerStruct.ScaledVolColor = layerStruct.ScaledVolColor.*uint8(layerStruct.ScaledVolAlpha~=0);
    end

    function [match] = DimCheck()
        
        match = 0;
        
        if (~isempty(layerStruct.ScaledVolColor))&&(~isempty(layerStruct.ScaledVolAlpha))
            
            colSize = size(layerStruct.ScaledVolColor);
            alpSize = size(layerStruct.ScaledVolAlpha);

        
            if length(colSize)==length(alpSize) && sum(colSize==alpSize)==3
                match = 1;
            end
            
            if match == 1;
                disp('Alpha Vol and Color Vol size match');
            else
                disp('Alpha Vol and Color Vol size mis-match'); % later throw error !
            end
            
        end
        
    end

    %EXTERNAL REFERENCE FUNCTIONS
    function [] = UpdateEditLayer( layerStructNew)
        
        layerStruct = layerStructNew;
        SelectionChanged();
        clear layerStructNew;
    end

    function [] = ColorPicked(color)
        
        colorTarg = color;
        
        displayColor(:,:,1) = ones([colorPickerPixSize,1]) * colorTarg(1);
        displayColor(:,:,2) = ones([colorPickerPixSize,1]) * colorTarg(2);
        displayColor(:,:,3) = ones([colorPickerPixSize,1]) * colorTarg(3);
        
        set(colorPickerButton, 'Cdata', displayColor)
        
    
    end

    function [] = AlphaPicked(alpha_)
    
        displayAlpha(:,:,1) = ones([alphaPickerPixSize, 1]) * alpha;
        displayAlpha(:,:,2) = ones([alphaPickerPixSize, 1]) * alpha;
        displayAlpha(:,:,3) = ones([alphaPickerPixSize, 1]) * alpha;
        
        % Closed form
        %y = 1-(1-x).^l;
        % -(y+1) = (1-x).^l
        % (-(y+1)).^1/l = 1-x
        % 1-(-1(y+1)).^1/l = x
        % remember to use l+1
        
        alphaTarg = -(-alpha+1).^(1/(numTrans+1))+1;
        
        disp(alphaTarg)
        
        set(alphaPickerButton, 'Cdata', displayAlpha);
    end

end

