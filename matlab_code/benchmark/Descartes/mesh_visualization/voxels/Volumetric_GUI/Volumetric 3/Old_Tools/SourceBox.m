%Author: James W. Ryland
%June 14, 2012

function [getSettingsHandle getBoundsHandle] = SourceBox( fig, title, pos, externalUpdateHandle, initSettings )
%SOURCEBOX allows the user to select region of volume from a matrix by
%setting a minimum and maximum bound.
%   fig is the figure or panel that will be the parent to SourceBox's
%   graphical components, pos is the position that SourceBox will occupy in
%   the parent window or panel. ExternalUpdateHandle update the parent
%   function with the selected region represented by a binary volume and
%   the the whole version of the source volume.

    sourceVolume = [];
    selectionVolume = [];
    maxV = [];
    minV = [];
    
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
       
    sourcePanel = uipanel('Parent', fig, 'Title', title, 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 140 ],...
        'DeleteFcn', @sourcePanel_DeleteFcn);
        
    setSourceButton = uicontrol('Parent', sourcePanel, 'Style', 'PushButton', 'String', 'Set Source', 'Position', [10 100 100 20],...
        'CallBack', @setSourceButton_CallBack);

    setSelectionButton = uicontrol('Parent', sourcePanel, 'Style', 'PushButton', 'String', 'Set Selection', 'Position', [10 10 100 20],...
        'CallBack', @setSelectionButton_CallBack);
    
    maxLabel = uicontrol('Parent', sourcePanel, 'Style', 'Text', 'String', 'Max', 'Position', [10 65 30 20]);
    
    minLabel = uicontrol('Parent', sourcePanel, 'Style', 'Text', 'String', 'Min', 'Position', [10 40 30 20]);
    
    maxEdit = uicontrol('Parent', sourcePanel, 'Style', 'Edit', 'String', '1', 'Position', [50 65 30 20],...
        'CallBack', @maxEdit_CallBack);
    
    minEdit = uicontrol('Parent', sourcePanel, 'Style', 'Edit', 'String', '0', 'Position', [50 40 30 20],...
        'CallBack', @minEdit_CallBack);
    
    maxSlider = uicontrol('Parent', sourcePanel, 'Style', 'Slider', 'String', 'Max', 'Position', [80 65 25 20],...
        'CallBack', @maxSlider_CallBack);
    
    minSlider = uicontrol('Parent', sourcePanel, 'Style', 'Slider', 'String', 'Min', 'Position', [80 40 25 20],...
        'CallBack', @minSlider_CallBack);
    
    depthSlider = uicontrol('Parent', sourcePanel, 'Style', 'Slider', 'String', 'Depth','Position', [140 0 140 20],...
         'Value', 1, 'Min', 1, 'Max', 2, 'CallBack', @depthSlider_CallBack);
    
    %Cross Platform Formating
    uicomponents = [sourcePanel setSourceButton setSelectionButton maxLabel minLabel maxEdit minEdit maxSlider minSlider depthSlider];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    getSettingsHandle = @getSettings;
    getBoundsHandle = @getBounds;
     
    viewAxis = axes('Parent', sourcePanel, 'Units', 'pixels', 'Position', [140 20 140 100 ]);
    
    set(0,'CurrentFigure',fig);
    set(fig,'CurrentAxes', viewAxis);
    defaultImage = rand(50,50,3);
    image(defaultImage);
    axis('off');
    axis('image');
    
    % initialize previous settings
    if ~isempty(initSettings)
        updateSource(initSettings.sourceVolume, initSettings.selectionVolume);
        set(maxSlider, 'Value', initSettings.maxV);
        set(minSlider, 'Value', initSettings.minV);
        updateSelectionParam();
        updateView();
        externalUpdate();
    end
    
    
    % Call Backs
    function setSourceButton_CallBack(h, EventData)
        % make file selection window, pass internal
        FileWindow('File Window',[],@updateSource);
    end

    
    function setSelectionButton_CallBack(h, EventData)
        externalUpdate();
    end

    function depthSlider_CallBack(h, EventData)
        updateView();
    end

    function sourcePanel_DeleteFcn(h, EventData)
        clear('sourceVolume');
        clear('selectionVolume');
    end

    % Sliders
    function maxSlider_CallBack(h, EventData)
        set(maxEdit, 'String', num2str(get(maxSlider,'Value')));
        updateSelectionParam();
        updateView();
    end

    function minSlider_CallBack(h, EventData)
        set(minEdit, 'String', num2str(get(minSlider,'Value')));
        updateSelectionParam();
        updateView();
    end
    
    % Edits
    % needs exception handling
    function maxEdit_CallBack(h, EventData)
        set(maxSlider, 'Value', str2double(get(maxEdit, 'String')));
        updateSelectionParam();
        updateView();
    end

    function minEdit_CallBack(h, EventData)
        set(minSlider, 'Value', str2double(get(minEdit, 'String')));
        updateSelectionParam();
        updateView();
    end
    
    % Update Functions
    
    function updateSource(newSource,fileName)
        sourceVolume = double(newSource); 
        [sx sy sz] = size(sourceVolume);
        selectionVolume = zeros(sx,sy,sz,'uint8');
        set(depthSlider,'Max', sx, 'Value', sx/2);
        maxV = max(max(max(sourceVolume)));
        minV = min(min(min(sourceVolume)));
        set(maxSlider, 'Max', maxV+1, 'Min', minV-1, 'Value', maxV);
        set(minSlider, 'Max', maxV+1, 'Min', minV-1, 'Value', minV);
        set(maxEdit, 'String', maxV);
        set(minEdit, 'String', minV);
        updateView();
    end

    function updateSelectionParam()
        maxV = get(maxSlider, 'Value');
        minV = get(minSlider, 'Value');
    end

    function [c] = updateSelectionSlice()
        index = round(get(depthSlider, 'Value'));
        mono = squeeze(sourceVolume(index,:,:));
        high = ((minV<=mono)&(mono<=maxV));
        mono = mono/max(max(mono));
        c(:,:,3)=mono.*(high==0); 
        c(:,:,2)=mono.*(high==0); 
        c(:,:,1)=(high==1)+mono.*(high==0);
    end

    function updateView()
        c = updateSelectionSlice();
        set(0,'CurrentFigure',fig)
        set(fig,'CurrentAxes', viewAxis);
        image(c);
        axis('off');
        axis('image');
    end
    
    % External interactions
    function externalUpdate()
        if ~isempty(externalUpdateHandle)
            if ~isempty(sourceVolume)
                selectionVolume = uint8((minV<=sourceVolume)&(sourceVolume<=maxV));
                externalUpdateHandle(selectionVolume, sourceVolume);
            end
        end
    end

    %settings related functions
    function [settings] = getSettings()
       settings.sourceVolume = sourceVolume;
       settings.selectionVolume = selectionVolume;
       settings.maxV = maxV;
       settings.minV = minV;
    end

    function [min max] = getBounds()
        min = minV;
        max = maxV;
    end
end
