%Author: James W. Ryland
%June 14, 2012

function [ filterBoxUpdateHandle] = FilterBox( fig, pos, title, externalUpdateHandle )
%FILTERBOX This function is obsolete and no longer called by volumetric.
%This function allows a user to filter out high frequency sporratic voxels
%in a binary volume. It basically sets voxels with to few neighbors to
%zero.
%

    volume = double(rand(50, 50, 50)>.5);
    filteredVolume = [];

    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end

    filterPanel = uipanel('Parent', fig, 'Title', title, 'Units', 'pixels', 'Position', [pos(1) pos(2) 280 150 ] );
    
    widthLabel = uicontrol('Parent', filterPanel, 'Style', 'Text', 'String', 'Width', 'Position', [10 110 40 20]);
    
    widthEdit = uicontrol('Parent', filterPanel, 'Style', 'Edit', 'String', '3', 'Position', [60 110 40 20],...
        'CallBack', @widthEdit_CallBack);
    
    percentLabel = uicontrol('Parent', filterPanel, 'Style', 'Text', 'String', 'Percent', 'Position', [10 80 50 20]);
    
    percentEdit = uicontrol('Parent', filterPanel, 'Style', 'Edit', 'String', '0', 'Position', [60 80 40 20],...
        'CallBack', @percentEdit_CallBack);
    
    percentSlider = uicontrol('Parent', filterPanel, 'Style', 'Slider', 'String', 'Percent', 'Position', [100 80 25 20],... 
        'Max', 1, 'Min', 0, 'Value', 0, 'CallBack', @percentSlider_CallBack);
    
    applyFilterButton = uicontrol('Parent', filterPanel, 'Style', 'PushButton', 'String', 'Apply Filter', 'Position', [10 50 80 20],...
        'CallBack', @applyFilterButton_CallBack);
    
    noFilterButton = uicontrol('Parent', filterPanel, 'Style', 'PushButton', 'String', 'No Filter', 'Position', [10 20 80 20],...
        'CallBack', @noFilterButton_CallBack);
    
    %Cross Platform Formating
    uicomponents = [filterPanel widthLabel widthEdit percentLabel percentEdit percentSlider applyFilterButton noFilterButton];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
    
    
    [updateVolumeHandle updateImageHandle]= ViewBox(filterPanel, [140 0], []);
    filterBoxUpdateHandle = @filterBoxUpdate;
    noFilterButton_CallBack([],[]);
    
    
    
    % CallBacks
    function widthEdit_CallBack(h, EventData)
        width = get(widthEdit, 'String');
        width = round(str2double(width));
        set(widthEdit, 'String', num2str(width));
    end
    
    function percentEdit_CallBack(h, EventData)
        percent = str2double(get(percentEdit, 'String'));
        set(percentSlider, 'Value', percent/100);
    end

    function percentSlider_CallBack(h, EventData)
        ratio = get(percentSlider, 'Value');
        set(percentEdit, 'String', num2str(ratio*100));
    end

    function applyFilterButton_CallBack(h, EventData)
        width = str2double(get(widthEdit, 'String'));
        ratio = get(percentSlider, 'Value');
        filter = fspecial3('ellipsoid', [width width width]);
        
        filteredVolume = (convn(volume, filter, 'same')>ratio).*volume;
        
        updateVolumeHandle(filteredVolume);
        externalUpdate();
    end

    function noFilterButton_CallBack(h, EventData)
        filteredVolume = volume;
        updateVolumeHandle(filteredVolume);
        externalUpdate();
    end
    
    % Update Functions
    function externalUpdate()
        if ~isempty(externalUpdateHandle)
            if ~isempty(filteredVolume)
                externalUpdateHandle(filteredVolume);
            end
        end
    end
    
    function filterBoxUpdate(newVolume)
        volume = newVolume;
        noFilterButton_CallBack([],[]);
    end
    
    % View Modify Function none needed
    
end
