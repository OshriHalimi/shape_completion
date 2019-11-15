%Author: James W. Ryland
%June 14, 2012

function [updateVolumeHandle updateImageHandle] = ViewBox( fig, pos, customeEditHandle)
%VIEWBOX this function creates a panel that allows the user to look at 2d
%slices from either a scaler volume or a CAV volume.
%   fig is the parent figure that the ViewBox will be placed in, pos is the
%   position in the parent figure that ViewBox will occupy. The
%   costomeEditHandle allows the parent figure or panel to determine the
%   specifics of how a slice is rendered.
    
    viewVolume = uint8(rand(50,30,40));
    viewAngle = 1;
    
    if isempty(fig)
        fig = figure();
    end
    
    if isempty(pos)
        pos = [0 0];
    end
        
    viewPanel = uipanel('Parent', fig, 'Units', 'pixels', 'Position', [pos(1) pos(2) 140 140 ] );
    
    axisGroup = uibuttongroup('Parent', viewPanel,'Units', 'Pixels', 'Position', [120 20 20 120],...
        'SelectionChangeFcn', @axisGroup_SelectionChangeFcn);
    
    xRadio = uicontrol('Parent', axisGroup, 'Style', 'ToggleButton', 'String', 'X','Position', [0 0 20 40]);
    
    yRadio = uicontrol('Parent', axisGroup, 'Style', 'ToggleButton', 'String', 'Y','Position', [0 40 20 40]);
    
    zRadio = uicontrol('Parent', axisGroup, 'Style', 'ToggleButton', 'String', 'Z','Position', [0 80 20 40]);
    
    depthSlider = uicontrol('Parent', viewPanel, 'Style', 'Slider', 'String', 'Depth','Position', [0 0 140 20],...
         'Value', 1, 'Min', 1, 'Max', 2, 'CallBack', @depthSlider_CallBack);
    
    %Cross Platform Formating
    uicomponents = [ xRadio yRadio zRadio];
    set(uicomponents,'FontUnits', 'pixels', 'FontSize', 12, 'FontName', 'FixedWidth');
     
     
    viewAxis = axes('Parent', viewPanel, 'Units', 'pixels', 'Position', [ 0 20 120 100 ]);
    
    
    % sets default
    updateVolumeHandle = @updateVolume;
    updateImageHandle = @updateImage;
    updateVolume(viewVolume);
    
    
    % Call Backs
    function depthSlider_CallBack(h, EventData)
        updateView();
    end
    
    function axisGroup_SelectionChangeFcn(h, eventData)
        [sx sy sz dum] = size(viewVolume);
        switch eventData.NewValue
            case xRadio
                set(depthSlider, 'Max', sx, 'Value', sx/2);
                viewAngle = 1;
            case yRadio
                set(depthSlider, 'Max', sy, 'Value', sy/2);
                viewAngle = 2;
            case zRadio
                set(depthSlider, 'Max', sz, 'Value', sz/2);
                viewAngle = 3;
        end
        updateView();
    end
    

    % Update Functions
    function updateView()
        imageSlice = [];
        i = round(get(depthSlider, 'Value'));
        switch viewAngle
            case 1
                imageSlice = squeeze(viewVolume(i,:,:,:));
            case 2
                imageSlice = squeeze(viewVolume(:,i,:,:));
            case 3
                imageSlice = squeeze(viewVolume(:,:,i,:));
        end
        
        if ~isempty(customeEditHandle)
            imageSlice = customeEditHandle(imageSlice);
        
        else
            maxV = max(max(max(imageSlice)));
            minV = min(min(min(imageSlice)));
            if ((maxV>1)||(minV<0))
                imageSlice = uint8((double(imageSlice)-double(minV))/(double(maxV)-double(minV))*255);
            
            elseif ((maxV<=1)||(minV>=0))
                imageSlice = uint8(imageSlice*255);
            
            elseif (maxV-minV)==0
                imageSlice = zeros(size(imageSlice), 'uint8');
            end
        end
        
    	axes(viewAxis);
        image(imageSlice);
        axis('off');
        axis('image');
    end

    function updateVolume(newVolume)
        % EDIT THIS new volume is not scaled at all!!!!!
        % This produces clipping at the upper end 300 becomes 255
        maxV = max(max(max(newVolume)));
        minV = min(min(min(newVolume)));
        
        scaleVolume = uint8( ((newVolume-minV)/(maxV-minV))*255 );
        
        [sx sy sz sc] = size(scaleVolume);
        set(depthSlider, 'Max', sx, 'Value', sx/2);
        viewAngle = 1;
        viewVolume = zeros(0,0, 'uint8');
        if sc==1
            viewVolume(:,:,:,1) = scaleVolume;
            viewVolume(:,:,:,2) = scaleVolume;
            viewVolume(:,:,:,3) = scaleVolume;
        elseif sc==3
            viewVolume = scaleVolume;
        end
        updateView();
    end

    function updateImage()
        updateView();
    end
end
