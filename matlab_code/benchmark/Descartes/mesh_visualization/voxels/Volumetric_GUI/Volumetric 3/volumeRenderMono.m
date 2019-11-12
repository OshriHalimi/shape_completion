%Author: James W. Ryland
%June 14, 2012

function [slices1, slices2, slices3] = volumeRenderMono( CA, slices1, slices2, slices3, rendAxes)
% [slices1 slices2 slices3] = volumeRender(ColorAlphaVolume, slices1
% slices2 slices3, axisToRenderOn);
% VOLUMERENDER is the core function that creates the simple surfaces 
% or (slices) from the ColorAlphaVolume. All of the three dimensional
% visualizations in volumetric are based on this technique; However it is
% left up to the different render windows functions to decide how to
% manipulate the slice visibility. Generally to reduce visual artifacts
% only the slices along the axis with the greatest projection onto the view
% axis will be made visible by the render windows. Each surface has its
% alpha data and CDATA changes to that it will match the ColorAlphaVolume
% based on its position in space.

    disp('rendering');
    
    
    %cdatAll = zeros(sX,sY,sZ);
    %adatAll = zeros(sX,sY,sZ);
    
    
    
    cdatAll = CA(:,:,:,1);
    adatAll = CA(:,:,:,2);
    
    [Xs, Ys, Zs] = Crop3(adatAll);
    
    cdatAll = cdatAll(Xs,Ys,Zs);
    adatAll = adatAll(Xs,Ys,Zs);
    
    [sX, sY, sZ] = size(cdatAll);
    
    if isempty(rendAxes)
        rendAxes = axes();
    end
    
    %important for displaying data correctly (matlab likes the y axis to face)
    %   In a different direction than things like native fmrie data...
    temp = load('GlobalSettings.mat');
    if strcmp(temp.GlobalSettings.yInverse,'yes');
        set(rendAxes, 'YDir', 'reverse');
    else
        set(rendAxes, 'YDir', 'normal');
    end
    
    
    rerender = 0;
    
    if (isempty(slices1)||isempty(slices2)||isempty(slices3))
        slices1 = zeros(sZ,1);
        slices2 = zeros(sY,1);
        slices3 = zeros(sX,1);
        rerender = 1;
    end
    
    
    % The Rendering
    
    %create z planes
    for i = 1:sZ
        
        cdat = zeros(sX, sY);
        adat = zeros(sX, sY);
        cdat(:,:) = squeeze(cdatAll(:,:,i));
        adat(:,:) = squeeze(adatAll(:,:,i));
        cdat = permute(cdat, [2 1 3]);
        adat = permute(adat, [2 1 3]);
        
        if rerender==1
            slices1(i) = surface([0 sX; 0 sX], [0 0; sY sY], [i i; i i]);
        end
        
        set(slices1(i), 'cdatamapping', 'direct', 'facecolor','texture', 'cdata', cdat,...
            'edgealpha',0,'alphadata',double(adat), 'facealpha','texturemap',...
            'alphadatamapping', 'direct', 'facelighting', 'flat', 'parent', rendAxes, 'Visible', 'off', 'Clipping', 'Off');
    end
    
    %create y planes       
    for i = 1:sY
        
        cdat = zeros(sX, sZ);
        adat = zeros(sX, sZ);
        cdat(:,:,1) = squeeze(cdatAll(:,i,:));
        adat(:,:) = squeeze(adatAll(:,i,:));
        cdat = permute(cdat, [2 1 3]);
        adat = permute(adat, [2 1 3]);
        
        if rerender==1
            slices2(i) = surface([0 sX; 0 sX], [i i; i i], [0 0; sZ sZ]);
        end
        set(slices2(i), 'cdatamapping', 'direct', 'facecolor','texture', 'cdata', cdat,...
            'edgealpha',0,'alphadata',double(adat), 'facealpha','texturemap',...
            'alphadatamapping', 'direct', 'facelighting', 'flat', 'parent', rendAxes, 'Visible', 'off', 'Clipping', 'Off');
            
    end

    %create x planes
    for i = 1:sX
        
        cdat = zeros(sY, sZ);
        adat = zeros(sY, sZ);
        cdat(:,:,1) = squeeze(cdatAll(i,:,:));
        adat(:,:) = squeeze(adatAll(i,:,:));
        cdat = permute(cdat, [2 1 3]);
        adat = permute(adat, [2 1 3]);    
        
        if rerender==1
            slices3(i) = surface([i i; i i], [0 sY; 0 sY], [0 0; sZ sZ]);
        end
        set(slices3(i), 'cdatamapping', 'direct', 'facecolor','texture', 'cdata', cdat,...
            'edgealpha',0,'alphadata',double(adat), 'facealpha','texturemap',...
            'alphadatamapping', 'direct', 'facelighting', 'flat', 'parent', rendAxes, 'Visible', 'off', 'Clipping', 'Off');    
    end
    %daspect(rendAxes, [sX sY sZ]);
    daspect(rendAxes, [1 1 1]);
    
    set(rendAxes, 'color', 'black');
    disp('rendering done');
    
    clear('Xs', 'Ys', 'Zs');
    
    clear('cdat');
    clear('adat');
    
    clear('cdatAll');
    clear('adatAll');
    clear('CA');
    
    %additional formating
    [0 sX 0 sY 0 sZ]
    axis([0 sX 0 sY 0  sZ])
    
    %line([0 sX],[0 sY],[0 sZ]);
    
    
    %testing
    %set([slices1; slices2; slices3;], 'AlphaDataMapping', 'scaled');
    %set(rendAxes, 'ALimMode', 'manual');
    %set(rendAxes, 'ALim', [0 1]);
    
end
