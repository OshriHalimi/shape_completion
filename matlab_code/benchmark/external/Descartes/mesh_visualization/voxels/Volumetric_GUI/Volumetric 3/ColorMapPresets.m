function [ presets, names ] = ColorMapPresets( color)
%ALPHAMAPPRESETS Summary of this function goes here
%   Detailed explanation goes here


    color2(1,:) = color; 

    %Shell
    names{1} = 'shell';
    x = ((1:64)/64)';
    y = (x).*(1-x);
    y = y/max(y);
    y = mdaTimes(y, color2, [], 'no');
    presets{1} = y;
    
    
    %Volume Shaded 
    names{2} = 'shaded volume';
    x = ((1:64)/64)';
    y = mdaTimes(((x)).^2, color2, [], 'no');
    presets{2} = y;
    

    
    

end

