function [ presets, names ] = AlphaMapPresets( )
%ALPHAMAPPRESETS Summary of this function goes here
%   Detailed explanation goes here


    %Shell
    names{1} = 'shell';
    x = ((1:64)/64)';
    y = ((x).*(1-x)).^2;
    presets{1} = y;
    
    
    %Volume Shaded 
    names{2} = 'shaded volume';
    x = ((1:64)/64)';
    y = ((x)).^2;
    presets{2} = y;
    

    
    
    
    
end

