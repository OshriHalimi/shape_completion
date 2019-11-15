function [ masterMap  ] = MakeMap( mapCells )
%MAKEMAP
    
%    {}     treat as single overiding colormap
%    {}{}{} treat as blend colormaps 

    numGroups = length(mapCells);
    
    masterMap = [];
    
    for g = 1:numGroups
        
        mapGroup = mapCells{g};
        numChan = length(mapGroup);
        
        newMap = CombineMap(mapGroup, numChan);
        
        masterMap = [masterMap; newMap];
        
        clear newMap;
        clear mapGroup;
    end
    
    clear mapCells;



end

